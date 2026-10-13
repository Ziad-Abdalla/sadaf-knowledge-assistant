"""The whole question-to-answer path, in order. Each step is a small function elsewhere.

  question + logged-in user
    -> history intent? (versions.asks_for_history)
    -> hybrid search over ONLY the chunks this user may read, in the right version state
    -> relevance gate: nothing close enough -> "not in the documents", no model call
    -> mask highly confidential values (or answer them locally)
    -> one Claude Haiku call, structured output, sentences with source ids
    -> post-check every sentence against its sources, restore masked values
    -> conflicts: pick the authoritative source, tell the user, flag the owner
    -> audit row (release, model, prompt hash, retrieved ids, decision, cost)
"""

import time
import uuid
from dataclasses import dataclass, field
from datetime import date

import anthropic
from agent_cost_tracker import BudgetExceeded

from . import config
from .access import User, can_read
from .arabic import is_arabic, normalize_light
from .audit import AuditLog
from .embeddings import Embedder, load_embedder
from .generation import GenerationError, generate, source_block
from .grounding import WARN_FLAGS, check_sentence
from .index import Index, index_path
from .redaction import Redactor
from .retrieval import Hit, search
from .versions import CURRENT, SUPERSEDED, asks_for_history, statuses

TEXT = {
    "ar": {
        "no_source": "لم أجد إجابة لهذا السؤال في المستندات المتاحة لك، لذلك لن أخمّن.",
        "not_in_docs": "المستندات التي وجدتها لا تحتوي على إجابة هذا السؤال، لذلك لن أخمّن.",
        "ask_owner": "إذا كنت تعتقد أن المعلومة موجودة في مستند لا تملك صلاحيته، تواصل مع مديرك "
                     "أو مع الجهة المالكة للمستند.",
        "error": "تعذّر إنشاء الإجابة الآن. لم يُعرض عليك أي نص غير متحقق منه. حاول مرة أخرى لاحقًا.",
        "used": "اعتمدت الإجابة على: «{title}» الإصدار {version}، الساري من {date}.",
        "older": "توجد نسخة أقدم من «{title}» (الإصدار {version}، {date}) ولم أعتمد عليها لأنها "
                 "استُبدلت.",
        "history": "سؤالك عن نسخ سابقة، لذلك عرضت النسخ القديمة والحالية مع تاريخ كل منها. "
                   "النسخة المعمول بها اليوم هي «{title}» الإصدار {version}.",
        "conflict": "تنبيه: يوجد تعارض بين «{a}» ({a_type}) و«{b}» ({b_type}). المعتمد هو «{a}» "
                    "لأنه {why}. أُرسل تنبيه لمراجعة التعارض إلى: {owner}.",
        "why_type": "مستند رسمي أعلى مرتبة من {b_type}",
        "why_date": "الأحدث تاريخًا",
        "local_only": "هذا المستند سري للغاية، لذلك لم يُرسل إلى أي نموذج خارجي. هذه المقاطع "
                      "المطابقة من النص الأصلي:",
        "redacted": "أُخفيت المبالغ والأسماء قبل إرسال النص للنموذج، ثم أُعيدت هنا محليًا.",
    },
    "en": {
        "no_source": "I couldn't find an answer to this in the documents available to you, so I "
                     "won't guess.",
        "not_in_docs": "The documents I found don't contain the answer to this, so I won't guess.",
        "ask_owner": "If you think the information is in a document you don't have access to, "
                     "ask your manager or the document's owner.",
        "error": "I couldn't produce an answer right now. Nothing unverified was shown. Please "
                 "try again later.",
        "used": "Based on: \"{title}\" version {version}, effective {date}.",
        "older": "An older version of \"{title}\" exists (version {version}, {date}); I did not "
                 "use it because it was replaced.",
        "history": "You asked about earlier versions, so old and current versions are shown "
                   "with their dates. The one in force today is \"{title}\" version {version}.",
        "conflict": "Note: \"{a}\" ({a_type}) and \"{b}\" ({b_type}) disagree. \"{a}\" applies "
                    "because it is {why}. A review flag was sent to: {owner}.",
        "why_type": "an official document that outranks {b_type}",
        "why_date": "the newer one",
        "local_only": "This document is highly confidential, so it was not sent to any external "
                      "model. Matching passages from the original:",
        "redacted": "Amounts and names were masked before the model saw the text, then restored "
                    "here locally.",
    },
}
TYPE_NAMES = {"ar": {"policy": "سياسة", "spec": "مواصفات", "guide": "دليل", "template": "نموذج",
                     "memo": "مذكرة", "minutes": "محضر اجتماع"},
              "en": {}}


@dataclass
class Citation:
    n: int
    chunk_id: str
    doc_id: str
    title: str
    section: str
    section_title: str
    version: str
    effective_date: str
    status: str
    doc_type: str
    odoo_link: str | None
    quote: str

    def label(self, lang: str = "ar") -> str:
        if lang == "ar":
            return (f"{self.title}، القسم {self.section} «{self.section_title}»، "
                    f"الإصدار {self.version} ({self.effective_date})")
        return (f"{self.title}, section {self.section} \"{self.section_title}\", "
                f"version {self.version} ({self.effective_date})")


@dataclass
class AnswerSentence:
    text: str
    citations: list[int]
    flags: list[str] = field(default_factory=list)


@dataclass
class Answer:
    request_id: str
    question: str
    lang: str
    status: str                      # answered | partial | refused | error
    reason: str = ""
    message: str = ""
    sentences: list[AnswerSentence] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)
    missing: str = ""
    history_mode: bool = False
    redacted: bool = False
    best_dense: float = 0.0
    release_id: str = ""
    model: str = ""
    prompt_hash: str = ""
    cost_usd: float = 0.0
    latency_ms: int = 0

    @property
    def text(self) -> str:
        """Plain text with inline citations, for the CLI and the eval."""
        if self.status in ("refused", "error"):
            return self.message
        cites = {c.n: c for c in self.citations}
        parts = []
        for s in self.sentences:
            refs = " ".join(f"[{cites[n].label(self.lang)}]" for n in s.citations if n in cites)
            warn = " ⚠" if set(s.flags) & WARN_FLAGS else ""
            parts.append(f"{s.text} {refs}{warn}".strip())
        return "\n".join(parts + self.notices)


class Assistant:
    def __init__(self, index: Index, embedder: Embedder, release: config.Release,
                 audit: AuditLog, client=None, today: date | None = None):
        if index.embed_model != embedder.name:
            raise ValueError(f"Index was built with {index.embed_model}, release uses "
                             f"{embedder.name}. Re-run ingest.")
        self.index, self.embedder, self.release, self.audit = index, embedder, release, audit
        self._client = client
        self.today = today

    @classmethod
    def from_config(cls, release: config.Release | None = None) -> "Assistant":
        release = release or config.active_release()
        index = Index.load(index_path(config.INDEX_DIR, release.embed_model))
        return cls(index, load_embedder(release.embed_model), release,
                   AuditLog(config.AUDIT_DB))

    @property
    def client(self):
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    # ------------------------------------------------------------------ main path
    def ask(self, question: str, user: User, force_history: bool = False) -> Answer:
        t0 = time.perf_counter()
        rel = self.release
        question = normalize_light(question.strip())
        lang = "ar" if is_arabic(question) else "en"
        T = TEXT[lang]
        history = force_history or asks_for_history(question)
        ans = Answer(request_id=uuid.uuid4().hex[:12], question=question, lang=lang,
                     status="refused", history_mode=history, release_id=rel.id,
                     model=rel.llm_model, prompt_hash=rel.prompt_hash)

        result = search(self.index, self.embedder, question, user, top_k=rel.top_k,
                        history=history, today=self.today,
                        audit_threshold=rel.min_dense)
        ans.best_dense = result.best_dense
        hits = result.hits
        log = dict(request_id=ans.request_id, user_id=user.user_id, roles=",".join(user.roles),
                   question=question, release_id=rel.id, model=rel.llm_model,
                   prompt_version=rel.prompt, prompt_hash=rel.prompt_hash,
                   embed_model=rel.embed_model, history_mode=int(history),
                   candidates=result.candidates, blocked_by_access=result.blocked_by_access,
                   blocked_by_version=result.blocked_by_version,
                   retrieved=[{"chunk_id": h.chunk.chunk_id, "doc_id": h.doc.doc_id,
                               "version": h.doc.version, "status": h.status,
                               "dense": round(h.dense, 4), "bm25": round(h.bm25, 3),
                               "rrf": round(h.rrf, 5)} for h in hits],
                   restricted_matches=result.restricted_matches,
                   best_dense=round(result.best_dense, 4), redacted=0)

        # 1. Relevance gate: no model call when nothing relevant is visible to this user.
        if not hits or result.best_dense < rel.min_dense:
            ans.reason = "no_relevant_source"
            ans.message = f"{T['no_source']} {T['ask_owner']}"
            return self._finish(ans, log, t0)

        # 2. Confidential handling, then the prompt sources.
        secret = [h for h in hits if h.doc.confidentiality == "highly_confidential"]
        if secret and rel.external_policy == "local_only":
            if hits[0] is secret[0]:
                # The best match is highly confidential: show its passages, no model call.
                return self._finish(self._extractive(ans, secret, T), log, t0)
            # Otherwise drop the confidential hits and send the rest to the model as usual.
            hits = [h for h in hits if h.doc.confidentiality != "highly_confidential"]
        redactor = Redactor()
        sources, blocks = {}, []
        for i, h in enumerate(hits, 1):
            text = h.chunk.text
            if h.doc.confidentiality == "highly_confidential" and rel.external_policy == "redact":
                text = redactor.redact(text, h.doc.sensitive_terms)
            block = source_block(f"S{i}", h, text)
            sources[f"S{i}"] = block
            blocks.append(block)
        ans.redacted = bool(redactor.mapping)
        log["redacted"] = int(ans.redacted)

        # 3. The model call.
        try:
            gen = generate(self.client, rel, question, blocks, ans.request_id, lang)
        except (GenerationError, BudgetExceeded, anthropic.APIError) as e:
            ans.status, ans.message = "error", T["error"]
            ans.reason = getattr(e, "reason", type(e).__name__)
            log["error"] = f"{type(e).__name__}: {e}"
            return self._finish(ans, log, t0)
        ans.model = gen.model
        ans.cost_usd = gen.cost_usd
        log.update(gen.usage, cost_usd=gen.cost_usd, stop_reason=gen.stop_reason, model=gen.model)
        out = gen.answer

        if out.status == "not_in_documents" or not out.sentences:
            ans.reason = "model_not_in_documents"
            ans.missing = out.missing
            ans.message = f"{T['not_in_docs']} {T['ask_owner']}"
            return self._finish(ans, log, t0)

        # 4. Post-check each sentence, then restore masked values and number the citations.
        sid_to_n: dict[str, int] = {}
        for s in out.sentences:
            flags = check_sentence(s.text, s.sources, sources, question)
            nums = []
            for sid in s.sources:
                if sid in sources:
                    sid_to_n.setdefault(sid, len(sid_to_n) + 1)
                    nums.append(sid_to_n[sid])
            ans.sentences.append(AnswerSentence(redactor.restore(s.text), nums, flags))
        hit_by_sid = {f"S{i}": h for i, h in enumerate(hits, 1)}
        ans.citations = [self._citation(n, hit_by_sid[sid]) for sid, n in sid_to_n.items()]
        ans.status = "partial" if out.status == "partial" else "answered"
        ans.missing = out.missing

        # 5. Conflicts and version notices.
        self._conflicts(ans, out.conflicts, hit_by_sid, T, lang)
        self._version_notices(ans, user, T)
        if ans.redacted:
            ans.notices.append(T["redacted"])
        return self._finish(ans, log, t0)

    # ------------------------------------------------------------------ helpers
    def _citation(self, n: int, h: Hit) -> Citation:
        d = h.doc
        return Citation(n=n, chunk_id=h.chunk.chunk_id, doc_id=d.doc_id, title=d.title,
                        section=h.chunk.section, section_title=h.chunk.section_title,
                        version=d.version, effective_date=d.effective_date.isoformat(),
                        status=h.status, doc_type=d.doc_type, odoo_link=d.odoo_link,
                        quote=h.chunk.text)

    def _extractive(self, ans: Answer, secret: list[Hit], T: dict) -> Answer:
        ans.status, ans.reason = "answered", "local_only_extractive"
        ans.notices.append(T["local_only"])
        for i, h in enumerate(secret[:2], 1):
            ans.citations.append(self._citation(i, h))
            ans.sentences.append(AnswerSentence(h.chunk.text, [i]))
        return ans

    def _conflicts(self, ans: Answer, conflicts, hit_by_sid, T, lang) -> None:
        names = TYPE_NAMES[lang]
        for c in conflicts:
            docs = {hit_by_sid[s].doc.doc_id: hit_by_sid[s].doc for s in c.sources
                    if s in hit_by_sid}
            if len(docs) < 2:
                continue
            if any(d.supersedes in docs for d in docs.values()):
                continue          # two versions of one policy differ on purpose; not a conflict
            if any(set(x["docs"]) == set(docs) for x in ans.conflicts):
                continue          # the model reported the same pair twice
            ranked = sorted(docs.values(), key=lambda d: (d.authority, d.effective_date),
                            reverse=True)
            a, b = ranked[0], ranked[1]
            why = (T["why_type"].format(b_type=names.get(b.doc_type, b.doc_type))
                   if a.authority > b.authority else T["why_date"])
            owners = sorted({d.owner for d in ranked if d.owner})
            ans.notices.append(T["conflict"].format(
                a=a.title, a_type=names.get(a.doc_type, a.doc_type), b=b.title,
                b_type=names.get(b.doc_type, b.doc_type), why=why, owner=", ".join(owners)))
            ans.conflicts.append({"topic": c.topic, "preferred": a.doc_id,
                                  "docs": [d.doc_id for d in ranked], "note": c.note})
            for owner in owners:
                self.audit.flag(request_id=ans.request_id, kind="conflict", owner=owner,
                                doc_ids=[d.doc_id for d in ranked], topic=c.topic,
                                detail=f"Q: {ans.question} | {c.note}")

    def _version_notices(self, ans: Answer, user: User, T: dict) -> None:
        docs = self.index.docs
        st = statuses(list(docs.values()), self.today)
        # A document that lost a conflict is cited, but the answer isn't "based on" it.
        seen = {d for x in ans.conflicts for d in x["docs"] if d != x["preferred"]}
        for c in ans.citations:
            if c.doc_id in seen:
                continue
            seen.add(c.doc_id)
            if ans.history_mode:
                continue
            ans.notices.append(T["used"].format(title=c.title, version=c.version,
                                                date=c.effective_date))
            old = docs[c.doc_id].supersedes
            if old and st.get(old) == SUPERSEDED and can_read(user, docs[old]):
                o = docs[old]
                ans.notices.append(T["older"].format(title=o.title, version=o.version,
                                                     date=o.effective_date.isoformat()))
        if ans.history_mode:
            current = [docs[c.doc_id] for c in ans.citations if st[c.doc_id] == CURRENT]
            if current:
                ans.notices.append(T["history"].format(title=current[0].title,
                                                       version=current[0].version))

    def _finish(self, ans: Answer, log: dict, t0: float) -> Answer:
        ans.latency_ms = int((time.perf_counter() - t0) * 1000)
        flagged = [s for s in ans.sentences if set(s.flags) & WARN_FLAGS]
        self.audit.record(**log, decision=ans.status if ans.status != "partial" else "answered",
                          reason=ans.reason, cited_docs=sorted({c.doc_id for c in ans.citations}),
                          flagged_sentences=len(flagged),
                          flags=sorted({f for s in ans.sentences for f in s.flags}),
                          conflicts=ans.conflicts, latency_ms=ans.latency_ms)
        return ans
