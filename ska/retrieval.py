"""Hybrid retrieval: meaning search + keyword search, fused with Reciprocal Rank Fusion.

Order of operations matters:
1. Build the candidate set: chunks the user may read AND that are in the right version
   state. Everything else is removed here, before any scoring, so a forbidden chunk cannot
   be ranked, cannot reach the prompt, and cannot be leaked by a clever question.
2. Score only the candidates, two ways:
   - dense: cosine similarity between the question's embedding and each chunk's embedding
     (finds "بدل يومي" when the policy says "مياومة");
   - BM25: classic keyword scoring on normalized Arabic tokens (finds exact things like
     "ProRes", "LUFS", "GACA", a doc number or an amount).
3. Fuse the two rankings with RRF: each chunk gets sum(1 / (k + rank)) over both lists.
   It uses ranks only, so we never have to make cosine and BM25 scores comparable.
"""

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from .access import User, can_read
from .arabic import tokenize
from .chunking import Chunk
from .documents import Document
from .embeddings import Embedder
from .index import Index
from .versions import CURRENT, SUPERSEDED, statuses

POOL = 20  # how many results each method contributes before fusion


@dataclass
class Hit:
    chunk: Chunk
    doc: Document
    status: str
    dense: float
    bm25: float
    rrf: float
    dense_rank: int | None
    bm25_rank: int | None


@dataclass
class RetrievalResult:
    hits: list[Hit]
    best_dense: float
    candidates: int
    blocked_by_access: int          # chunk counts only; never which documents
    blocked_by_version: int
    history_mode: bool
    restricted_matches: list[str] = field(default_factory=list)  # for the audit log only


def bm25_scores(query_tokens: list[str], docs_tokens: list[list[str]],
                k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    """Okapi BM25 over the given chunks only (statistics come from the candidate set, so
    forbidden documents don't even influence the word weights)."""
    n = len(docs_tokens)
    if n == 0 or not query_tokens:
        return np.zeros(n, dtype=np.float32)
    avgdl = sum(len(t) for t in docs_tokens) / n or 1.0
    df = Counter()
    for toks in docs_tokens:
        df.update(set(toks))
    scores = np.zeros(n, dtype=np.float32)
    for i, toks in enumerate(docs_tokens):
        tf = Counter(toks)
        dl = len(toks)
        s = 0.0
        for q in set(query_tokens):
            if q not in tf:
                continue
            idf = math.log(1 + (n - df[q] + 0.5) / (df[q] + 0.5))
            s += idf * tf[q] * (k1 + 1) / (tf[q] + k1 * (1 - b + b * dl / avgdl))
        scores[i] = s
    return scores


def candidate_mask(index: Index, user: User, *, history: bool, today: date | None = None):
    st = statuses(list(index.docs.values()), today)
    allowed = np.array([can_read(user, d) for d in index.chunk_doc], dtype=bool)
    ok_status = {CURRENT, SUPERSEDED} if history else {CURRENT}
    version_ok = np.array([st[d.doc_id] in ok_status for d in index.chunk_doc], dtype=bool)
    return allowed, version_ok, st


def search(index: Index, embedder: Embedder, question: str, user: User, *, top_k: int = 6,
           rrf_k: int = 60, history: bool = False, today: date | None = None,
           audit_threshold: float | None = None) -> RetrievalResult:
    allowed, version_ok, st = candidate_mask(index, user, history=history, today=today)
    mask = allowed & version_ok
    cand = np.flatnonzero(mask)

    qvec = embedder.encode_query(question)
    dense_all = index.matrix @ qvec

    restricted = []
    if audit_threshold is not None:
        # Audit only: which forbidden documents *would* have matched. This list goes to the
        # security log and is never shown to the user or sent to the model.
        forbidden = np.flatnonzero(~allowed & (dense_all >= audit_threshold))
        restricted = sorted({index.chunks[i].doc_id for i in forbidden})

    if len(cand) == 0:
        return RetrievalResult([], 0.0, 0, int((~allowed).sum()), int((allowed & ~version_ok).sum()),
                               history, restricted)

    dense = dense_all[cand]
    bm25 = bm25_scores(tokenize(question, query=True), [index.tokens[i] for i in cand])

    dense_order = np.argsort(-dense)[:POOL]
    bm25_order = [i for i in np.argsort(-bm25)[:POOL] if bm25[i] > 0]
    dense_rank = {int(i): r + 1 for r, i in enumerate(dense_order)}
    bm25_rank = {int(i): r + 1 for r, i in enumerate(bm25_order)}
    fused = Counter()
    for i, r in dense_rank.items():
        fused[i] += 1.0 / (rrf_k + r)
    for i, r in bm25_rank.items():
        fused[i] += 1.0 / (rrf_k + r)

    hits = []
    for i, score in fused.most_common(top_k):
        gi = int(cand[i])
        doc = index.chunk_doc[gi]
        hits.append(Hit(chunk=index.chunks[gi], doc=doc, status=st[doc.doc_id],
                        dense=float(dense[i]), bm25=float(bm25[i]), rrf=float(score),
                        dense_rank=dense_rank.get(i), bm25_rank=bm25_rank.get(i)))
    return RetrievalResult(hits=hits, best_dense=float(dense.max()), candidates=len(cand),
                           blocked_by_access=int((~allowed).sum()),
                           blocked_by_version=int((allowed & ~version_ok).sum()),
                           history_mode=history, restricted_matches=restricted)
