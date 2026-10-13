from datetime import date

from ska.retrieval import bm25_scores, search
from ska.versions import CURRENT, NOT_YET, SUPERSEDED, asks_for_history, statuses


def test_old_per_diem_policy_is_superseded(index):
    st = statuses(list(index.docs.values()), date(2026, 10, 12))
    assert st["FIN-PERDIEM-2025"] == SUPERSEDED
    assert st["FIN-PERDIEM-2026"] == CURRENT


def test_a_future_version_does_not_replace_the_current_one_yet(index):
    st = statuses(list(index.docs.values()), date(2025, 6, 1))
    assert st["FIN-PERDIEM-2026"] == NOT_YET
    assert st["FIN-PERDIEM-2025"] == CURRENT


def test_history_intent_detection():
    assert asks_for_history("كم كانت المياومة في 2025؟")
    assert asks_for_history("وش تغير في سياسة المياومة؟")
    assert asks_for_history("What changed from the previous version?")
    assert not asks_for_history("كم المياومة في أبها؟")
    assert not asks_for_history("قبل كم يوم أطلب الإجازة؟")


def test_default_search_never_returns_superseded_version(index, embedder, users):
    r = search(index, embedder, "كم المياومة اليومية داخل المملكة", users["saad"], top_k=10)
    ids = {h.doc.doc_id for h in r.hits}
    assert "FIN-PERDIEM-2025" not in ids
    assert "FIN-PERDIEM-2026" in ids
    assert r.blocked_by_version > 0


def test_history_mode_returns_both_versions_labelled(index, embedder, users):
    r = search(index, embedder, "المياومة اليومية داخل المملكة", users["saad"], top_k=10,
               history=True)
    by_id = {h.doc.doc_id: h.status for h in r.hits}
    assert by_id.get("FIN-PERDIEM-2025") == SUPERSEDED
    assert by_id.get("FIN-PERDIEM-2026") == CURRENT


def test_bm25_prefers_rare_matching_terms():
    docs = [["مياومه", "ريال"], ["اجازه", "سنويه"], ["مياومه", "مياومه", "سكن"]]
    s = bm25_scores(["مياومه"], docs)
    assert s[1] == 0 and s[2] > s[0] > 0


def test_keyword_leg_finds_exact_technical_terms(index, embedder, users):
    r = search(index, embedder, "ProRes 422 HQ", users["saad"], top_k=3)
    assert r.hits[0].doc.doc_id == "TECH-DELIVERY-BROADCAST"
    assert r.hits[0].bm25_rank == 1


def test_hits_carry_fusion_details(index, embedder, users):
    r = search(index, embedder, "إرجاع المعدات بعد التصوير", users["saad"], top_k=5)
    h = r.hits[0]
    assert h.rrf > 0 and (h.dense_rank or h.bm25_rank)
    assert [x.rrf for x in r.hits] == sorted([x.rrf for x in r.hits], reverse=True)


def test_english_question_finds_arabic_policy_through_the_glossary(index, embedder, users):
    from ska.arabic import expand_query, tokenize
    assert "مياومة" in expand_query("What is the per diem in Abha?")
    assert "مياومه" in tokenize("What is the per diem in Abha?", query=True)
    q = tokenize("What is the daily per diem for the crew shooting in Abha?", query=True)
    scores = bm25_scores(q, index.tokens)
    top = {index.chunks[i].doc_id for i in scores.argsort()[::-1][:3]}
    assert "FIN-PERDIEM-2026" in top          # the keyword leg now finds the Arabic policy


def test_history_mode_gets_room_for_both_versions(index, embedder, users):
    r = search(index, embedder, "سقف الفندق", users["saad"], top_k=6, history=True)
    assert len(r.hits) == 12
