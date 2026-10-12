import shutil

from fastapi.testclient import TestClient

from ska import config
from ska.evaluation import check_gate, load_items, score_item, summarize
from ska.web import create_app

from .conftest import FakeAnthropic


# ------------------------------------------------------------------ releases and rollback
def test_rollback_switch_rewrites_only_the_active_line(tmp_path):
    path = tmp_path / "releases.yaml"
    shutil.copy(config.RELEASES_FILE, path)
    before = path.read_text(encoding="utf-8")
    config.set_active_release("2026-10-r1", path)
    after = path.read_text(encoding="utf-8")
    assert config.active_release(path).id == "2026-10-r1"
    assert [l for l in before.splitlines() if not l.startswith("active:")] == \
           [l for l in after.splitlines() if not l.startswith("active:")]


def test_env_var_pins_a_release(monkeypatch):
    monkeypatch.setenv("SKA_RELEASE", "2026-10-r1")
    assert config.active_release().prompt == "answer_v1"


def test_each_prompt_version_has_its_own_hash():
    _, rels = config.load_releases()
    assert rels["2026-10-r1"].prompt_hash != rels["2026-10-r2"].prompt_hash


# ------------------------------------------------------------------ eval scoring
def test_eval_set_is_big_and_mostly_arabic():
    items = load_items()
    assert len(items) >= 40
    assert sum(i.get("style") in ("msa", "dialect") for i in items) / len(items) > 0.75
    assert {i["type"] for i in items} == {"answerable", "not_in_docs", "access_denied",
                                         "injection", "conflict", "history"}


def test_leak_is_detected_by_value_even_without_a_citation(make_assistant, users):
    it = {"id": "x", "type": "access_denied", "expect": "refuse", "forbidden": ["FIN-BUDGET-RIMAL"],
          "exclude": ["18,500,000"]}
    def leaky(kw):
        sid = kw["messages"][0]["content"].split('id="')[1].split('"')[0]
        return {"status": "answered", "conflicts": [], "missing": "",
                "sentences": [{"text": "الميزانية 18,500,000", "sources": [sid]}]}
    a = make_assistant(client=FakeAnthropic(responder=leaky))
    ans = a.ask("كم المياومة؟", users["saad"])
    row = score_item(it, ans, users["saad"], a.index.docs)
    assert row["leak"] is True
    m = summarize([row])
    assert m["leak_rate"] == 1.0
    assert "leak_rate=1.0 > max 0" in check_gate(m, {"leak_rate": {"max": 0}})


# ------------------------------------------------------------------ web page
def test_chat_page_renders_rtl_with_role_switcher(make_assistant, users):
    app = create_app(make_assistant(), users)
    r = TestClient(app).get("/")
    assert r.status_code == 200 and 'dir="rtl"' in r.text
    assert "سعد" in r.text and "ريم" in r.text


def test_asking_shows_citations_and_refusal_reasons(make_assistant, users):
    client = TestClient(create_app(make_assistant(), users))
    r = client.post("/ask", data={"question": "كم المياومة اليومية داخل المملكة؟", "user_id": "saad"})
    assert "المصادر" in r.text and "الإصدار" in r.text

    def nope(kw):
        return {"status": "not_in_documents", "sentences": [], "conflicts": [], "missing": "x"}
    client = TestClient(create_app(make_assistant(client=FakeAnthropic(responder=nope)), users))
    r = client.post("/ask", data={"question": "هل يوجد موقف سيارات؟", "user_id": "saad"})
    assert "لماذا لا أستطيع الإجابة" in r.text


def test_audit_page_lists_questions(make_assistant, users):
    a = make_assistant()
    a.ask("كم المياومة؟", users["saad"])
    r = TestClient(create_app(a, users)).get("/admin/audit")
    assert r.status_code == 200 and "كم المياومة" in r.text


def test_question_as_a_link(make_assistant, users):
    r = TestClient(create_app(make_assistant(), users)).get(
        "/ask", params={"q": "كم المياومة اليومية داخل المملكة؟", "user_id": "saad"})
    assert r.status_code == 200 and "المصادر" in r.text


def test_history_csv_keeps_old_rows_aligned_when_columns_change(tmp_path):
    from ska.evaluation import HISTORY_FIELDS, append_history
    import csv
    p = tmp_path / "history.csv"
    p.write_text("ts,release_id,leak_rate\n1,r1,0.0\n", encoding="utf-8")
    append_history(p, {"ts": "2", "release_id": "r2", "leak_rate": 0.0, "git_commit": "abc"})
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    assert list(rows[0].keys()) == HISTORY_FIELDS
    assert rows[0]["release_id"] == "r1" and rows[1]["git_commit"] == "abc"
