from ska.grounding import (DERIVED, NO_CITATION, NUMBER_NOT_IN_SOURCE, UNKNOWN_SOURCE,
                           WEAK_OVERLAP, check_sentence)
from ska.redaction import Redactor

from .conftest import FakeAnthropic, sources_in

MEMO = ("1.1 الميزانية الإجمالية 18,500,000 ريال سعودي.\n2.1 أجر الممثل الرئيسي فهد العتيبي "
        "2,100,000 ريال سعودي.\n3.1 الاحتياطي 8% من الميزانية.\nرمز تتبع داخلي: CANARY-BUD-7731\n"
        "للتواصل finance-director@sadaf.example")


def test_redaction_masks_amounts_names_codes_and_emails():
    r = Redactor()
    out = r.redact(MEMO, ["فهد العتيبي"])
    for secret in ["18,500,000", "2,100,000", "فهد العتيبي", "CANARY-BUD-7731",
                   "finance-director@sadaf.example", "8%"]:
        assert secret not in out
    assert "[AMOUNT_1]" in out and "[NAME_1]" in out
    assert r.restore(out) == MEMO


def test_same_value_gets_the_same_placeholder_across_chunks():
    r = Redactor()
    a = r.redact("الميزانية 18,500,000 ريال")
    b = r.redact("تكرار: 18,500,000 ريال")
    assert "[AMOUNT_1]" in a and "[AMOUNT_1]" in b and "[AMOUNT_2]" not in b


def test_ordinary_small_numbers_are_kept():
    assert Redactor().redact("30 حلقة خلال 7 أيام") == "30 حلقة خلال 7 أيام"


def test_finance_question_on_budget_sends_no_real_amount_to_the_model(make_assistant, users):
    def responder(kw):
        sid, _, text = next(s for s in sources_in(kw) if "[AMOUNT_" in s[2])
        ph = next(tok for tok in text.split() if tok.startswith("[AMOUNT_"))
        return {"status": "answered", "conflicts": [], "missing": "",
                "sentences": [{"text": f"الميزانية الإجمالية المعتمدة {ph} ريال سعودي.",
                               "sources": [sid]}]}

    client = FakeAnthropic(responder=responder)
    a = make_assistant(client=client)
    ans = a.ask("كم الميزانية الإجمالية لمسلسل رمال الذاكرة؟", users["khalid"])
    prompt = client.sent[0]["messages"][0]["content"]
    assert "18,500,000" not in prompt and "CANARY-BUD-7731" not in prompt
    assert ans.redacted
    assert "18,500,000" in ans.text           # restored locally for the user
    assert not ans.sentences[0].flags


def test_local_only_policy_answers_without_calling_the_model(make_assistant, users):
    from .conftest import TEST_RELEASE
    rel = type(TEST_RELEASE)(**{**TEST_RELEASE.__dict__, "external_policy": "local_only"})
    client = FakeAnthropic()
    a = make_assistant(client=client, release=rel)
    ans = a.ask("كم الميزانية الإجمالية لمسلسل رمال الذاكرة؟", users["khalid"])
    assert client.sent == []
    assert ans.reason == "local_only_extractive"
    assert ans.citations[0].doc_id == "FIN-BUDGET-RIMAL"


SRC = {"S1": '<source id="S1" version="2.0" effective_date="2026-01-01">\n'
             "2.1 تُصرف المياومة بمبلغ 350 ريالًا سعوديًا يوميًا داخل المملكة.\n"
             "3.1 بدل مشقة إضافي قدره 100 ريال سعودي يوميًا للمواقع النائية.\n</source>"}


def test_supported_sentence_passes():
    assert check_sentence("تُصرف المياومة بمبلغ 350 ريالًا يوميًا داخل المملكة.", ["S1"], SRC) == []


def test_invented_number_is_flagged():
    assert NUMBER_NOT_IN_SOURCE in check_sentence("المياومة 400 ريال يوميًا داخل المملكة.", ["S1"], SRC)


def test_sum_of_source_numbers_is_marked_derived_not_wrong():
    flags = check_sentence("المجموع 350 + 100 = 450 ريالًا يوميًا للمواقع النائية.", ["S1"], SRC)
    assert DERIVED in flags and NUMBER_NOT_IN_SOURCE not in flags


def test_missing_or_unknown_citation_is_flagged():
    assert check_sentence("المياومة 350 ريال.", [], SRC) == [NO_CITATION]
    assert check_sentence("المياومة 350 ريال.", ["S7"], SRC) == [UNKNOWN_SOURCE]


def test_unrelated_sentence_has_weak_overlap():
    assert WEAK_OVERLAP in check_sentence("يحق للموظف استخدام النادي الرياضي مجانًا طوال العام.",
                                          ["S1"], SRC)


def test_flagged_sentence_is_marked_in_the_answer(make_assistant, users):
    def responder(kw):
        sid = sources_in(kw)[0][0]
        return {"status": "answered", "conflicts": [], "missing": "",
                "sentences": [{"text": "المياومة 999 ريال.", "sources": [sid]}]}
    a = make_assistant(client=FakeAnthropic(responder=responder))
    ans = a.ask("كم المياومة داخل المملكة؟", users["saad"])
    assert NUMBER_NOT_IN_SOURCE in ans.sentences[0].flags
    assert "⚠" in ans.text
    assert a.audit.recent(1)[0]["flagged_sentences"] == 1
