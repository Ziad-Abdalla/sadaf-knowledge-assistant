from ska.arabic import is_arabic, normalize_light, normalize_search, numbers_in, tokenize


def test_light_normalization_removes_diacritics_tatweel_and_converts_digits():
    assert normalize_light("المِياومةُ ٣٥٠ ريالـــًا") == "المياومة 350 ريالا"


def test_search_normalization_folds_letter_variants():
    assert normalize_search("أبها إجازة آلة مستشفى المياومة") == "ابها اجازه اله مستشفي المياومه"


def test_tokenize_strips_article_and_prepositions_and_stopwords():
    toks = tokenize("وبالمياومة في المواقع النائية")
    assert "مياومه" in toks
    assert "في" not in toks
    assert "مواقع" in toks or "مواقع".startswith(toks[1][:4])


def test_same_stem_for_spelling_variants():
    assert tokenize("المياومة") == tokenize("مياومه")


def test_dialect_words_mapped_for_queries_only():
    assert "اريد" in tokenize("ابغى إجازة", query=True)
    assert "ابغي" in tokenize("ابغى إجازة", query=False)


def test_numbers_keep_amounts_comparable():
    assert numbers_in("الميزانية 18,500,000 ريال و ٣٥٠") == {"18500000", "350"}
    assert "0.80" in numbers_in("0.80 ريال لكل كيلومتر")


def test_language_detection():
    assert is_arabic("كم المياومة؟")
    assert not is_arabic("What is the per diem for Abha?")
