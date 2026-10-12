"""Arabic text normalization and tokenization.

Two levels:
- normalize_light: what we embed and show. Removes diacritics (tashkeel) and the tatweel
  stretch character, and turns Arabic-Indic digits into 0-9. Letters stay as written.
- normalize_search: for keyword search (BM25) only. Also folds letter variants that people
  type interchangeably (أ إ آ -> ا, ى -> ي, ة -> ه, ؤ -> و, ئ -> ي), so "المياومة" and
  "المياومه" match.
"""

import re
import unicodedata

TASHKEEL = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
TATWEEL = "ـ"
DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
LETTER_FOLD = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي",
                             "ة": "ه", "ؤ": "و", "ئ": "ي"})
ARABIC_CHAR = re.compile(r"[؀-ۿ]")
TOKEN = re.compile(r"[\w]+", re.UNICODE)

# Definite article and attached prepositions, longest first. We only strip them from words
# long enough that what is left is still a word.
PREFIXES = ("وبال", "وال", "بال", "كال", "فال", "لل", "ال")
SUFFIXES = ("ات", "ون", "ين")

STOPWORDS = set("""
في من على الى عن مع او ثم هل ما ماذا متى كيف كم اين اي هذا هذه ذلك تلك التي الذي الذين
هو هي هم انا نحن انت انتم كان كانت يكون تكون لا لم لن قد كل بعد قبل عند بين اذا ان لو
و ف ب ل يا عندي عندنا لي لنا له لها
the a an of to in on for and or is are was be by with at as it this that what how when
which who do does can i my we our you your
""".split())

# A few common Saudi/Gulf dialect words mapped to the MSA word a policy would use.
# Applied to queries only, before keyword search.
DIALECT = {
    "وش": "ما", "ايش": "ما", "شنو": "ما",
    "ابي": "اريد", "ابغى": "اريد", "ابغي": "اريد", "ابغا": "اريد", "نبي": "نريد",
    "يعطوني": "يصرف", "يعطونا": "يصرف", "فلوس": "مبلغ", "الشغل": "العمل", "شغل": "عمل",
    "زين": "جيد", "وين": "اين", "ليش": "لماذا", "كذا": "هكذا",
    "اطلع": "اسافر", "رحت": "سافرت", "نروح": "نسافر", "يروح": "يسافر",
}


def normalize_light(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = TASHKEEL.sub("", text).replace(TATWEEL, "")
    return text.translate(DIGITS)


def normalize_search(text: str) -> str:
    return normalize_light(text).translate(LETTER_FOLD).lower()


def _stem(token: str) -> str:
    if not ARABIC_CHAR.match(token):
        return token
    for p in PREFIXES:
        if token.startswith(p) and len(token) - len(p) >= 2:
            token = token[len(p):]
            break
    for s in SUFFIXES:
        if token.endswith(s) and len(token) - len(s) >= 3:
            token = token[: -len(s)]
            break
    return token


def tokenize(text: str, *, query: bool = False) -> list[str]:
    """Tokens for BM25: normalized, light-stemmed, stopwords removed.
    Numbers are kept, because amounts and versions matter in policies."""
    words = TOKEN.findall(normalize_search(text))
    if query:
        words = [DIALECT.get(w, w) for w in words]
    out = []
    for w in words:
        if w in STOPWORDS or (len(w) < 2 and not w.isdigit()):
            continue
        stem = _stem(w)
        if stem and stem not in STOPWORDS:
            out.append(stem)
    return out


def is_arabic(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return sum(1 for c in letters if ARABIC_CHAR.match(c)) / len(letters) > 0.3


NUMBER = re.compile(r"\d[\d,٬]*(?:\.\d+)?")


def numbers_in(text: str) -> set[str]:
    """All numbers in a text, with thousands separators removed: '18,500,000' -> '18500000'."""
    found = set()
    for m in NUMBER.findall(normalize_light(text)):
        n = m.replace(",", "").replace("٬", "").rstrip(".")
        if n.endswith(".0"):
            n = n[:-2]
        found.add(n)
    return found
