"""The cheap post-check: is every sentence of the answer backed by the source it cites?

No second model call. For each sentence:
- it must cite at least one source, and only sources we actually gave the model;
- every number in it must appear in the cited source (or its metadata line: version and
  date) or in the question itself. The one exception is a sum the sentence shows: in
  "350 + 100 = 450", 350 and 100 are in the source and in the sentence, so 450 is marked
  "derived" (fine, but worth showing). A number that is neither, like 700 or 400 against a
  350 source, is flagged;
- if the sentence and the source are in the same language, a reasonable share of its content
  words must appear in the source.
Anything flagged with a WARN_FLAGS flag is still shown, but with a warning, and it is logged
and counted in the eval.
Numbers are where a confident wrong answer hurts most (amounts, days, deadlines), so the
number rule is strict and the word-overlap rule is soft.
"""

from itertools import combinations

from .arabic import is_arabic, numbers_in, tokenize

MIN_OVERLAP = 0.2
NO_CITATION, UNKNOWN_SOURCE, NUMBER_NOT_IN_SOURCE, WEAK_OVERLAP, DERIVED = (
    "no_citation", "unknown_source", "number_not_in_source", "weak_overlap", "derived_number")
# These flags put a warning next to the sentence. Nothing is hidden; the reader is warned.
WARN_FLAGS = {NO_CITATION, UNKNOWN_SOURCE, NUMBER_NOT_IN_SOURCE}


def _same(a: str, b: str) -> bool:
    return a == b or a.lstrip("0") == b.lstrip("0")


def _shown_sum(n: str, operands: set[str]) -> bool:
    """True if n is the sum of two different numbers that the sentence itself shows."""
    try:
        target = float(n)
        values = {float(x) for x in operands}
    except ValueError:
        return False
    return any(a + b == target for a, b in combinations(values, 2))


def check_sentence(text: str, cited: list[str], sources: dict[str, str],
                   question: str = "") -> list[str]:
    """`sources` maps source id (S1..) to the exact text the model saw for it."""
    if not cited:
        return [NO_CITATION]
    if any(c not in sources for c in cited):
        return [UNKNOWN_SOURCE]
    support = "\n".join(sources[c] for c in cited)
    flags = []

    allowed = numbers_in(support) | numbers_in(question)
    in_sentence = numbers_in(text)
    supported = {n for n in in_sentence if any(_same(n, s) for s in allowed)}
    for n in in_sentence - supported:
        if _shown_sum(n, supported):
            flags.append(DERIVED)
        else:
            flags.append(NUMBER_NOT_IN_SOURCE)
            break

    if is_arabic(text) == is_arabic(support):
        words = set(tokenize(text))
        if len(words) >= 4:
            overlap = len(words & set(tokenize(support))) / len(words)
            if overlap < MIN_OVERLAP:
                flags.append(WEAK_OVERLAP)
    return sorted(set(flags))
