"""The cheap post-check: is every sentence of the answer backed by the source it cites?

No second model call. For each sentence:
- it must cite at least one source, and only sources we actually gave the model;
- every number in it must appear in the cited source (or its metadata line: version and
  date), or be a simple sum/difference/product of two numbers that do (then it is marked
  "derived", which is fine but worth showing, e.g. 350 + 100 = 450 per day in Abha);
- if the sentence and the source are in the same language, a reasonable share of its content
  words must appear in the source.
Anything that fails is shown to the user with a warning, logged, and counted in the eval.
Numbers are where a confident wrong answer hurts most (amounts, days, deadlines), so the
number rule is strict and the word-overlap rule is soft.
"""

from itertools import product

from .arabic import is_arabic, numbers_in, tokenize

MIN_OVERLAP = 0.2
NO_CITATION, UNKNOWN_SOURCE, NUMBER_NOT_IN_SOURCE, WEAK_OVERLAP, DERIVED = (
    "no_citation", "unknown_source", "number_not_in_source", "weak_overlap", "derived_number")
BLOCKING = {NO_CITATION, UNKNOWN_SOURCE, NUMBER_NOT_IN_SOURCE}


def _derivable(n: str, source_numbers: set[str]) -> bool:
    try:
        target = float(n)
        values = [float(x) for x in source_numbers]
    except ValueError:
        return False
    for a, b in product(values, repeat=2):
        if target in (a + b, a - b, a * b):
            return True
    return False


def check_sentence(text: str, cited: list[str], sources: dict[str, str]) -> list[str]:
    """`sources` maps source id (S1..) to the exact text the model saw for it."""
    if not cited:
        return [NO_CITATION]
    if any(c not in sources for c in cited):
        return [UNKNOWN_SOURCE]
    support = "\n".join(sources[c] for c in cited)
    flags = []

    source_numbers = numbers_in(support)
    for n in numbers_in(text):
        if n in source_numbers or n.lstrip("0") in {s.lstrip("0") for s in source_numbers}:
            continue
        if _derivable(n, source_numbers):
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
