"""Masking before an external model sees highly confidential text (form question 54).

For documents marked `highly_confidential` the release chooses one of:
- "redact" (default): amounts, names listed by the document owner in `sensitive_terms`,
  emails, phone numbers, IBANs and tracking codes are swapped for placeholders like
  [AMOUNT_1] before the call. The model answers with the placeholders, and we put the real
  values back locally. The external model never sees "18,500,000".
- "local_only": when the best match is highly confidential, the external model is not called
  at all. The user gets the matching passages verbatim, with citations. Highly confidential
  chunks lower down the list are dropped, and the rest go to the model as usual.
Any other value stops the app when the releases file is loaded (ska/config.py).

Honest limit: regexes and an owner-maintained term list will miss things a person would
catch (an unusual name, an amount written in words). That's why the strict options are
local_only or a self-hosted model.
"""

import re
from dataclasses import dataclass, field

from .arabic import normalize_light

CURRENCY = r"(?:ريال(?:\s*سعودي)?|ر\.س|SAR|SR|﷼)"
# (kind, pattern). When a pattern has a group, only the group is masked, so the currency
# word stays readable for the model: "[AMOUNT_1] ريال سعودي".
PATTERNS = [
    # Tracking codes. In this corpus they are canary tokens planted for the leak tests.
    ("CODE", re.compile(r"CANARY-[A-Z]{3}-\d{3,}")),
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("IBAN", re.compile(r"\bSA\d{2}[\dA-Z ]{18,26}\b")),
    ("PHONE", re.compile(r"(?:\+966|00966|\b05)\d[\d ]{7,10}\d")),
    ("AMOUNT", re.compile(rf"(\d[\d,٬]*(?:\.\d+)?)(?=\s*(?:مليون|ألف|million|M|k)?\s*{CURRENCY})")),
    ("AMOUNT", re.compile(rf"{CURRENCY}\s*(\d[\d,٬]*(?:\.\d+)?)")),
    ("AMOUNT", re.compile(r"(?<![\w\[])\d{1,3}(?:[,٬]\d{3})+(?:\.\d+)?\b|(?<![\w\[])\d{5,}\b")),
    ("PERCENT", re.compile(r"\d+(?:\.\d+)?\s*[%٪]")),
]


@dataclass
class Redactor:
    """One per request, so the same value gets the same placeholder in every chunk."""
    mapping: dict[str, str] = field(default_factory=dict)   # placeholder -> original
    counts: dict[str, int] = field(default_factory=dict)

    def _placeholder(self, kind: str, original: str) -> str:
        for ph, orig in self.mapping.items():
            if orig == original and ph.startswith(f"[{kind}_"):
                return ph
        self.counts[kind] = self.counts.get(kind, 0) + 1
        ph = f"[{kind}_{self.counts[kind]}]"
        self.mapping[ph] = original
        return ph

    def redact(self, text: str, sensitive_terms: list[str] = ()) -> str:
        text = normalize_light(text)
        for term in sorted(sensitive_terms, key=len, reverse=True):
            if term:
                text = text.replace(term, self._placeholder("NAME", term))
        for kind, pattern in PATTERNS:
            text = pattern.sub(lambda m, k=kind: self._mask(k, m), text)
        return text

    def _mask(self, kind: str, m: re.Match) -> str:
        if m.re.groups:
            whole, value = m.group(0), m.group(1)
            return whole.replace(value, self._placeholder(kind, value), 1)
        return self._placeholder(kind, m.group(0))

    def restore(self, text: str) -> str:
        for ph in sorted(self.mapping, key=len, reverse=True):
            text = text.replace(ph, self.mapping[ph])
        return text
