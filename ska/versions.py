"""The latest-version rule, decided from metadata, not by the model.

A document is *superseded* when another document that is already in force names it in
`supersedes`. A document whose effective_date is in the future is *not yet effective*.
By default only *current* documents can be retrieved. When someone explicitly asks about
history ("what changed from the 2025 version?"), superseded documents are allowed too, and
every source is labelled with its version and date so the answer can compare them.
"""

import re
from datetime import date

from .arabic import normalize_search
from .documents import Document

CURRENT, SUPERSEDED, NOT_YET = "current", "superseded", "not_yet_effective"

HISTORY_PATTERNS = re.compile(
    r"النسخه السابقه|النسخه القديمه|الاصدار السابق|الاصدار القديم|السياسه القديمه|السياسه السابقه|"
    r"قبل التحديث|ما الذي تغير|ماذا تغير|وش تغير|ايش تغير|كيف تغير|الفرق بين|كم كانت|كم كان|"
    r"في 2025|عام 2025|سنه 2025|العام الماضي|السنه الماضيه|"
    r"previous version|old version|older version|what changed|history of|used to be|last year|in 2025"
)


def statuses(docs: list[Document], today: date | None = None) -> dict[str, str]:
    today = today or date.today()
    in_force = {d.doc_id for d in docs if d.effective_date <= today}
    replaced = {d.supersedes for d in docs if d.supersedes and d.doc_id in in_force}
    out = {}
    for d in docs:
        if d.doc_id not in in_force:
            out[d.doc_id] = NOT_YET
        elif d.doc_id in replaced:
            out[d.doc_id] = SUPERSEDED
        else:
            out[d.doc_id] = CURRENT
    return out


def asks_for_history(question: str) -> bool:
    return bool(HISTORY_PATTERNS.search(normalize_search(question)))

