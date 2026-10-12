"""Audit log: one row per question, plus a table of conflicts flagged to document owners.

Every row carries what you need to trace a regression after a model or prompt update
(form question 55): the release id, the exact model, the prompt version and its hash, the
embedding model, which chunks were retrieved with their scores, what was blocked, the
decision, the post-check flags, tokens, cost, latency and stop reason. Filter by release and
by user role, and you can see whether errors went up for one group only.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    request_id TEXT NOT NULL,
    user_id TEXT, roles TEXT,
    question TEXT,
    release_id TEXT, model TEXT, prompt_version TEXT, prompt_hash TEXT, embed_model TEXT,
    history_mode INTEGER,
    candidates INTEGER, blocked_by_access INTEGER, blocked_by_version INTEGER,
    retrieved TEXT,             -- JSON: chunk_id, doc_id, version, status, dense, bm25, rrf
    restricted_matches TEXT,    -- JSON: forbidden doc_ids that would have matched (security)
    best_dense REAL,
    decision TEXT, reason TEXT,
    cited_docs TEXT, flagged_sentences INTEGER, flags TEXT, conflicts TEXT,
    redacted INTEGER,
    input_tokens INTEGER, output_tokens INTEGER, cache_read_tokens INTEGER,
    cache_creation_tokens INTEGER,
    cost_usd REAL, latency_ms INTEGER, stop_reason TEXT, error TEXT
);
CREATE TABLE IF NOT EXISTS flags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL, request_id TEXT, kind TEXT, owner TEXT,
    doc_ids TEXT, topic TEXT, detail TEXT, status TEXT DEFAULT 'open'
);
"""


class AuditLog:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def record(self, **row) -> None:
        row = {k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
               for k, v in row.items()}
        row.setdefault("ts", self._now())
        cols = ", ".join(row)
        with self._conn() as c:
            c.execute(f"INSERT INTO queries ({cols}) VALUES ({', '.join('?' * len(row))})",
                      list(row.values()))

    def flag(self, *, request_id: str, kind: str, owner: str, doc_ids: list[str], topic: str,
             detail: str) -> None:
        with self._conn() as c:
            c.execute("INSERT INTO flags (ts, request_id, kind, owner, doc_ids, topic, detail) "
                      "VALUES (?, ?, ?, ?, ?, ?, ?)",
                      (self._now(), request_id, kind, owner, json.dumps(doc_ids), topic, detail))

    def recent(self, n: int = 20) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM queries ORDER BY id DESC LIMIT ?", (n,))]

    def open_flags(self) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute(
                "SELECT * FROM flags WHERE status='open' ORDER BY id DESC")]

    def summary_by(self, field: str = "release_id") -> list[dict]:
        """Error and refusal rates per release (or per roles): the first thing to look at
        after an update."""
        if field not in ("release_id", "roles", "model", "prompt_hash"):
            raise ValueError(field)
        with self._conn() as c:
            return [dict(r) for r in c.execute(f"""
                SELECT {field} AS key, COUNT(*) AS n,
                  ROUND(AVG(decision='error'), 3) AS error_rate,
                  ROUND(AVG(decision='refused'), 3) AS refusal_rate,
                  ROUND(AVG(flagged_sentences > 0), 3) AS flagged_rate,
                  ROUND(AVG(latency_ms)) AS avg_latency_ms,
                  ROUND(SUM(cost_usd), 5) AS cost_usd
                FROM queries GROUP BY {field} ORDER BY key""")]
