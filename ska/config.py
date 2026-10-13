"""Settings and releases.

A *release* pins the settings that change answers without a code change: the Claude model,
the prompt file, the effort, the embedding model and the retrieval thresholds. Code changes
are tracked separately: every eval run writes the git commit into eval/history.csv. The
active release is set in config/releases.yaml, and SKA_RELEASE in the environment overrides
it. Rolling back a bad model or prompt update means pointing `active` (or SKA_RELEASE) at an
earlier release; rolling back code means checking out the commit from history.csv.
"""

import csv
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
RELEASES_FILE = ROOT / "config" / "releases.yaml"
CORPUS_DIR = ROOT / "corpus" / "docs"
INDEX_DIR = Path(os.environ.get("SKA_INDEX_DIR", ROOT / "index"))
AUDIT_DB = Path(os.environ.get("SKA_AUDIT_DB", ROOT / "logs" / "audit.sqlite"))
USERS_FILE = ROOT / "config" / "users.yaml"
HISTORY_FILE = ROOT / "eval" / "history.csv"

# The cost ledger (agent_cost_tracker) holds the running total that the budget is checked
# against. A relative path is resolved against the repo, so running from another folder
# can't silently start a new ledger with a fresh budget.
_cost_db = Path(os.environ.get("COST_DB_PATH", "costs.sqlite"))
os.environ["COST_DB_PATH"] = str(_cost_db if _cost_db.is_absolute() else ROOT / _cost_db)

# What to do with highly confidential chunks before an external model call (ska/redaction.py).
EXTERNAL_POLICIES = ("redact", "local_only")


@dataclass(frozen=True)
class Release:
    id: str
    llm_model: str
    prompt: str                 # file name in ska/prompts without .md
    effort: str
    embed_model: str
    top_k: int
    min_dense: float            # relevance gate: best cosine score must reach this
    external_policy: str = "redact"   # for highly confidential docs: redact | local_only
    notes: str = ""

    def __post_init__(self):
        # A typo here ("local-only") must stop the app, not quietly send unmasked text.
        if self.external_policy not in EXTERNAL_POLICIES:
            raise ValueError(f"Release {self.id}: external_policy must be one of "
                             f"{EXTERNAL_POLICIES}, got {self.external_policy!r}")

    @property
    def prompt_text(self) -> str:
        return (PROMPTS_DIR / f"{self.prompt}.md").read_text(encoding="utf-8")

    @property
    def prompt_hash(self) -> str:
        """Short hash of the exact prompt text, logged with every answer."""
        return hashlib.sha256(self.prompt_text.encode("utf-8")).hexdigest()[:12]


def load_releases(path: Path | None = None) -> tuple[str, dict[str, Release]]:
    data = yaml.safe_load(Path(path or RELEASES_FILE).read_text(encoding="utf-8"))
    releases = {}
    for rid, values in data["releases"].items():
        releases[rid] = Release(id=rid, **values)     # an unknown or misspelled key is an error
    return data["active"], releases


def active_release(path: Path | None = None) -> Release:
    active, releases = load_releases(path)
    rid = os.environ.get("SKA_RELEASE") or active
    if rid not in releases:
        raise KeyError(f"Release {rid!r} is not in {path or RELEASES_FILE}")
    return releases[rid]


def set_active_release(rid: str, path: Path | None = None) -> None:
    """The rollback switch: rewrite only the `active:` line, keep the rest of the file."""
    path = Path(path or RELEASES_FILE)
    _, releases = load_releases(path)
    if rid not in releases:
        raise KeyError(f"Unknown release {rid!r}. Known: {', '.join(releases)}")
    lines = path.read_text(encoding="utf-8").splitlines()
    lines = [f'active: "{rid}"' if line.startswith("active:") else line for line in lines]
    with open(path, "w", encoding="utf-8", newline="\n") as f:    # keep LF endings on Windows
        f.write("\n".join(lines) + "\n")


def last_eval_gate(rid: str, history: Path | None = None) -> str | None:
    """The gate result ("pass" / "FAIL") of the newest eval run of this release, or None."""
    history = Path(history or HISTORY_FILE)
    if not history.exists():
        return None
    with open(history, newline="", encoding="utf-8") as f:
        runs = [r for r in csv.DictReader(f) if r["release_id"] == rid]
    return runs[-1]["gate"] if runs else None
