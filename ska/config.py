"""Settings and releases.

A *release* pins everything that can change the answers: the Claude model, the prompt
version, the embedding model and the retrieval thresholds. The active release is set in
config/releases.yaml, and SKA_RELEASE in the environment overrides it. Rolling back after a
bad model or prompt update means pointing `active` (or SKA_RELEASE) at the previous release.
"""

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
RELEASES_FILE = Path(os.environ.get("SKA_RELEASES_FILE", ROOT / "config" / "releases.yaml"))
CORPUS_DIR = Path(os.environ.get("SKA_CORPUS_DIR", ROOT / "corpus" / "docs"))
INDEX_DIR = Path(os.environ.get("SKA_INDEX_DIR", ROOT / "index"))
AUDIT_DB = Path(os.environ.get("SKA_AUDIT_DB", ROOT / "logs" / "audit.sqlite"))
USERS_FILE = ROOT / "config" / "users.yaml"


@dataclass(frozen=True)
class Release:
    id: str
    llm_model: str
    prompt: str                 # file name in ska/prompts without .md
    effort: str
    embed_model: str
    top_k: int
    min_dense: float            # relevance gate: best cosine score must reach this
    rrf_k: int = 60
    external_policy: str = "redact"   # for highly confidential docs: redact | local_only | allow
    notes: str = ""
    extra: dict = field(default_factory=dict)

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
        known = {k: v for k, v in values.items() if k in Release.__dataclass_fields__}
        releases[rid] = Release(id=rid, **known)
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
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
