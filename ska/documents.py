"""Load the corpus: Markdown files with YAML front-matter, and PDFs with a sidecar
<name>.meta.yaml file holding the same metadata fields."""

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

from .arabic import normalize_light

REQUIRED = ("doc_id", "title", "department", "doc_type", "version", "effective_date", "access",
            "confidentiality")
ROLES = ("crew", "producer", "finance", "executive")
CONFIDENTIALITY = ("public", "internal", "confidential", "highly_confidential")

# Which kind of document wins when two sources disagree. Higher wins; ties go to the newer one.
AUTHORITY = {"policy": 3, "spec": 3, "template": 2, "guide": 2, "memo": 2, "minutes": 1}


@dataclass
class Document:
    doc_id: str
    title: str
    department: str
    doc_type: str
    version: str
    effective_date: date
    access: list[str]
    confidentiality: str
    body: str
    owner: str = ""
    supersedes: str | None = None
    language: str = "ar"
    odoo_link: str | None = None
    sensitive_terms: list[str] = field(default_factory=list)
    source_path: str = ""

    @property
    def authority(self) -> int:
        return AUTHORITY.get(self.doc_type, 1)


class CorpusError(ValueError):
    pass


FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)


def _from_meta(meta: dict, body: str, path: Path) -> Document:
    missing = [k for k in REQUIRED if meta.get(k) in (None, "")]
    if missing:
        raise CorpusError(f"{path.name}: missing metadata {missing}")
    access = meta["access"]
    if isinstance(access, str):
        access = [access]
    access = [a.strip().lower() for a in access]
    unknown = [a for a in access if a != "all" and a not in ROLES]
    if unknown:
        raise CorpusError(f"{path.name}: unknown roles in access: {unknown}")
    if meta["confidentiality"] not in CONFIDENTIALITY:
        raise CorpusError(f"{path.name}: bad confidentiality {meta['confidentiality']!r}")
    eff = meta["effective_date"]
    if isinstance(eff, str):
        eff = date.fromisoformat(eff)
    return Document(
        doc_id=str(meta["doc_id"]), title=str(meta["title"]), department=meta["department"],
        doc_type=meta["doc_type"], version=str(meta["version"]), effective_date=eff,
        access=access, confidentiality=meta["confidentiality"], body=normalize_light(body),
        owner=meta.get("owner") or "", supersedes=meta.get("supersedes") or None,
        language=meta.get("language", "ar"), odoo_link=meta.get("odoo_link") or None,
        sensitive_terms=list(meta.get("sensitive_terms") or []), source_path=str(path),
    )


def load_markdown(path: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    m = FRONT_MATTER.match(text)
    if not m:
        raise CorpusError(f"{path.name}: no YAML front-matter")
    return _from_meta(yaml.safe_load(m.group(1)), m.group(2), path)


def load_pdf(path: Path) -> Document:
    """PDF text via pypdf. The metadata lives next to it in <name>.meta.yaml, because a PDF
    exported from Word or a scanner has no reliable place for our fields."""
    from pypdf import PdfReader

    meta_path = path.with_suffix(".meta.yaml")
    if not meta_path.exists():
        raise CorpusError(f"{path.name}: needs {meta_path.name} with the document metadata")
    pages = [page.extract_text() or "" for page in PdfReader(str(path)).pages]
    return _from_meta(yaml.safe_load(meta_path.read_text(encoding="utf-8")),
                      "\n".join(pages), path)


def load_corpus(folder: Path) -> list[Document]:
    docs = []
    for path in sorted(Path(folder).iterdir()):
        if path.suffix == ".md":
            docs.append(load_markdown(path))
        elif path.suffix == ".pdf":
            docs.append(load_pdf(path))
    ids = [d.doc_id for d in docs]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise CorpusError(f"duplicate doc_id: {sorted(dupes)}")
    for d in docs:
        if d.supersedes and d.supersedes not in ids:
            raise CorpusError(f"{d.doc_id} supersedes unknown {d.supersedes}")
    return docs
