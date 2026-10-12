"""Structure-aware chunking.

Policies are written as numbered sections ("## 3. المياومة") with numbered clauses ("3.1").
We cut on section headings first, so a chunk never mixes two topics, and a citation can say
"section 3". A section that is too long is cut again between clauses, never inside one.
Each chunk carries its document title and section heading, so the embedding knows that a
clause saying "350 ريال" is about per diem even if the clause itself doesn't say so.
"""

import re
from dataclasses import dataclass

from .documents import Document

MAX_CHARS = 1200
MD_HEADING = re.compile(r"^##\s+(?:(\d+)[.)]\s*)?(.+?)\s*$")
PLAIN_HEADING = re.compile(r"^(\d+)\.\s+(\S.{0,80})$")      # for PDF text: "3. Audio"
CLAUSE = re.compile(r"^\s*\d+\.\d+")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    section: str          # "3"
    section_title: str    # "المياومة"
    text: str             # the clause text as written

    def embed_text(self, doc_title: str) -> str:
        return f"{doc_title} — {self.section_title}\n{self.text}"


def _sections(body: str) -> list[tuple[str, str, list[str]]]:
    lines = body.splitlines()
    use_md = any(MD_HEADING.match(line) for line in lines)
    sections: list[tuple[str, str, list[str]]] = []
    current = ("0", "", [])
    for line in lines:
        m = MD_HEADING.match(line) if use_md else PLAIN_HEADING.match(line.strip())
        if m:
            if any(x.strip() for x in current[2]):
                sections.append(current)
            number = m.group(1) or str(len(sections) + 1)
            current = (number, m.group(2).strip(), [])
            continue
        if line.startswith("# ") or line.startswith(">"):
            continue                       # document title and the synthetic-data notice
        current[2].append(line)
    if any(x.strip() for x in current[2]):
        sections.append(current)
    return sections


def _split_long(lines: list[str]) -> list[str]:
    """Pack clauses into pieces of at most MAX_CHARS, cutting only before a clause."""
    pieces, buf = [], []
    for line in lines:
        starts_clause = bool(CLAUSE.match(line))
        if starts_clause and buf and len("\n".join(buf)) + len(line) > MAX_CHARS:
            pieces.append("\n".join(buf).strip())
            buf = []
        buf.append(line)
    if buf and "\n".join(buf).strip():
        pieces.append("\n".join(buf).strip())
    return pieces


def chunk_document(doc: Document) -> list[Chunk]:
    chunks = []
    for number, title, lines in _sections(doc.body):
        text = "\n".join(lines).strip()
        if number == "0" and text.replace("\n", " ").strip() == doc.title:
            continue                       # a PDF's title line on its own is not content
        pieces = [text] if len(text) <= MAX_CHARS else _split_long(lines)
        for i, piece in enumerate(pieces):
            suffix = f"-{i + 1}" if len(pieces) > 1 else ""
            chunks.append(Chunk(chunk_id=f"{doc.doc_id}#s{number}{suffix}", doc_id=doc.doc_id,
                                section=number, section_title=title or doc.title, text=piece))
    return chunks
