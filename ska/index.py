"""The index: one SQLite file per embedding model, plus numpy in memory.

Why not a vector database? About 30 documents become a few hundred chunks. Comparing a
question with every chunk (brute force) takes well under a millisecond and is exact, and the
access/version filter is a plain boolean mask, so we can prove what was searched. The file can
be opened with any SQLite tool to see every chunk and its metadata. At tens of thousands of
documents I would move the same design to pgvector or Qdrant with a payload filter on
access and status, applied inside the search, not after it.
"""

import json
import sqlite3
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np

from .arabic import tokenize
from .chunking import Chunk, chunk_document
from .documents import Document, load_corpus
from .embeddings import Embedder

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS docs (doc_id TEXT PRIMARY KEY, meta_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL, section TEXT, section_title TEXT,
    text TEXT NOT NULL, embedding BLOB NOT NULL
);
"""


def index_path(index_dir: Path, embed_model: str) -> Path:
    return Path(index_dir) / (embed_model.replace("/", "__") + ".sqlite")


def _doc_to_json(doc: Document) -> str:
    d = asdict(doc)
    d["effective_date"] = doc.effective_date.isoformat()
    return json.dumps(d, ensure_ascii=False)


def _doc_from_json(s: str) -> Document:
    d = json.loads(s)
    d["effective_date"] = date.fromisoformat(d["effective_date"])
    return Document(**d)


def _load_and_embed(corpus_dir: Path, embedder: Embedder):
    """Read the documents, cut them into chunks, embed each chunk with its title."""
    docs = load_corpus(corpus_dir)
    titles = {d.doc_id: d.title for d in docs}
    chunks = [c for d in docs for c in chunk_document(d)]
    vectors = embedder.encode_docs([c.embed_text(titles[c.doc_id]) for c in chunks])
    return docs, chunks, vectors


def build_index(corpus_dir: Path, embedder: Embedder, out_path: Path) -> dict:
    docs, chunks, vectors = _load_and_embed(corpus_dir, embedder)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    conn = sqlite3.connect(out_path)
    conn.executescript(SCHEMA)
    conn.execute("INSERT INTO meta VALUES ('embed_model', ?)", (embedder.name,))
    conn.execute("INSERT INTO meta VALUES ('dim', ?)", (str(embedder.dim),))
    conn.executemany("INSERT INTO docs VALUES (?, ?)", [(d.doc_id, _doc_to_json(d)) for d in docs])
    conn.executemany(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
        [(c.chunk_id, c.doc_id, c.section, c.section_title, c.text,
          vectors[i].astype(np.float32).tobytes()) for i, c in enumerate(chunks)])
    conn.commit()
    conn.close()
    return {"docs": len(docs), "chunks": len(chunks), "path": str(out_path)}


class Index:
    def __init__(self, docs: dict[str, Document], chunks: list[Chunk], matrix: np.ndarray,
                 embed_model: str):
        self.docs = docs
        self.chunks = chunks
        self.matrix = matrix
        self.embed_model = embed_model
        self.chunk_doc = [docs[c.doc_id] for c in chunks]
        self.tokens = [tokenize(c.embed_text(docs[c.doc_id].title)) for c in chunks]

    @classmethod
    def load(cls, path: Path) -> "Index":
        if not Path(path).exists():
            raise FileNotFoundError(f"No index at {path}. Run: python -m ska ingest")
        conn = sqlite3.connect(path)
        embed_model = conn.execute("SELECT value FROM meta WHERE key='embed_model'").fetchone()[0]
        docs = {r[0]: _doc_from_json(r[1]) for r in conn.execute("SELECT * FROM docs")}
        rows = conn.execute("SELECT chunk_id, doc_id, section, section_title, text, embedding "
                            "FROM chunks ORDER BY rowid").fetchall()
        conn.close()
        chunks = [Chunk(r[0], r[1], r[2], r[3], r[4]) for r in rows]
        matrix = np.stack([np.frombuffer(r[5], dtype=np.float32) for r in rows])
        return cls(docs, chunks, matrix, embed_model)

    @classmethod
    def from_corpus(cls, corpus_dir: Path, embedder: Embedder) -> "Index":
        """Build in memory, without a file. Used by tests."""
        docs, chunks, vectors = _load_and_embed(corpus_dir, embedder)
        return cls({d.doc_id: d for d in docs}, chunks, vectors, embedder.name)
