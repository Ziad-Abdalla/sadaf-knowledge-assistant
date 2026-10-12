from datetime import date
from pathlib import Path

import pytest

from ska import config
from ska.chunking import MAX_CHARS, chunk_document
from ska.documents import CorpusError, Document, load_corpus, load_markdown


def test_corpus_loads_and_every_doc_is_marked_synthetic():
    docs = load_corpus(config.CORPUS_DIR)
    assert len(docs) >= 25
    for d in docs:
        text = Path(d.source_path).read_text(encoding="utf-8") if d.source_path.endswith(".md") else ""
        if text:
            assert "synthetic: true" in text, d.doc_id


def test_bad_role_is_rejected(tmp_path):
    p = tmp_path / "X.md"
    p.write_text("---\ndoc_id: X\ntitle: t\ndepartment: hr\ndoc_type: policy\nversion: '1'\n"
                 "effective_date: 2026-01-01\naccess: [interns]\nconfidentiality: internal\n---\n# t\n",
                 encoding="utf-8")
    with pytest.raises(CorpusError, match="unknown roles"):
        load_markdown(p)


def test_missing_metadata_is_rejected(tmp_path):
    p = tmp_path / "Y.md"
    p.write_text("---\ndoc_id: Y\ntitle: t\n---\n# t\n", encoding="utf-8")
    with pytest.raises(CorpusError, match="missing metadata"):
        load_markdown(p)


def _doc(body: str) -> Document:
    return Document(doc_id="D", title="عنوان", department="hr", doc_type="policy", version="1",
                    effective_date=date(2026, 1, 1), access=["all"], confidentiality="internal",
                    body=body)


def test_chunks_follow_numbered_sections_and_skip_title_and_notice():
    body = "# عنوان\n> وثيقة تجريبية\n\n## 1. النطاق\n1.1 نص أول\n## 2. المياومة\n2.1 350 ريال\n"
    chunks = chunk_document(_doc(body))
    assert [(c.chunk_id, c.section, c.section_title) for c in chunks] == [
        ("D#s1", "1", "النطاق"), ("D#s2", "2", "المياومة")]
    assert "وثيقة تجريبية" not in chunks[0].text
    assert chunks[1].embed_text("عنوان").startswith("عنوان — المياومة")


def test_long_section_is_split_between_clauses_never_inside():
    clauses = "\n".join(f"3.{i} " + "كلمة " * 60 for i in range(1, 8))
    chunks = chunk_document(_doc(f"## 3. طويل\n{clauses}\n"))
    assert len(chunks) > 1
    assert all(len(c.text) <= MAX_CHARS + 400 for c in chunks)
    assert all(c.text.startswith("3.") for c in chunks)
    assert chunks[0].chunk_id == "D#s3-1"


def test_pdf_style_plain_headings_are_detected():
    body = "Broadcast spec\n1. Scope\n1.1 text\n2. Audio\n2.1 -23 LUFS\n"
    chunks = chunk_document(_doc(body))
    assert [c.section_title for c in chunks][-2:] == ["Scope", "Audio"]


def test_every_real_chunk_has_a_section():
    for d in load_corpus(config.CORPUS_DIR):
        for c in chunk_document(d):
            assert c.section and c.text.strip(), c.chunk_id
