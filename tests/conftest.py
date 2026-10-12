"""Test helpers. Unit tests never call the network or load a real embedding model:
a hashing embedder stands in for bge-m3 and a fake client stands in for Claude."""

import json
import re
from types import SimpleNamespace

import pytest

from ska import config
from ska.access import load_users
from ska.assistant import Assistant
from ska.audit import AuditLog
from ska.embeddings import HashEmbedder
from ska.index import Index

SOURCE = re.compile(r'<source id="(S\d+)" title="([^"]*)"[^>]*>\n(.*?)\n</source>', re.DOTALL)


def sources_in(kwargs) -> list[tuple[str, str, str]]:
    """(id, title, text) of every source block in the prompt the fake client received."""
    return SOURCE.findall(kwargs["messages"][0]["content"])


def first_clause(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return text


def answer_from_first_source(kwargs) -> dict:
    sid, _, text = sources_in(kwargs)[0]
    return {"status": "answered", "sentences": [{"text": first_clause(text), "sources": [sid]}],
            "conflicts": [], "missing": ""}


class FakeAnthropic:
    """Records every request. `responder(kwargs) -> dict` builds the JSON answer."""

    def __init__(self, responder=answer_from_first_source, stop_reason="end_turn",
                 cache_read=0):
        self.responder = responder
        self.stop_reason = stop_reason
        self.cache_read = cache_read
        self.sent = []
        self.messages = self

    def create(self, **kwargs):
        self.sent.append(kwargs)
        payload = self.responder(kwargs)
        return SimpleNamespace(
            model=kwargs["model"], stop_reason=self.stop_reason,
            usage=SimpleNamespace(input_tokens=1500, output_tokens=200,
                                  cache_creation_input_tokens=0,
                                  cache_read_input_tokens=self.cache_read),
            content=[SimpleNamespace(type="thinking", thinking=""),
                     SimpleNamespace(type="text", text=json.dumps(payload, ensure_ascii=False))])


TEST_RELEASE = config.Release(id="test-r2", llm_model="claude-haiku-5-5", prompt="answer_v2",
                              effort="low", embed_model=HashEmbedder.name, top_k=6,
                              min_dense=0.05)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("COST_DB_PATH", str(tmp_path / "costs.sqlite"))
    monkeypatch.setenv("COST_BUDGET_USD", "15")
    monkeypatch.delenv("SKA_RELEASE", raising=False)


@pytest.fixture(scope="session")
def embedder():
    return HashEmbedder()


@pytest.fixture(scope="session")
def index(embedder):
    return Index.from_corpus(config.CORPUS_DIR, embedder)


@pytest.fixture(scope="session")
def users():
    return load_users(config.USERS_FILE)


@pytest.fixture
def audit(tmp_path):
    return AuditLog(tmp_path / "audit.sqlite")


@pytest.fixture
def make_assistant(index, embedder, audit):
    def make(client=None, release=TEST_RELEASE, **kw):
        return Assistant(index, embedder, release, audit, client=client or FakeAnthropic(), **kw)
    return make
