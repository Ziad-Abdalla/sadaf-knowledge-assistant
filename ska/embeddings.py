"""Embedding models. Local and free: nothing leaves the machine at this step.

An embedding turns a text into a list of numbers (a vector) so that texts with similar
meaning get vectors pointing in a similar direction. We compare directions with cosine
similarity; the vectors are normalized, so that is just a dot product.
"""

import hashlib
import re

import numpy as np

from .arabic import normalize_search, tokenize


class Embedder:
    name: str
    dim: int

    def encode_docs(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def encode_query(self, text: str) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    """bge-m3 or multilingual-e5 through sentence-transformers, on the GPU when there is one.
    e5 models were trained with "query: " / "passage: " prefixes and work worse without them;
    bge-m3 needs no prefix."""

    def __init__(self, name: str, device: str | None = None, batch_size: int = 16):
        import torch
        from sentence_transformers import SentenceTransformer

        self.name = name
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(name, device=device)
        if device == "cuda":
            self.model.half()
        self.dim = self.model.get_sentence_embedding_dimension()
        self.batch_size = batch_size
        self.e5 = "e5" in name.lower()

    def _encode(self, texts):
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    def encode_docs(self, texts):
        return self._encode([f"passage: {t}" if self.e5 else t for t in texts])

    def encode_query(self, text):
        return self._encode([f"query: {text}" if self.e5 else text])[0]


class HashEmbedder(Embedder):
    """A tiny stand-in for tests: hashed bag of stemmed words plus character trigrams.
    Deterministic, instant, no GPU. Not used in the real system."""

    name = "hash-test-embedder"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        feats = tokenize(text, query=True)
        flat = re.sub(r"\s+", " ", normalize_search(text))
        feats += [flat[i:i + 3] for i in range(max(0, len(flat) - 2))]
        for f in feats:
            h = int.from_bytes(hashlib.md5(f.encode("utf-8")).digest()[:4], "little")
            v[h % self.dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_docs(self, texts):
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim))

    def encode_query(self, text):
        return self._vec(text)


def load_embedder(name: str) -> Embedder:
    if name == HashEmbedder.name:
        return HashEmbedder()
    return SentenceTransformerEmbedder(name)
