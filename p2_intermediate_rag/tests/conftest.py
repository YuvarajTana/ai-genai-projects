"""Pytest fixtures for P2 RAG tests."""

import numpy as np
import pytest

from src.chunking.base import Chunk
from src.loaders.base import Document


class MockEmbedder:
    """Mock embedder that returns deterministic embeddings."""

    def __init__(self, dim: int = 4):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        n = len(texts)
        # Deterministic embeddings based on text content
        rng = np.random.RandomState(hash(str(texts)) % 2**32)
        emb = rng.randn(n, self._dim).astype(np.float32)
        return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8)

    def embed_query(self, query: str):
        return self.embed([query])[0]

    def embed_documents(self, documents):
        return self.embed(documents)


@pytest.fixture
def mock_embedder():
    """Provide a mock embedder for tests."""
    return MockEmbedder(dim=4)


@pytest.fixture
def sample_chunks():
    """Provide sample chunks for retrieval tests."""
    return [
        Chunk(
            content="Retrieval augmented generation combines retrieval with generation.",
            metadata={"source": "doc1.txt", "chunk_index": 0},
            index=0,
        ),
        Chunk(
            content="Vector databases store embeddings for similarity search.",
            metadata={"source": "doc1.txt", "chunk_index": 1},
            index=1,
        ),
        Chunk(
            content="BM25 is a keyword-based ranking function.",
            metadata={"source": "doc2.txt", "chunk_index": 0},
            index=2,
        ),
    ]
