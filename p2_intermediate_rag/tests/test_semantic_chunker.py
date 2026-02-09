"""Tests for SemanticChunker edge cases."""

import pytest
import numpy as np

from src.loaders import Document
from src.chunking import SemanticChunker, Chunk


class TestSemanticChunker:
    """Tests for SemanticChunker."""

    def test_empty_document(self, monkeypatch):
        """Test chunking empty document returns empty list."""
        doc = Document(content="", metadata={})
        chunker = SemanticChunker(min_chunk_size=10)
        chunks = chunker.chunk(doc)
        assert chunks == []

    def test_single_sentence(self, monkeypatch):
        """Test chunking single sentence returns one chunk."""
        doc = Document(content="This is a single sentence.", metadata={"source": "test.txt"})
        chunker = SemanticChunker(min_chunk_size=10)

        class DummyEmbedder:
            def encode(self, sentences, normalize_embeddings=True, show_progress_bar=False):
                return np.ones((len(sentences), 4), dtype=float)

        monkeypatch.setattr(SemanticChunker, "embedder", property(lambda self: DummyEmbedder()))

        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].content == "This is a single sentence."

    def test_min_chunk_size_enforcement(self, monkeypatch):
        """Test that chunks meet minimum size when merging is needed."""
        doc_text = "A. B. C. D. E. F. G. H."
        doc = Document(content=doc_text, metadata={"source": "test.txt"})
        chunker = SemanticChunker(
            chunk_size=1000,
            chunk_overlap=0,
            breakpoint_threshold=0.0,
            min_chunk_size=10,
        )

        class DummyEmbedder:
            def encode(self, sentences, normalize_embeddings=True, show_progress_bar=False):
                return np.ones((len(sentences), 4), dtype=float)

        monkeypatch.setattr(SemanticChunker, "embedder", property(lambda self: DummyEmbedder()))
        monkeypatch.setattr(SemanticChunker, "_find_breakpoints", lambda self, emb: [1, 2, 3, 4, 5, 6, 7])

        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1
        for c in chunks:
            assert len(c.content) >= chunker.min_chunk_size or len(chunks) == 1

    def test_very_long_text_splits_into_multiple(self, monkeypatch):
        """Test that very long text gets split into multiple chunks."""
        doc_text = "First sentence. " * 200  # Many short sentences
        doc = Document(content=doc_text, metadata={"source": "test.txt"})
        chunker = SemanticChunker(
            chunk_size=500,
            chunk_overlap=50,
            min_chunk_size=50,
        )

        class DummyEmbedder:
            def encode(self, sentences, normalize_embeddings=True, show_progress_bar=False):
                # Create varying similarities to induce breakpoints
                n = len(sentences)
                emb = np.random.RandomState(42).rand(n, 8).astype(np.float32)
                return emb / np.linalg.norm(emb, axis=1, keepdims=True)

        monkeypatch.setattr(SemanticChunker, "embedder", property(lambda self: DummyEmbedder()))

        chunks = chunker.chunk(doc)
        assert len(chunks) >= 2
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_metadata_inherited(self, monkeypatch):
        """Test that chunks inherit document metadata."""
        doc = Document(content="Hello. World.", metadata={"source": "test.txt", "author": "Test"})
        chunker = SemanticChunker(min_chunk_size=5)

        class DummyEmbedder:
            def encode(self, sentences, normalize_embeddings=True, show_progress_bar=False):
                return np.ones((len(sentences), 4), dtype=float)

        monkeypatch.setattr(SemanticChunker, "embedder", property(lambda self: DummyEmbedder()))

        chunks = chunker.chunk(doc)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata.get("source") == "test.txt"
            assert chunk.metadata.get("author") == "Test"
