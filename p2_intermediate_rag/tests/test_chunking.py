"""Tests for text chunking."""

import pytest

from src.loaders import Document
from src.chunking import (
    CharacterChunker,
    RecursiveChunker,
    SemanticChunker,
    ChunkerFactory,
    chunk_document,
    Chunk,
)


class TestCharacterChunker:
    """Tests for CharacterChunker."""
    
    def test_basic_chunking(self):
        """Test basic character chunking."""
        doc = Document(content="A" * 1000, metadata={"source": "test.txt"})
        chunker = CharacterChunker(chunk_size=300, chunk_overlap=50)
        
        chunks = chunker.chunk(doc)
        
        assert len(chunks) > 1
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(len(c) <= 300 for c in chunks)
    
    def test_empty_document(self):
        """Test chunking empty document."""
        doc = Document(content="", metadata={})
        chunker = CharacterChunker()
        
        chunks = chunker.chunk(doc)
        
        assert chunks == []
    
    def test_small_document(self):
        """Test chunking document smaller than chunk_size."""
        doc = Document(content="Small text", metadata={})
        chunker = CharacterChunker(chunk_size=1000)
        
        chunks = chunker.chunk(doc)
        
        assert len(chunks) == 1
        assert chunks[0].content == "Small text"
    
    def test_chunk_metadata(self):
        """Test that chunks inherit document metadata."""
        doc = Document(
            content="A" * 500,
            metadata={"source": "test.txt", "author": "Test"}
        )
        chunker = CharacterChunker(chunk_size=200, chunk_overlap=20)
        
        chunks = chunker.chunk(doc)
        
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert chunk.metadata["source"] == "test.txt"
            assert "chunk_index" in chunk.metadata


class TestRecursiveChunker:
    """Tests for RecursiveChunker."""
    
    def test_splits_on_paragraphs(self):
        """Test that recursive chunker splits on paragraphs."""
        content = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        doc = Document(content=content, metadata={})
        
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=0)
        chunks = chunker.chunk(doc)
        
        assert len(chunks) >= 2
    
    def test_respects_chunk_size(self):
        """Test that chunks respect maximum size."""
        content = "A" * 2000
        doc = Document(content=content, metadata={})
        
        chunker = RecursiveChunker(chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk(doc)
        
        # Most chunks should be near chunk_size (allowing for overlap merging)
        assert all(len(c) <= 600 for c in chunks)  # Some tolerance


class TestSemanticChunker:
    """Tests for SemanticChunker."""

    def test_enforces_min_chunk_size_by_merging(self, monkeypatch):
        """
        The semantic chunker should merge consecutive breakpoint chunks until
        the minimum chunk size is met.
        """
        doc_text = "A. B. C. D. E. F. G. H."
        doc = Document(content=doc_text, metadata={"source": "test.txt"})

        chunker = SemanticChunker(
            chunk_size=1000,
            chunk_overlap=0,
            breakpoint_threshold=0.0,  # doesn't matter; we'll patch breakpoints
            min_chunk_size=10,         # forces merges (each sentence is tiny)
        )

        # Avoid downloading/loading real models in tests
        class DummyEmbedder:
            def encode(self, sentences, normalize_embeddings=True, show_progress_bar=False):
                # identical embeddings => no semantic breaks unless patched
                import numpy as np
                return np.ones((len(sentences), 4), dtype=float)

        monkeypatch.setattr(SemanticChunker, "embedder", property(lambda self: DummyEmbedder()))
        monkeypatch.setattr(SemanticChunker, "_find_breakpoints", lambda self, emb: [1, 2, 3, 4, 5, 6, 7])

        chunks = chunker.chunk(doc)

        assert len(chunks) >= 1
        assert all(len(c.content) >= chunker.min_chunk_size or len(chunks) == 1 for c in chunks)


class TestChunkerFactory:
    """Tests for ChunkerFactory."""
    
    def test_get_available_strategies(self):
        """Test getting available chunking strategies."""
        strategies = ChunkerFactory.get_available_strategies()
        
        assert "character" in strategies
        assert "recursive" in strategies
        assert "semantic" in strategies
    
    def test_get_character_chunker(self):
        """Test getting character chunker."""
        chunker = ChunkerFactory.get_chunker("character", chunk_size=500)
        
        assert isinstance(chunker, CharacterChunker)
        assert chunker.chunk_size == 500
    
    def test_get_recursive_chunker(self):
        """Test getting recursive chunker."""
        chunker = ChunkerFactory.get_chunker("recursive")
        
        assert isinstance(chunker, RecursiveChunker)
    
    def test_invalid_strategy(self):
        """Test error for invalid strategy."""
        with pytest.raises(ValueError) as exc_info:
            ChunkerFactory.get_chunker("invalid")
        
        assert "Unknown chunking strategy" in str(exc_info.value)


class TestChunkDocument:
    """Tests for chunk_document convenience function."""
    
    def test_chunk_document(self):
        """Test chunking a document via convenience function."""
        doc = Document(content="A" * 500, metadata={"source": "test.txt"})
        
        chunks = chunk_document(
            doc,
            strategy="character",
            chunk_size=200,
            chunk_overlap=20,
        )
        
        assert len(chunks) > 1
        assert all(isinstance(c, Chunk) for c in chunks)
