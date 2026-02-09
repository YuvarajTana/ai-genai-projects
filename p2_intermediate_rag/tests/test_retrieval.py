"""Tests for retrieval components."""

import pytest

from src.chunking.base import Chunk
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_search import BM25Search
from src.retrieval.hybrid_retriever import HybridRetriever
from tests.conftest import mock_embedder, sample_chunks


class TestVectorStore:
    """Tests for VectorStore."""

    def test_add_and_search(self, mock_embedder, sample_chunks, tmp_path):
        """Test adding chunks and searching."""
        store = VectorStore(
            collection_name="test_collection",
            persist_directory=str(tmp_path),
            embedder=mock_embedder,
        )
        store.add_chunks(sample_chunks)
        assert store.count() == 3

        results = store.search("retrieval generation", k=2)
        assert len(results) == 2
        assert "text" in results[0]
        assert "metadata" in results[0]
        assert "distance" in results[0]

    def test_clear(self, mock_embedder, sample_chunks, tmp_path):
        """Test clearing the store."""
        store = VectorStore(
            collection_name="test_collection",
            persist_directory=str(tmp_path),
            embedder=mock_embedder,
        )
        store.add_chunks(sample_chunks)
        assert store.count() == 3
        store.clear()
        assert store.count() == 0


class TestBM25Search:
    """Tests for BM25Search."""

    def test_index_and_search(self, sample_chunks):
        """Test indexing and searching."""
        bm25 = BM25Search()
        bm25.index(sample_chunks)
        results = bm25.search("retrieval vector", k=2)
        assert len(results) <= 2
        for r in results:
            assert "id" in r
            assert "text" in r
            assert "score" in r

    def test_search_finds_keyword_matches(self, sample_chunks):
        """Test that BM25 finds keyword matches."""
        bm25 = BM25Search()
        bm25.index(sample_chunks)
        results = bm25.search("BM25", k=3)
        assert len(results) >= 1
        assert any("BM25" in r["text"] for r in results)

    def test_clear(self, sample_chunks):
        """Test clearing BM25 index."""
        bm25 = BM25Search()
        bm25.index(sample_chunks)
        bm25.clear()
        with pytest.raises(ValueError, match="Index not built"):
            bm25.search("test", k=1)


class TestHybridRetriever:
    """Tests for HybridRetriever."""

    def test_retrieve_without_reranker(self, mock_embedder, sample_chunks, tmp_path):
        """Test hybrid retrieval without reranker."""
        retriever = HybridRetriever(
            embedder=mock_embedder,
            collection_name="test_hybrid",
            persist_directory=str(tmp_path),
            use_bm25=True,
            use_reranker=False,
        )
        retriever.index(sample_chunks)
        results = retriever.retrieve("retrieval generation", k=2, use_reranker=False)
        assert len(results) == 2
        assert "combined_score" in results[0] or "vector_score" in results[0]

    def test_retrieve_respects_use_reranker_param(self, mock_embedder, sample_chunks, tmp_path):
        """Test that use_reranker parameter overrides instance setting."""
        retriever = HybridRetriever(
            embedder=mock_embedder,
            collection_name="test_hybrid2",
            persist_directory=str(tmp_path),
            use_bm25=True,
            use_reranker=True,  # Instance has reranker enabled
        )
        retriever.index(sample_chunks)
        # Disable reranker per-request (reranker would need model loaded)
        results = retriever.retrieve("retrieval", k=2, use_reranker=False)
        assert len(results) == 2

    def test_deduplication(self, mock_embedder, sample_chunks, tmp_path):
        """Test that hybrid combines and deduplicates vector + BM25 results."""
        retriever = HybridRetriever(
            embedder=mock_embedder,
            collection_name="test_hybrid3",
            persist_directory=str(tmp_path),
            use_bm25=True,
            use_reranker=False,
        )
        retriever.index(sample_chunks)
        results = retriever.retrieve("vector database", k=5)
        # Should have unique IDs (no duplicates)
        ids = [r["id"] for r in results]
        assert len(ids) == len(set(ids))

    def test_count_and_clear(self, mock_embedder, sample_chunks, tmp_path):
        """Test count and clear."""
        retriever = HybridRetriever(
            embedder=mock_embedder,
            collection_name="test_hybrid4",
            persist_directory=str(tmp_path),
            use_bm25=True,
            use_reranker=False,
        )
        retriever.index(sample_chunks)
        assert retriever.count() == 3
        retriever.clear()
        assert retriever.count() == 0
