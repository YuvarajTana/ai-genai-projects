"""Tests for FastAPI endpoints."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from src.api.main import app, create_app


@pytest.fixture
def client():
    """Create test client with mocked components."""
    # Create fresh app to avoid state leaks
    test_app = create_app()
    # Mock components to avoid loading real models
    mock_retriever = MagicMock()
    mock_retriever.count.return_value = 1
    mock_retriever.retrieve.return_value = [
        {
            "id": "test::chunk_0",
            "text": "Test context from document.",
            "metadata": {"source": "test.txt", "chunk_index": 0},
            "distance": 0.1,
        }
    ]
    mock_retriever.index = MagicMock()
    mock_retriever.clear = MagicMock()

    mock_llm = MagicMock()
    mock_llm.agenerate = AsyncMock(return_value="This is a test answer.")
    mock_llm.is_available.return_value = True

    mock_prompt_builder = MagicMock()
    mock_prompt_builder.build_prompt.return_value = "Context: ...\nQuestion: ..."

    test_app.state.embedder = MagicMock()
    test_app.state.retriever = mock_retriever
    test_app.state.llm = mock_llm
    test_app.state.prompt_builder = mock_prompt_builder

    return TestClient(test_app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self, client):
        """Test health check returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data


class TestStatsEndpoint:
    """Tests for /stats endpoint."""

    def test_stats_returns_200(self, client):
        """Test stats returns 200."""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "chunks_indexed" in data
        assert "supported_extensions" in data
        assert "llm_model" in data


class TestQueryEndpoint:
    """Tests for /query endpoint."""

    def test_query_returns_answer(self, client):
        """Test query returns answer when index has documents."""
        response = client.post(
            "/query",
            json={"query": "What is the test?", "k": 4, "use_reranker": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "This is a test answer." in data["answer"]

    def test_query_respects_use_reranker_param(self, client):
        """Test that use_reranker is passed to retriever."""
        client.post(
            "/query",
            json={"query": "test", "k": 4, "use_reranker": True},
        )
        client.app.state.retriever.retrieve.assert_called_once()
        call_kwargs = client.app.state.retriever.retrieve.call_args[1]
        assert call_kwargs["use_reranker"] is True

    def test_query_returns_400_when_no_index(self, client):
        """Test query returns 400 when no documents indexed."""
        client.app.state.retriever.count.return_value = 0
        response = client.post(
            "/query",
            json={"query": "test", "k": 4},
        )
        assert response.status_code == 400


class TestIndexEndpoint:
    """Tests for /index endpoint."""

    def test_index_succeeds_with_documents(self, client, tmp_path):
        """Test index returns 200 when directory has supported docs."""
        (tmp_path / "test.txt").write_text("This is test content for indexing.")
        response = client.post(
            "/index",
            json={"directory": str(tmp_path)},
        )
        # With mocked retriever, index should succeed (retriever.index is mocked)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["chunks_indexed"] > 0

    def test_index_returns_400_when_no_documents(self, client, tmp_path):
        """Test index returns 400 when directory has no supported docs."""
        # Empty directory
        response = client.post(
            "/index",
            json={"directory": str(tmp_path)},
        )
        assert response.status_code == 400


class TestClearIndexEndpoint:
    """Tests for DELETE /index endpoint."""

    def test_clear_index_returns_200(self, client):
        """Test clear index returns success."""
        response = client.delete("/index")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
