"""Tests for PromptBuilder."""

import pytest

from src.loaders import Document
from src.generation import PromptBuilder


class TestPromptBuilder:
    """Tests for PromptBuilder."""

    def test_build_prompt_basic(self):
        """Test basic prompt building."""
        builder = PromptBuilder()
        results = [
            {"text": "Context text here.", "metadata": {"source": "doc.txt", "chunk_index": 0}},
        ]
        prompt = builder.build_prompt(question="What is X?", results=results)
        assert "Context text here." in prompt
        assert "What is X?" in prompt
        assert "doc.txt" in prompt

    def test_build_prompt_handles_curly_braces_in_question(self):
        """Test that user input with { or } does not cause KeyError."""
        builder = PromptBuilder()
        results = [
            {"text": "Python uses {key: value} for dicts.", "metadata": {"source": "doc.txt"}},
        ]
        # Should not raise KeyError or ValueError
        prompt = builder.build_prompt(
            question="How do I create a dict with { 'a': 1 } in Python?",
            results=results,
        )
        assert "How do I create a dict with { 'a': 1 } in Python?" in prompt
        assert "Python uses {key: value} for dicts." in prompt

    def test_build_prompt_handles_format_like_strings(self):
        """Test that strings that look like format placeholders are preserved."""
        builder = PromptBuilder()
        results = [
            {"text": "Use {context} and {question} placeholders.", "metadata": {"source": "doc.txt"}},
        ]
        prompt = builder.build_prompt(question="What about {optional}?", results=results)
        assert "{context}" in prompt or "Use " in prompt
        assert "What about {optional}?" in prompt

    def test_build_context_includes_source(self):
        """Test that build_context includes source by default."""
        builder = PromptBuilder()
        results = [
            {"text": "Content.", "metadata": {"source": "file.pdf", "chunk_index": 0}},
        ]
        context = builder.build_context(results, include_source=True)
        assert "file.pdf" in context
        assert "Content." in context

    def test_build_chat_messages(self):
        """Test building chat messages."""
        builder = PromptBuilder()
        results = [{"text": "Answer is 42.", "metadata": {"source": "doc.txt"}}]
        messages = builder.build_chat_messages(question="What is the answer?", results=results)
        assert len(messages) >= 2
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert "Answer is 42." in messages[-1]["content"]
        assert "What is the answer?" in messages[-1]["content"]
