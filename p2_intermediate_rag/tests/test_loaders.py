"""Tests for document loaders."""

import os
import tempfile
import pytest

from src.loaders import (
    TextLoader,
    LoaderFactory,
    load_document,
    Document,
)


class TestTextLoader:
    """Tests for TextLoader."""
    
    def test_load_txt_file(self):
        """Test loading a .txt file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello, World!\nThis is a test.")
            temp_path = f.name
        
        try:
            loader = TextLoader()
            doc = loader.load(temp_path)
            
            assert isinstance(doc, Document)
            assert "Hello, World!" in doc.content
            assert doc.metadata["format"] == "plaintext"
            assert doc.metadata["line_count"] == 2
        finally:
            os.unlink(temp_path)
    
    def test_load_md_file(self):
        """Test loading a .md file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("# Title\n\nSome content.")
            temp_path = f.name
        
        try:
            loader = TextLoader()
            doc = loader.load(temp_path)
            
            assert doc.metadata["format"] == "markdown"
        finally:
            os.unlink(temp_path)
    
    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        loader = TextLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/file.txt")


class TestLoaderFactory:
    """Tests for LoaderFactory."""
    
    def test_get_supported_extensions(self):
        """Test getting supported extensions."""
        extensions = LoaderFactory.get_supported_extensions()
        
        assert ".txt" in extensions
        assert ".pdf" in extensions
        assert ".docx" in extensions
        assert ".html" in extensions
    
    def test_is_supported(self):
        """Test checking if file type is supported."""
        assert LoaderFactory.is_supported("test.txt") is True
        assert LoaderFactory.is_supported("test.pdf") is True
        assert LoaderFactory.is_supported("test.xyz") is False
    
    def test_get_loader_for_txt(self):
        """Test getting loader for .txt files."""
        loader = LoaderFactory.get_loader("document.txt")
        assert isinstance(loader, TextLoader)
    
    def test_unsupported_extension(self):
        """Test error for unsupported extension."""
        with pytest.raises(ValueError) as exc_info:
            LoaderFactory.get_loader("file.xyz")
        
        assert "No loader registered" in str(exc_info.value)


class TestLoadDocument:
    """Tests for load_document function."""
    
    def test_load_document(self):
        """Test loading a document via convenience function."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content")
            temp_path = f.name
        
        try:
            doc = load_document(temp_path)
            assert isinstance(doc, Document)
            assert "Test content" in doc.content
        finally:
            os.unlink(temp_path)
