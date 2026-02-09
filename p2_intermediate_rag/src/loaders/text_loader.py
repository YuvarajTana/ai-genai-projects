"""
Text File Loader
================
Load plain text and markdown files.
"""

from pathlib import Path
from typing import List

from .base import BaseLoader, Document


class TextLoader(BaseLoader):
    """
    Loader for plain text and markdown files.
    
    Supports: .txt, .md, .markdown
    """
    
    supported_extensions: List[str] = [".txt", ".md", ".markdown"]
    
    def __init__(self, encoding: str = "utf-8"):
        """
        Initialize the text loader.
        
        Args:
            encoding: File encoding (default: utf-8)
        """
        self.encoding = encoding
    
    def load(self, file_path: str) -> Document:
        """
        Load a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Document with content and metadata
        """
        path = self._validate_file(file_path)
        
        # Read file content
        try:
            content = path.read_text(encoding=self.encoding, errors="ignore")
        except Exception as e:
            raise ValueError(f"Error reading file {file_path}: {e}")
        
        # Create metadata
        metadata = self._create_metadata(path)
        metadata["encoding"] = self.encoding
        metadata["line_count"] = content.count("\n") + 1
        
        # Check if markdown
        if path.suffix.lower() in [".md", ".markdown"]:
            metadata["format"] = "markdown"
        else:
            metadata["format"] = "plaintext"
        
        return Document(content=content, metadata=metadata)
