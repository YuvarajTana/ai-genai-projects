"""
HTML Loader
===========
Load HTML documents with text extraction.
"""

from pathlib import Path
from typing import List

from .base import BaseLoader, Document


class HTMLLoader(BaseLoader):
    """
    Loader for HTML documents.
    
    Uses BeautifulSoup for parsing and text extraction.
    Supports: .html, .htm
    """
    
    supported_extensions: List[str] = [".html", ".htm"]
    
    def __init__(self, encoding: str = "utf-8", remove_scripts: bool = True):
        """
        Initialize the HTML loader.
        
        Args:
            encoding: File encoding
            remove_scripts: Whether to remove script and style tags
        """
        self.encoding = encoding
        self.remove_scripts = remove_scripts
    
    def load(self, file_path: str) -> Document:
        """
        Load an HTML file.
        
        Args:
            file_path: Path to the HTML file
            
        Returns:
            Document with extracted text and metadata
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("beautifulsoup4 is required for HTML loading. Install with: pip install beautifulsoup4")
        
        path = self._validate_file(file_path)
        
        # Read file
        try:
            html_content = path.read_text(encoding=self.encoding, errors="ignore")
        except Exception as e:
            raise ValueError(f"Error reading HTML {file_path}: {e}")
        
        # Parse HTML (prefer lxml, but gracefully fall back)
        try:
            soup = BeautifulSoup(html_content, "lxml")
        except Exception:
            soup = BeautifulSoup(html_content, "html.parser")
        
        # Remove script and style elements if requested
        if self.remove_scripts:
            for element in soup(["script", "style", "noscript"]):
                element.decompose()
        
        # Extract text
        text = soup.get_text(separator="\n", strip=True)
        
        # Clean up multiple newlines
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        content = "\n\n".join(lines)
        
        # Create metadata
        metadata = self._create_metadata(path)
        metadata["format"] = "html"
        metadata["encoding"] = self.encoding
        
        # Extract title if present
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            metadata["title"] = title_tag.string.strip()
        
        # Extract meta description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            metadata["description"] = meta_desc["content"]
        
        # Count elements
        metadata["link_count"] = len(soup.find_all("a"))
        metadata["image_count"] = len(soup.find_all("img"))
        
        return Document(content=content, metadata=metadata)
