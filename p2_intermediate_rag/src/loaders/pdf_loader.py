"""
PDF Loader
==========
Load PDF documents with text extraction.
"""

from pathlib import Path
from typing import List, Optional

from .base import BaseLoader, Document


class PDFLoader(BaseLoader):
    """
    Loader for PDF documents.
    
    Uses pypdf for text extraction.
    Supports: .pdf
    """
    
    supported_extensions: List[str] = [".pdf"]
    
    def __init__(self, extract_images: bool = False):
        """
        Initialize the PDF loader.
        
        Args:
            extract_images: Whether to extract images (not implemented yet)
        """
        self.extract_images = extract_images
    
    def load(self, file_path: str) -> Document:
        """
        Load a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Document with extracted text and metadata
        """
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError("pypdf is required for PDF loading. Install with: pip install pypdf")
        
        path = self._validate_file(file_path)
        
        # Read PDF
        try:
            reader = PdfReader(str(path))
        except Exception as e:
            raise ValueError(f"Error reading PDF {file_path}: {e}")
        
        # Extract text from all pages
        pages_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(text)
        
        content = "\n\n".join(pages_text)
        
        # Create metadata
        metadata = self._create_metadata(path)
        metadata["format"] = "pdf"
        metadata["page_count"] = len(reader.pages)
        metadata["pages_with_text"] = len(pages_text)
        
        # Extract PDF metadata if available
        if reader.metadata:
            pdf_meta = reader.metadata
            if pdf_meta.title:
                metadata["title"] = pdf_meta.title
            if pdf_meta.author:
                metadata["author"] = pdf_meta.author
            if pdf_meta.subject:
                metadata["subject"] = pdf_meta.subject
            if pdf_meta.creation_date:
                metadata["pdf_created"] = str(pdf_meta.creation_date)
        
        return Document(content=content, metadata=metadata)
