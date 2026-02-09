"""
Document Loaders
================
Load documents from various file formats.
"""

from .base import BaseLoader, Document
from .text_loader import TextLoader
from .pdf_loader import PDFLoader
from .docx_loader import DocxLoader
from .html_loader import HTMLLoader
from .loader_factory import LoaderFactory, load_document, load_documents

__all__ = [
    "BaseLoader",
    "Document",
    "TextLoader",
    "PDFLoader",
    "DocxLoader",
    "HTMLLoader",
    "LoaderFactory",
    "load_document",
    "load_documents",
]
