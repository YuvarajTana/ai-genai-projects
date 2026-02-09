"""
Loader Factory
==============
Factory for creating appropriate loaders based on file type.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Type

from .base import BaseLoader, Document
from .text_loader import TextLoader
from .pdf_loader import PDFLoader
from .docx_loader import DocxLoader
from .html_loader import HTMLLoader


class LoaderFactory:
    """
    Factory class for creating document loaders based on file extension.
    """
    
    # Mapping of extensions to loader classes
    _loaders: Dict[str, Type[BaseLoader]] = {}
    
    @classmethod
    def register_loader(cls, extensions: List[str], loader_class: Type[BaseLoader]):
        """
        Register a loader for specific file extensions.
        
        Args:
            extensions: List of file extensions (e.g., [".txt", ".md"])
            loader_class: The loader class to use
        """
        for ext in extensions:
            cls._loaders[ext.lower()] = loader_class
    
    @classmethod
    def get_loader(cls, file_path: str) -> BaseLoader:
        """
        Get the appropriate loader for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Appropriate loader instance
            
        Raises:
            ValueError: If no loader is registered for the file type
        """
        ext = Path(file_path).suffix.lower()
        
        if ext not in cls._loaders:
            raise ValueError(
                f"No loader registered for extension: {ext}. "
                f"Supported: {list(cls._loaders.keys())}"
            )
        
        return cls._loaders[ext]()
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get list of all supported file extensions."""
        return list(cls._loaders.keys())
    
    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """Check if a file type is supported."""
        ext = Path(file_path).suffix.lower()
        return ext in cls._loaders


# Register default loaders
LoaderFactory.register_loader([".txt", ".md", ".markdown"], TextLoader)
LoaderFactory.register_loader([".pdf"], PDFLoader)
LoaderFactory.register_loader([".docx"], DocxLoader)
LoaderFactory.register_loader([".html", ".htm"], HTMLLoader)


def load_document(file_path: str) -> Document:
    """
    Load a single document using the appropriate loader.
    
    Args:
        file_path: Path to the document
        
    Returns:
        Loaded Document
    """
    loader = LoaderFactory.get_loader(file_path)
    return loader.load(file_path)


def load_documents(
    directory: str,
    recursive: bool = True,
    extensions: Optional[List[str]] = None
) -> List[Document]:
    """
    Load all documents from a directory.
    
    Args:
        directory: Path to the directory
        recursive: Whether to search subdirectories
        extensions: Filter by extensions (None = all supported)
        
    Returns:
        List of loaded Documents
    """
    documents = []
    dir_path = Path(directory)
    
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    
    # Get supported extensions
    supported = extensions or LoaderFactory.get_supported_extensions()
    supported = [ext.lower() for ext in supported]
    
    # Find files
    pattern = "**/*" if recursive else "*"
    
    for file_path in dir_path.glob(pattern):
        if file_path.is_file() and file_path.suffix.lower() in supported:
            try:
                doc = load_document(str(file_path))
                documents.append(doc)
            except Exception as e:
                print(f"Warning: Failed to load {file_path}: {e}")
    
    return documents
