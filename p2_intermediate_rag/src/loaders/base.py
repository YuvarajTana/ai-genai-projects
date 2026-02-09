"""
Base Loader Interface
=====================
Abstract base class for all document loaders.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    """
    Represents a loaded document with content and metadata.
    
    Attributes:
        content: The text content of the document
        metadata: Additional information about the document
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set default metadata values."""
        if "created_at" not in self.metadata:
            self.metadata["created_at"] = datetime.now().isoformat()
    
    def __len__(self) -> int:
        """Return the length of the content."""
        return len(self.content)
    
    def __str__(self) -> str:
        """String representation."""
        source = self.metadata.get("source", "unknown")
        return f"Document(source={source}, length={len(self)})"


class BaseLoader(ABC):
    """
    Abstract base class for document loaders.
    
    All loaders must implement the `load` method that returns a Document.
    """
    
    # File extensions this loader supports
    supported_extensions: List[str] = []
    
    @abstractmethod
    def load(self, file_path: str) -> Document:
        """
        Load a document from the given file path.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            Document object with content and metadata
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file format is not supported
        """
        pass
    
    def _validate_file(self, file_path: str) -> Path:
        """
        Validate that the file exists and has a supported extension.
        
        Args:
            file_path: Path to validate
            
        Returns:
            Path object
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If extension not supported
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not path.is_file():
            raise ValueError(f"Not a file: {file_path}")
        
        ext = path.suffix.lower()
        if self.supported_extensions and ext not in self.supported_extensions:
            raise ValueError(
                f"Unsupported file extension: {ext}. "
                f"Supported: {self.supported_extensions}"
            )
        
        return path
    
    def _create_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Create basic metadata for a document.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with metadata
        """
        stat = file_path.stat()
        
        return {
            "source": str(file_path),
            "filename": file_path.name,
            "extension": file_path.suffix.lower(),
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }
