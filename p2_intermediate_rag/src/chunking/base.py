"""
Base Chunker Interface
======================
Abstract base class for text chunking strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..loaders.base import Document


@dataclass
class Chunk:
    """
    Represents a text chunk from a document.
    
    Attributes:
        content: The text content of the chunk
        metadata: Metadata inherited from document + chunk-specific info
        index: Position of this chunk in the document
        start_char: Starting character position in original document
        end_char: Ending character position in original document
    """
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    index: int = 0
    start_char: int = 0
    end_char: int = 0
    
    def __len__(self) -> int:
        """Return the length of the content."""
        return len(self.content)
    
    def __str__(self) -> str:
        """String representation."""
        source = self.metadata.get("source", "unknown")
        return f"Chunk(source={source}, index={self.index}, length={len(self)})"
    
    @property
    def id(self) -> str:
        """Generate a unique ID for this chunk."""
        # Prefer full source path if available to avoid collisions
        # when different directories contain same filename.
        source = self.metadata.get("source") or self.metadata.get("filename") or "unknown"
        start = self.metadata.get("chunk_start", self.start_char)
        end = self.metadata.get("chunk_end", self.end_char)
        return f"{source}::chunk_{self.index}::{start}-{end}"


class BaseChunker(ABC):
    """
    Abstract base class for text chunking.
    
    All chunkers must implement the `chunk` method that splits
    a document into a list of Chunks.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target size for each chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """
        Split a document into chunks.
        
        Args:
            document: Document to split
            
        Returns:
            List of Chunk objects
        """
        pass
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Convenience method to chunk raw text.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to include
            
        Returns:
            List of Chunk objects
        """
        doc = Document(content=text, metadata=metadata or {})
        return self.chunk(doc)
    
    def _create_chunk(
        self,
        content: str,
        document: Document,
        index: int,
        start_char: int,
        end_char: int,
    ) -> Chunk:
        """
        Create a Chunk with proper metadata.
        
        Args:
            content: Chunk text content
            document: Source document
            index: Chunk index
            start_char: Start position
            end_char: End position
            
        Returns:
            Chunk object
        """
        # Inherit document metadata
        metadata = document.metadata.copy()
        metadata["chunk_index"] = index
        metadata["chunk_start"] = start_char
        metadata["chunk_end"] = end_char
        
        return Chunk(
            content=content,
            metadata=metadata,
            index=index,
            start_char=start_char,
            end_char=end_char,
        )
