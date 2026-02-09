"""
Character-Based Chunker
=======================
Simple chunking based on character count with overlap.
"""

import re
from typing import List

from .base import BaseChunker, Chunk
from ..loaders.base import Document


class CharacterChunker(BaseChunker):
    """
    Simple character-based chunking with overlap.
    
    This is the baseline chunker from P1, included for comparison
    and as a fallback option.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        strip_whitespace: bool = True,
    ):
        """
        Initialize character chunker.
        
        Args:
            chunk_size: Target size for each chunk in characters
            chunk_overlap: Characters to overlap between chunks
            strip_whitespace: Whether to clean up whitespace
        """
        super().__init__(chunk_size, chunk_overlap)
        self.strip_whitespace = strip_whitespace
    
    def chunk(self, document: Document) -> List[Chunk]:
        """
        Split document into fixed-size character chunks.
        
        Args:
            document: Document to split
            
        Returns:
            List of Chunk objects
        """
        text = document.content
        
        # Clean whitespace if requested
        if self.strip_whitespace:
            text = self._clean_text(text)
        
        if not text:
            return []
        
        chunks = []
        start = 0
        index = 0
        
        while start < len(text):
            # Calculate end position
            end = min(start + self.chunk_size, len(text))
            
            # Extract chunk content
            content = text[start:end]
            
            # Create chunk
            chunk = self._create_chunk(
                content=content,
                document=document,
                index=index,
                start_char=start,
                end_char=end,
            )
            chunks.append(chunk)
            
            # Move to next position with overlap
            start = end - self.chunk_overlap
            index += 1
            
            # Prevent infinite loop
            if start < 0:
                start = 0
            if end == len(text):
                break
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """
        Clean up whitespace in text.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Replace multiple whitespace with single space
        text = re.sub(r"\s+", " ", text)
        return text.strip()
