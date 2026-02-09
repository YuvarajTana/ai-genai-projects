"""
Recursive Text Chunker
======================
Hierarchical chunking that respects document structure.
"""

from typing import List, Optional

from .base import BaseChunker, Chunk
from ..loaders.base import Document


class RecursiveChunker(BaseChunker):
    """
    Recursive text chunker that tries to split on natural boundaries.
    
    Attempts to split on larger separators first (paragraphs),
    then falls back to smaller separators (sentences, words).
    """
    
    DEFAULT_SEPARATORS = [
        "\n\n",  # Paragraphs
        "\n",    # Lines
        ". ",    # Sentences
        "! ",    # Exclamations
        "? ",    # Questions
        "; ",    # Semicolons
        ", ",    # Commas
        " ",     # Words
        "",      # Characters (fallback)
    ]
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
    ):
        """
        Initialize recursive chunker.
        
        Args:
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
            separators: List of separators to try (in order)
            keep_separator: Whether to keep separators in chunks
        """
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.keep_separator = keep_separator
    
    def chunk(self, document: Document) -> List[Chunk]:
        """
        Recursively split document into chunks.
        
        Args:
            document: Document to split
            
        Returns:
            List of Chunk objects
        """
        text = document.content
        
        if not text:
            return []
        
        # Recursively split the text
        split_texts = self._recursive_split(text, self.separators)
        
        # Merge small chunks
        merged_texts = self._merge_splits(split_texts)
        
        # Create Chunk objects with position tracking
        chunks = []
        current_pos = 0
        
        for i, content in enumerate(merged_texts):
            # Find the actual position in original text
            needle = content[:100].strip()
            start_pos = text.find(needle, current_pos) if needle else -1
            if start_pos == -1:
                start_pos = current_pos
            
            end_pos = start_pos + len(content)
            
            chunk = self._create_chunk(
                content=content,
                document=document,
                index=i,
                start_char=start_pos,
                end_char=end_pos,
            )
            chunks.append(chunk)
            current_pos = max(0, end_pos - self.chunk_overlap)
        
        return chunks
    
    def _recursive_split(
        self,
        text: str,
        separators: List[str],
    ) -> List[str]:
        """
        Recursively split text using separators.
        
        Args:
            text: Text to split
            separators: Remaining separators to try
            
        Returns:
            List of text splits
        """
        final_chunks: List[str] = []
        
        # Get the current separator
        separator = separators[0] if separators else ""
        remaining_separators = separators[1:] if len(separators) > 1 else []
        
        # Split the text
        if separator:
            splits = text.split(separator)
        else:
            # Character-level split (fallback)
            splits = list(text)
        
        # Process each split in-order to preserve original text order
        for idx, split in enumerate(splits):
            if not split:
                continue
                
            # Add separator back if keeping it
            if self.keep_separator and separator and idx != len(splits) - 1:
                split = split + separator
            
            if len(split) <= self.chunk_size:
                final_chunks.append(split)
            else:
                # Need to split further
                if remaining_separators:
                    # Recursively split with next separator
                    sub_splits = self._recursive_split(split, remaining_separators)
                    final_chunks.extend(sub_splits)
                else:
                    # Force split at chunk_size
                    for i in range(0, len(split), self.chunk_size):
                        final_chunks.append(split[i:i + self.chunk_size])
        
        return final_chunks
    
    def _merge_splits(self, splits: List[str]) -> List[str]:
        """
        Merge small splits into larger chunks.
        
        Args:
            splits: List of text splits
            
        Returns:
            Merged chunks
        """
        if not splits:
            return []
        
        merged = []
        current_chunk = ""
        
        for split in splits:
            # Preserve paragraph boundaries: if the previous split ended a paragraph,
            # don't merge the next paragraph into the same chunk even if it fits.
            if current_chunk and current_chunk.endswith("\n\n"):
                merged.append(current_chunk.strip())
                current_chunk = ""

            # Check if adding this split would exceed chunk_size
            test_chunk = current_chunk + split if current_chunk else split
            
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # Save current chunk and start new one
                if current_chunk:
                    merged.append(current_chunk.strip())
                
                # Handle overlap
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + split
                else:
                    current_chunk = split
                
                # If still too large, force add it
                while len(current_chunk) > self.chunk_size:
                    merged.append(current_chunk[:self.chunk_size].strip())
                    current_chunk = current_chunk[self.chunk_size - self.chunk_overlap:]
        
        # Add the last chunk
        if current_chunk:
            merged.append(current_chunk.strip())
        
        return merged
