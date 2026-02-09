"""
Semantic Chunker
================
Chunk text based on semantic similarity between sentences.
"""

import re
from typing import List, Optional, Tuple

import numpy as np

from .base import BaseChunker, Chunk
from ..loaders.base import Document


class SemanticChunker(BaseChunker):
    """
    Semantic chunker that groups semantically similar sentences.
    
    Uses sentence embeddings to find natural breakpoints where
    the topic or meaning changes significantly.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        breakpoint_threshold: float = 0.3,
        min_chunk_size: int = 100,
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize semantic chunker.
        
        Args:
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Target overlap (approximate with semantic chunking)
            breakpoint_threshold: Similarity drop threshold for new chunk
            min_chunk_size: Minimum characters per chunk
            embedding_model: Sentence transformer model name
        """
        super().__init__(chunk_size, chunk_overlap)
        self.breakpoint_threshold = breakpoint_threshold
        self.min_chunk_size = min_chunk_size
        self.embedding_model = embedding_model or "sentence-transformers/all-MiniLM-L6-v2"
        self._embedder = None
    
    @property
    def embedder(self):
        """Lazy load the embedding model."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.embedding_model)
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for semantic chunking. "
                    "Install with: pip install sentence-transformers"
                )
        return self._embedder
    
    def chunk(self, document: Document) -> List[Chunk]:
        """
        Split document based on semantic boundaries.
        
        Args:
            document: Document to split
            
        Returns:
            List of semantically coherent Chunk objects
        """
        text = document.content
        
        if not text:
            return []
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) <= 1:
            # Single sentence or unable to split
            return [self._create_chunk(
                content=text,
                document=document,
                index=0,
                start_char=0,
                end_char=len(text),
            )]
        
        # Get sentence embeddings
        embeddings = self.embedder.encode(
            sentences,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        
        # Find semantic breakpoints
        breakpoints = self._find_breakpoints(embeddings)
        
        # Create chunks based on breakpoints
        chunks = self._create_semantic_chunks(
            sentences=sentences,
            breakpoints=breakpoints,
            document=document,
            original_text=text,
        )
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting regex
        # Handles common sentence endings while preserving abbreviations
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        
        sentences = re.split(sentence_pattern, text)
        
        # Clean and filter
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _find_breakpoints(self, embeddings: np.ndarray) -> List[int]:
        """
        Find semantic breakpoints using embedding similarity.
        
        Args:
            embeddings: Sentence embeddings array
            
        Returns:
            List of indices where breakpoints should occur
        """
        if len(embeddings) < 2:
            return []
        
        # Calculate cosine similarity between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = np.dot(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
        
        # Find breakpoints where similarity drops significantly
        breakpoints = []
        
        for i, sim in enumerate(similarities):
            # Check if similarity is below threshold
            if sim < (1 - self.breakpoint_threshold):
                breakpoints.append(i + 1)
        
        return breakpoints
    
    def _create_semantic_chunks(
        self,
        sentences: List[str],
        breakpoints: List[int],
        document: Document,
        original_text: str,
    ) -> List[Chunk]:
        """
        Create chunks based on semantic breakpoints.
        
        Args:
            sentences: List of sentences
            breakpoints: Indices of breakpoints
            document: Source document
            original_text: Original text for position tracking
            
        Returns:
            List of Chunk objects
        """
        chunks: List[Chunk] = []

        # Add start and end as implicit breakpoints
        all_breaks = [0] + breakpoints + [len(sentences)]

        def _find_start(sentence: str, start_from: int) -> int:
            needle = sentence[:100].strip()
            if not needle:
                return start_from
            pos = original_text.find(needle, start_from)
            return pos if pos != -1 else start_from

        current_pos = 0
        i = 0
        while i < len(all_breaks) - 1:
            start_idx = all_breaks[i]
            end_break_i = i + 1
            end_idx = all_breaks[end_break_i]

            # Ensure minimum chunk size by extending to subsequent breakpoint(s)
            chunk_sentences = sentences[start_idx:end_idx]
            content = " ".join(chunk_sentences)
            while (
                len(content) < self.min_chunk_size
                and end_break_i < len(all_breaks) - 1
            ):
                end_break_i += 1
                end_idx = all_breaks[end_break_i]
                chunk_sentences = sentences[start_idx:end_idx]
                content = " ".join(chunk_sentences)

            # Handle max chunk size by splitting if too large
            if len(content) > self.chunk_size:
                start_char = _find_start(chunk_sentences[0], current_pos)
                sub_chunks = self._split_large_chunk(
                    content=content,
                    document=document,
                    start_index=len(chunks),
                    start_pos=start_char,
                )
                chunks.extend(sub_chunks)
                current_pos = start_char + len(content)
            else:
                start_char = _find_start(chunk_sentences[0], current_pos)
                end_char = start_char + len(content)

                chunk = self._create_chunk(
                    content=content,
                    document=document,
                    index=len(chunks),
                    start_char=start_char,
                    end_char=end_char,
                )
                chunks.append(chunk)
                current_pos = end_char

            # Continue from the breakpoint we actually consumed
            i = end_break_i
        
        # Update indices after all processing
        for i, chunk in enumerate(chunks):
            chunk.index = i
            chunk.metadata["chunk_index"] = i
        
        return chunks
    
    def _split_large_chunk(
        self,
        content: str,
        document: Document,
        start_index: int,
        start_pos: int,
    ) -> List[Chunk]:
        """
        Split a large chunk into smaller ones.
        
        Args:
            content: Content to split
            document: Source document
            start_index: Starting chunk index
            start_pos: Starting character position
            
        Returns:
            List of smaller Chunk objects
        """
        chunks = []
        current_pos = start_pos
        
        for i in range(0, len(content), self.chunk_size - self.chunk_overlap):
            chunk_content = content[i:i + self.chunk_size]
            
            if not chunk_content.strip():
                continue
            
            chunk = self._create_chunk(
                content=chunk_content,
                document=document,
                index=start_index + len(chunks),
                start_char=current_pos,
                end_char=current_pos + len(chunk_content),
            )
            chunks.append(chunk)
            current_pos += len(chunk_content) - self.chunk_overlap
        
        return chunks
