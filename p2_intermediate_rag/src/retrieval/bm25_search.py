"""
BM25 Keyword Search
===================
Sparse retrieval using BM25 algorithm.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from ..chunking.base import Chunk


class BM25Search:
    """
    BM25 keyword search for sparse retrieval.
    
    Complements vector search by finding exact keyword matches.
    """
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        stopwords: Optional[List[str]] = None,
    ):
        """
        Initialize BM25 search.
        
        Args:
            k1: Term frequency saturation parameter
            b: Document length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self._stopwords = set(stopwords or self._default_stopwords())
        self._bm25 = None
        self._chunks: List[Chunk] = []
        self._tokenized_corpus: List[List[str]] = []

    @staticmethod
    def _default_stopwords() -> List[str]:
        # Small, dependency-free stopword list (good enough for demos)
        return [
            "a", "an", "and", "are", "as", "at", "be", "but", "by",
            "for", "from", "has", "have", "he", "her", "his", "i",
            "in", "into", "is", "it", "its", "me", "my", "not", "of",
            "on", "or", "our", "she", "so", "that", "the", "their",
            "them", "there", "these", "they", "this", "to", "was",
            "we", "were", "what", "when", "where", "which", "who",
            "will", "with", "you", "your",
        ]
    
    def index(self, chunks: List[Chunk]) -> None:
        """
        Index chunks for BM25 search.
        
        Args:
            chunks: List of chunks to index
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank-bm25 required for BM25 search. "
                "Install with: pip install rank-bm25"
            )
        
        self._chunks = chunks
        
        # Tokenize corpus
        self._tokenized_corpus = [
            self._tokenize(chunk.content) for chunk in chunks
        ]
        
        # Create BM25 index
        self._bm25 = BM25Okapi(
            self._tokenized_corpus,
            k1=self.k1,
            b=self.b,
        )
    
    def search(
        self,
        query: str,
        k: int = 4,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            List of results with text, metadata, and score
        """
        if self._bm25 is None:
            raise ValueError("Index not built. Call index() first.")
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores
        scores = self._bm25.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]
        
        # Format results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                chunk = self._chunks[idx]
                results.append({
                    "id": chunk.id,
                    "text": chunk.content,
                    "metadata": chunk.metadata,
                    "score": float(scores[idx]),
                })
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for BM25.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Lowercase and split on word boundaries
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)

        # Remove very short tokens + stopwords
        tokens = [t for t in tokens if len(t) > 1 and t not in self._stopwords]

        # Very light stemming (dependency-free; improves plural/tense matching)
        def stem(tok: str) -> str:
            for suf in ("ing", "edly", "ed", "ly", "es", "s"):
                if tok.endswith(suf) and len(tok) > len(suf) + 2:
                    return tok[: -len(suf)]
            return tok

        return [stem(t) for t in tokens]
    
    def add_chunks(self, new_chunks: List[Chunk]) -> None:
        """
        Add new chunks to the index.
        
        Note: This rebuilds the entire index.
        For incremental updates, consider using a different approach.
        
        Args:
            new_chunks: New chunks to add
        """
        all_chunks = self._chunks + new_chunks
        self.index(all_chunks)
    
    def clear(self) -> None:
        """Clear the index."""
        self._bm25 = None
        self._chunks = []
        self._tokenized_corpus = []
