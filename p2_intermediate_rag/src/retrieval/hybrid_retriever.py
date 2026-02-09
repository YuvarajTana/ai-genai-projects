"""
Hybrid Retriever
================
Combines vector search, BM25, and reranking for better retrieval.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from ..chunking.base import Chunk
from ..embeddings.base import BaseEmbedder
from .vector_store import VectorStore
from .bm25_search import BM25Search
from .reranker import Reranker


class HybridRetriever:
    """
    Hybrid retrieval system combining:
    - Vector similarity search (semantic)
    - BM25 keyword search (lexical)
    - Cross-encoder reranking
    """
    
    def __init__(
        self,
        embedder: BaseEmbedder,
        collection_name: str = "docs",
        persist_directory: str = "./chroma_db",
        use_bm25: bool = True,
        use_reranker: bool = True,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            embedder: Embedding model
            collection_name: Vector DB collection name
            persist_directory: Vector DB persist path
            use_bm25: Whether to use BM25 search
            use_reranker: Whether to use reranking
            vector_weight: Weight for vector search scores
            bm25_weight: Weight for BM25 scores
            reranker_model: Cross-encoder model name
        """
        self.embedder = embedder
        self.use_bm25 = use_bm25
        self.use_reranker = use_reranker
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        
        # Initialize components
        self.vector_store = VectorStore(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedder=embedder,
        )
        
        self.bm25 = BM25Search() if use_bm25 else None
        self.reranker = Reranker(model_name=reranker_model) if use_reranker else None
        
        # Track indexed chunks for BM25
        self._chunks: List[Chunk] = []
    
    def index(self, chunks: List[Chunk]) -> None:
        """
        Index chunks in both vector store and BM25.
        
        Args:
            chunks: List of chunks to index
        """
        if not chunks:
            return
        
        logger.debug("Indexing {} chunks (vector + BM25)", len(chunks))
        # Generate embeddings
        texts = [c.content for c in chunks]
        embeddings = self.embedder.embed_documents(texts).tolist()
        
        # Add to vector store
        self.vector_store.add_chunks(chunks, embeddings)
        
        # Add to BM25 index
        if self.use_bm25:
            self._chunks.extend(chunks)
            self.bm25.index(self._chunks)
    
    def retrieve(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[Dict[str, Any]] = None,
        use_reranker: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using hybrid search.
        
        Args:
            query: Query text
            k: Number of results to return
            filter_dict: Metadata filters (vector search only)
            
        Returns:
            List of retrieved chunks with scores
        """
        # Fetch more candidates for hybrid search
        fetch_k = k * 3 if (self.use_bm25 or self.use_reranker) else k
        logger.debug("Retrieving k={} with fetch_k={}, reranker={}", k, fetch_k, use_reranker)
        
        # Vector search
        vector_results = self.vector_store.search(
            query=query,
            k=fetch_k,
            filter_dict=filter_dict,
        )
        
        # Normalize vector scores (distances to similarities)
        for r in vector_results:
            # Convert distance to similarity score (0-1)
            r["vector_score"] = 1 / (1 + r["distance"])
        
        # BM25 search
        if self.use_bm25 and self.bm25 is not None:
            bm25_results = self.bm25.search(query, k=fetch_k)
            
            # Normalize BM25 scores
            if bm25_results:
                max_score = max(r["score"] for r in bm25_results)
                if max_score > 0:
                    for r in bm25_results:
                        r["bm25_score"] = r["score"] / max_score
                else:
                    for r in bm25_results:
                        r["bm25_score"] = 0
            
            # Combine results
            combined = self._combine_results(vector_results, bm25_results)
        else:
            combined = vector_results
            for r in combined:
                r["combined_score"] = r["vector_score"]
        
        # Sort by combined score
        combined.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        
        # Rerank if enabled (optionally overridable per request)
        rerank_enabled = self.use_reranker if use_reranker is None else use_reranker
        if rerank_enabled and self.reranker is not None:
            # Take top candidates for reranking
            candidates = combined[:fetch_k]
            reranked = self.reranker.rerank(query, candidates, top_n=k)
            return reranked
        
        return combined[:k]
    
    def _combine_results(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Combine vector and BM25 results with weighted scores.
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            
        Returns:
            Combined and deduplicated results
        """
        # Create lookup by ID
        combined = {}
        
        # Add vector results
        for r in vector_results:
            rid = r["id"]
            combined[rid] = r.copy()
            combined[rid]["bm25_score"] = 0  # Default
        
        # Add/merge BM25 results
        for r in bm25_results:
            rid = r["id"]
            if rid in combined:
                combined[rid]["bm25_score"] = r.get("bm25_score", 0)
            else:
                combined[rid] = r.copy()
                combined[rid]["vector_score"] = 0  # Default
        
        # Calculate combined scores
        for rid, r in combined.items():
            vector_score = r.get("vector_score", 0)
            bm25_score = r.get("bm25_score", 0)
            r["combined_score"] = (
                self.vector_weight * vector_score +
                self.bm25_weight * bm25_score
            )
        
        return list(combined.values())
    
    def clear(self) -> None:
        """Clear all indexed data."""
        self.vector_store.clear()
        if self.bm25:
            self.bm25.clear()
        self._chunks = []
    
    def count(self) -> int:
        """Get number of indexed chunks."""
        return self.vector_store.count()
