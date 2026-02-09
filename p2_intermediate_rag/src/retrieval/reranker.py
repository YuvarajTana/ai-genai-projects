"""
Cross-Encoder Reranker
======================
Rerank search results using a cross-encoder model.
"""

from typing import Any, Dict, List, Optional, Tuple


class Reranker:
    """
    Rerank search results using a cross-encoder model.
    
    Cross-encoders are more accurate than bi-encoders for ranking
    but slower (they process query-document pairs together).
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ):
        """
        Initialize the reranker.
        
        Args:
            model_name: Cross-encoder model name
            device: Device to use (None for auto-detect)
        """
        self.model_name = model_name
        self.device = device
        self._model = None
    
    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for reranking. "
                    "Install with: pip install sentence-transformers"
                )
            
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
            )
        return self._model
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results based on relevance to query.
        
        Args:
            query: Query text
            results: List of search results (must have 'text' key)
            top_n: Number of results to return (None = all)
            
        Returns:
            Reranked results with 'rerank_score' added
        """
        if not results:
            return []
        
        # Prepare query-document pairs
        pairs = [(query, r["text"]) for r in results]
        
        # Get reranking scores
        scores = self.model.predict(pairs)
        
        # Add scores to results
        reranked = []
        for result, score in zip(results, scores):
            result_copy = result.copy()
            result_copy["rerank_score"] = float(score)
            reranked.append(result_copy)
        
        # Sort by rerank score (descending)
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # Return top_n if specified
        if top_n is not None:
            reranked = reranked[:top_n]
        
        return reranked
    
    def rerank_with_threshold(
        self,
        query: str,
        results: List[Dict[str, Any]],
        threshold: float = 0.0,
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank and filter results by score threshold.
        
        Args:
            query: Query text
            results: Search results
            threshold: Minimum rerank score
            top_n: Maximum results to return
            
        Returns:
            Filtered and reranked results
        """
        reranked = self.rerank(query, results, top_n=None)
        
        # Filter by threshold
        filtered = [r for r in reranked if r["rerank_score"] >= threshold]
        
        if top_n is not None:
            filtered = filtered[:top_n]
        
        return filtered
