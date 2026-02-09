"""
Sentence Transformer Embedder
=============================
Generate embeddings using Sentence Transformers.
"""

from typing import List, Optional, Union

import numpy as np

from .base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Embedder using Sentence Transformers models.
    
    Popular models:
    - all-MiniLM-L6-v2: Fast, good quality (384 dims)
    - all-mpnet-base-v2: Better quality, slower (768 dims)
    - BAAI/bge-small-en-v1.5: Good balance (384 dims)
    - BAAI/bge-large-en-v1.5: Best quality (1024 dims)
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize: bool = True,
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        """
        Initialize the embedder.
        
        Args:
            model_name: Name of the Sentence Transformer model
            normalize: Whether to L2-normalize embeddings
            batch_size: Batch size for encoding
            device: Device to use (None for auto-detect)
        """
        self.model_name = model_name
        self.normalize = normalize
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self._dimension = None
    
    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers required. "
                    "Install with: pip install sentence-transformers"
                )
            
            self._model = SentenceTransformer(
                self.model_name,
                device=self.device,
            )
        return self._model
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        if self._dimension is None:
            # Get dimension from model
            self._dimension = self.model.get_sentence_embedding_dimension()
        return self._dimension
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Numpy array of embeddings
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
        )
        
        return np.array(embeddings)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query.
        
        Some models have special query prefixes.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding
        """
        # Check if model expects query prefix (e.g., BGE models)
        if "bge" in self.model_name.lower():
            query = f"Represent this sentence for searching relevant passages: {query}"
        
        return self.embed(query)[0]
