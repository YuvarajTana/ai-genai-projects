"""
Base Embedder Interface
=======================
Abstract base class for embedding models.
"""

from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np


class BaseEmbedder(ABC):
    """
    Abstract base class for text embedding models.
    """
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        pass
    
    @abstractmethod
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Numpy array of embeddings (n_texts, dimension)
        """
        pass
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query (may have different behavior than documents).
        
        Args:
            query: Query text
            
        Returns:
            Query embedding
        """
        return self.embed(query)[0]
    
    def embed_documents(self, documents: List[str]) -> np.ndarray:
        """
        Embed multiple documents.
        
        Args:
            documents: List of document texts
            
        Returns:
            Document embeddings
        """
        return self.embed(documents)
