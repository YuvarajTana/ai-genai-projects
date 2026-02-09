"""
Embedding Models
================
Generate vector embeddings for text chunks.
"""

from .base import BaseEmbedder
from .sentence_transformer import SentenceTransformerEmbedder

__all__ = [
    "BaseEmbedder",
    "SentenceTransformerEmbedder",
]
