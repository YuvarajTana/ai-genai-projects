"""
Chunker Factory
===============
Factory for creating chunkers based on configuration.
"""

from typing import Dict, List, Optional, Type

from .base import BaseChunker, Chunk
from .character_chunker import CharacterChunker
from .recursive_chunker import RecursiveChunker
from .semantic_chunker import SemanticChunker
from ..loaders.base import Document


class ChunkerFactory:
    """
    Factory class for creating text chunkers.
    """
    
    # Mapping of strategy names to chunker classes
    _chunkers: Dict[str, Type[BaseChunker]] = {
        "character": CharacterChunker,
        "recursive": RecursiveChunker,
        "semantic": SemanticChunker,
    }
    
    @classmethod
    def register_chunker(cls, name: str, chunker_class: Type[BaseChunker]):
        """
        Register a new chunker type.
        
        Args:
            name: Strategy name
            chunker_class: Chunker class
        """
        cls._chunkers[name.lower()] = chunker_class
    
    @classmethod
    def get_chunker(
        cls,
        strategy: str = "recursive",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs,
    ) -> BaseChunker:
        """
        Get a chunker instance.
        
        Args:
            strategy: Chunking strategy name
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
            **kwargs: Additional chunker-specific arguments
            
        Returns:
            Chunker instance
        """
        strategy = strategy.lower()
        
        if strategy not in cls._chunkers:
            raise ValueError(
                f"Unknown chunking strategy: {strategy}. "
                f"Available: {list(cls._chunkers.keys())}"
            )
        
        chunker_class = cls._chunkers[strategy]
        return chunker_class(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs,
        )
    
    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """Get list of available chunking strategies."""
        return list(cls._chunkers.keys())


def chunk_document(
    document: Document,
    strategy: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    **kwargs,
) -> List[Chunk]:
    """
    Chunk a single document.
    
    Args:
        document: Document to chunk
        strategy: Chunking strategy
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        **kwargs: Additional chunker arguments
        
    Returns:
        List of Chunk objects
    """
    chunker = ChunkerFactory.get_chunker(
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs,
    )
    return chunker.chunk(document)


def chunk_documents(
    documents: List[Document],
    strategy: str = "recursive",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    **kwargs,
) -> List[Chunk]:
    """
    Chunk multiple documents.
    
    Args:
        documents: List of documents to chunk
        strategy: Chunking strategy
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        **kwargs: Additional chunker arguments
        
    Returns:
        List of all Chunk objects
    """
    chunker = ChunkerFactory.get_chunker(
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs,
    )
    
    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk(doc)
        all_chunks.extend(chunks)
    
    return all_chunks
