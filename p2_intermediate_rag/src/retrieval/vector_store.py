"""
Vector Store
============
ChromaDB vector database for storing and searching embeddings.
"""

from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings

from ..chunking.base import Chunk
from ..embeddings.base import BaseEmbedder


class VectorStore:
    """
    Vector store using ChromaDB for persistent storage.
    """
    
    def __init__(
        self,
        collection_name: str = "docs",
        persist_directory: str = "./chroma_db",
        embedder: Optional[BaseEmbedder] = None,
    ):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
            embedder: Embedding model to use
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedder = embedder
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    
    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """
        Add chunks to the vector store.
        
        Args:
            chunks: List of Chunk objects
            embeddings: Pre-computed embeddings (optional)
            
        Returns:
            List of chunk IDs
        """
        if not chunks:
            return []
        
        # Generate IDs
        ids = [chunk.id for chunk in chunks]
        
        # Extract content and metadata
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        # Generate embeddings if not provided
        if embeddings is None:
            if self.embedder is None:
                raise ValueError("No embedder provided for embedding generation")
            embeddings = self.embedder.embed_documents(documents).tolist()
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        
        return ids
    
    def search(
        self,
        query: str,
        k: int = 4,
        filter_dict: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.
        
        Args:
            query: Query text
            k: Number of results to return
            filter_dict: Metadata filters
            include_embeddings: Whether to include embeddings in results
            
        Returns:
            List of results with text, metadata, and distance
        """
        # Generate query embedding
        if self.embedder is None:
            raise ValueError("No embedder provided for query embedding")
        
        query_embedding = self.embedder.embed_query(query).tolist()
        
        return self.search_by_embedding(
            embedding=query_embedding,
            k=k,
            filter_dict=filter_dict,
            include_embeddings=include_embeddings,
        )
    
    def search_by_embedding(
        self,
        embedding: List[float],
        k: int = 4,
        filter_dict: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Search by embedding vector.
        
        Args:
            embedding: Query embedding
            k: Number of results
            filter_dict: Metadata filters
            include_embeddings: Whether to include embeddings
            
        Returns:
            List of results
        """
        # Prepare include list
        include = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include.append("embeddings")
        
        # Perform search
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=filter_dict,
            include=include,
        )
        
        # Format results
        formatted = []
        for i in range(len(results["ids"][0])):
            result = {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
            if include_embeddings and "embeddings" in results:
                result["embedding"] = results["embeddings"][0][i]
            formatted.append(result)
        
        return formatted
    
    def delete(self, ids: List[str]) -> None:
        """
        Delete chunks by ID.
        
        Args:
            ids: List of chunk IDs to delete
        """
        self.collection.delete(ids=ids)
    
    def count(self) -> int:
        """Get the number of chunks in the store."""
        return self.collection.count()
    
    def clear(self) -> None:
        """Clear all chunks from the collection."""
        # Delete and recreate the collection
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
