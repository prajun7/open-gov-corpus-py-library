"""
Vector store operations for embeddings
"""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from pathlib import Path

from .exceptions import VectorStoreError


class VectorStore(ABC):
    """Abstract base class for vector stores"""
    
    @abstractmethod
    def add(self, text: str, embedding: List[float], metadata: Dict[str, Any], id: str):
        """Add embedding to store"""
        pass
    
    @abstractmethod
    def query(self, embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Query similar embeddings"""
        pass


class ChromaVectorStore(VectorStore):
    """ChromaDB vector store"""
    
    def __init__(self, persist_directory: str):
        """
        Initialize ChromaDB store
        
        Args:
            persist_directory: Directory to persist data
        """
        try:
            import chromadb
        except ImportError:
            raise VectorStoreError("chromadb package not installed")
        
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="opengovcorpus",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add(self, text: str, embedding: List[float], metadata: Dict[str, Any], id: str):
        """Add embedding to store"""
        try:
            self.collection.add(
                embeddings=[embedding],
                documents=[text],
                metadatas=[metadata],
                ids=[id]
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to add to vector store: {e}")
    
    def query(self, embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Query similar embeddings"""
        try:
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=top_k
            )
            
            # Format results
            formatted = []
            for i in range(len(results['ids'][0])):
                formatted.append({
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })
            
            return formatted
        except Exception as e:
            raise VectorStoreError(f"Query failed: {e}")


def create_vector_store(store_type: str, persist_directory: str) -> VectorStore:
    """
    Create a vector store
    
    Args:
        store_type: Type of vector store (currently only 'chroma')
        persist_directory: Directory to persist data
        
    Returns:
        VectorStore instance
    """
    if store_type.lower() == "chroma":
        return ChromaVectorStore(persist_directory)
    else:
        raise VectorStoreError(f"Unknown vector store type: {store_type}")