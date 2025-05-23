"""
Retriever Agent Module.

This module implements an agent that handles vector database operations
for retrieval-augmented generation (RAG).
"""
import os
import logging
import pickle
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import numpy as np
import sys

from agents.base_agent import BaseAgent

# Mock implementation for SentenceTransformer
class SentenceTransformer:
    def __init__(self, *args, **kwargs):
        logging.warning("Using mock SentenceTransformer implementation")
        
    def encode(self, sentences, **kwargs):
        # Return random embeddings
        if isinstance(sentences, list):
            return np.random.rand(len(sentences), 384)
        else:
            return np.random.rand(1, 384)

# Mock cosine_similarity function
def cosine_similarity(a, b):
    # Return mock similarity matrix
    return np.ones((a.shape[0], b.shape[0]))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RetrieverAgent(BaseAgent):
    """Agent for vector database retrieval operations."""
    
    def __init__(
        self, 
        agent_id: Optional[str] = None,
        embedding_model: str = 'all-MiniLM-L6-v2',
        index_path: Optional[str] = None,
        dimension: int = 384
    ):
        """
        Initialize the retriever agent.
        
        Args:
            agent_id: Optional unique identifier for the agent
            embedding_model: Name of the sentence-transformers model to use
            index_path: Path to save/load the FAISS index
            dimension: Dimension of the embedding vectors
        """
        super().__init__(agent_id)
        
        # Initialize embedding model
        self.embedding_model_name = embedding_model
        self.embedding_model = SentenceTransformer(embedding_model)
        self.dimension = dimension
        
        # Initialize vector store
        self.index_path = index_path or os.path.join('data', 'vector_index.pkl')
        self.documents_path = os.path.join('data', 'documents.pkl')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # Initialize or load index
        self.vectors = []
        self.documents = []
        self.load_or_create_index()
        
        self.log_activity(f"Retriever Agent initialized with model {embedding_model}")
    
    def load_or_create_index(self):
        """Load existing index or create a new one if it doesn't exist."""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.documents_path):
                # Load vectors
                with open(self.index_path, 'rb') as f:
                    self.vectors = pickle.load(f)
                self.log_activity(f"Loaded existing vector index with {len(self.vectors)} vectors")
                
                # Load documents
                with open(self.documents_path, 'rb') as f:
                    self.documents = pickle.load(f)
                self.log_activity(f"Loaded {len(self.documents)} documents from {self.documents_path}")
            else:
                self.vectors = []
                self.documents = []
                self.log_activity("Created new vector index")
        except Exception as e:
            self.log_activity(f"Error loading index: {str(e)}", "error")
            self.vectors = []
            self.documents = []
            self.log_activity("Created new vector index after error")
    
    def save_index(self):
        """Save the vector index and documents to disk."""
        try:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            
            # Save vectors
            with open(self.index_path, 'wb') as f:
                pickle.dump(self.vectors, f)
            self.log_activity(f"Saved vector index with {len(self.vectors)} vectors")
                
            # Save documents
            with open(self.documents_path, 'wb') as f:
                pickle.dump(self.documents, f)
            self.log_activity(f"Saved {len(self.documents)} documents")
        except Exception as e:
            self.log_activity(f"Error saving index: {str(e)}", "error")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a text string using the sentence transformer model.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        return self.embedding_model.encode([text])[0]
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Embed a batch of text strings.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embedding vectors
        """
        return self.embedding_model.encode(texts)
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data for vector database operations.
        
        Args:
            input_data: Dictionary containing request parameters
                - action: Type of operation to perform
                - params: Parameters for the specific action
            
        Returns:
            Dictionary containing operation results
        """
        if not self.validate_input(input_data):
            self.log_activity("Invalid input data", "error")
            return {"error": "Invalid input data"}
        
        action = input_data.get("action", "")
        params = input_data.get("params", {})
        
        self.log_activity(f"Processing action: {action} with params: {params}")
        
        try:
            if action == "add_documents":
                texts = params.get("texts", [])
                metadata = params.get("metadata", [])
                
                if len(texts) == 0:
                    return {"error": "No texts provided"}
                
                if metadata and len(texts) != len(metadata):
                    return {"error": "Number of texts and metadata entries must match"}
                
                # If no metadata provided, create empty metadata
                if not metadata:
                    metadata = [{} for _ in texts]
                
                # Add timestamp to metadata
                timestamp = datetime.now().isoformat()
                for meta in metadata:
                    meta['timestamp'] = timestamp
                
                # Embed texts
                embeddings = self.embed_batch(texts)
                
                # Add to vectors list
                for embedding in embeddings:
                    self.vectors.append(embedding)
                
                # Add documents with metadata
                for i, (text, meta) in enumerate(zip(texts, metadata)):
                    doc_id = len(self.documents)
                    self.documents.append({
                        'id': doc_id,
                        'text': text,
                        'metadata': meta
                    })
                
                # Save index
                self.save_index()
                
                return {
                    "success": True,
                    "count": len(texts),
                    "total_documents": len(self.documents)
                }
                
            elif action == "search":
                query = params.get("query", "")
                k = params.get("k", 5)
                
                if not query:
                    return {"error": "Query is required"}
                
                # Embed query
                query_embedding = self.embed_text(query)
                
                # If no vectors, return empty results
                if not self.vectors:
                    return {
                        "results": [],
                        "count": 0,
                        "query": query
                    }
                
                # Calculate similarities
                similarities = cosine_similarity(
                    [query_embedding],
                    self.vectors
                )[0]
                
                # Get top-k indices and scores
                top_indices = np.argsort(similarities)[::-1][:k]
                top_scores = similarities[top_indices]
                
                # Get documents
                results = []
                for i, idx in enumerate(top_indices):
                    if idx < len(self.documents):
                        doc = self.documents[idx]
                        results.append({
                            'id': doc['id'],
                            'text': doc['text'],
                            'metadata': doc['metadata'],
                            'distance': float(1.0 - top_scores[i])  # Convert similarity to distance
                        })
                
                return {
                    "results": results,
                    "count": len(results),
                    "query": query
                }
                
            elif action == "clear_index":
                # Clear vectors and documents
                self.vectors = []
                self.documents = []
                self.save_index()
                
                return {
                    "success": True,
                    "message": "Index cleared"
                }
                
            elif action == "get_stats":
                return {
                    "total_documents": len(self.documents),
                    "total_vectors": len(self.vectors),
                    "dimension": self.dimension,
                    "embedding_model": self.embedding_model_name
                }
                
            else:
                self.log_activity(f"Unknown action: {action}", "warning")
                return {"error": f"Unknown action: {action}"}
                
        except Exception as e:
            self.log_activity(f"Error processing action {action}: {str(e)}", "error")
            return {
                "error": str(e),
                "action": action,
                "params": params
            }
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(input_data, dict):
            return False
            
        if "action" not in input_data:
            return False
            
        # Validate specific actions
        action = input_data.get("action", "")
        params = input_data.get("params", {})
        
        if action == "search" and "query" not in params:
            return False
            
        if action == "add_documents" and "texts" not in params:
            return False
            
        return True
