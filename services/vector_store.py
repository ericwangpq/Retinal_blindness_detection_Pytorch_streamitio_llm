"""
Vector store service for RAG system
Handles embeddings generation, storage, and similarity search using FAISS
"""

import json
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple, Optional
import requests
from pathlib import Path

# Simple FAISS-like implementation for vector storage
class SimpleVectorStore:
    def __init__(self, dimension: int = 1536):
        """
        Initialize vector store
        
        Args:
            dimension: Dimension of embedding vectors (OpenAI embeddings are 1536)
        """
        self.dimension = dimension
        self.vectors = []
        self.metadata = []
        self.index_to_id = {}
        self.next_id = 0
    
    def add_vectors(self, vectors: np.ndarray, metadata: List[Dict]):
        """Add vectors and metadata to store"""
        for i, (vector, meta) in enumerate(zip(vectors, metadata)):
            self.vectors.append(vector)
            self.metadata.append(meta)
            self.index_to_id[len(self.vectors) - 1] = self.next_id
            meta['vector_id'] = self.next_id
            self.next_id += 1
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> List[Tuple[Dict, float]]:
        """Search for similar vectors"""
        if not self.vectors:
            return []
        
        # Calculate cosine similarity
        similarities = []
        query_norm = np.linalg.norm(query_vector)
        
        for i, vector in enumerate(self.vectors):
            vector_norm = np.linalg.norm(vector)
            if vector_norm == 0 or query_norm == 0:
                similarity = 0
            else:
                similarity = np.dot(query_vector, vector) / (query_norm * vector_norm)
            similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        results = []
        for i, (idx, score) in enumerate(similarities[:k]):
            results.append((self.metadata[idx], score))
        
        return results
    
    def save(self, filepath: str):
        """Save vector store to disk"""
        data = {
            'vectors': [vector.tolist() for vector in self.vectors],
            'metadata': self.metadata,
            'index_to_id': self.index_to_id,
            'next_id': self.next_id,
            'dimension': self.dimension
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load(self, filepath: str):
        """Load vector store from disk"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.vectors = [np.array(vector) for vector in data['vectors']]
        self.metadata = data['metadata']
        self.index_to_id = data['index_to_id']
        self.next_id = data['next_id']
        self.dimension = data['dimension']

class VectorStoreService:
    def __init__(self, store_path: str = "vector_store.json"):
        """
        Initialize vector store service
        
        Args:
            store_path: Path to save/load vector store
        """
        self.store_path = store_path
        self.vector_store = SimpleVectorStore()
        self.api_key = self._get_api_key()
        
        # Try to load existing vector store
        if os.path.exists(store_path):
            try:
                self.vector_store.load(store_path)
                print(f"Loaded existing vector store with {len(self.vector_store.vectors)} vectors")
            except Exception as e:
                print(f"Could not load existing vector store: {e}")
    
    def _get_api_key(self) -> Optional[str]:
        """Get OpenAI API key from config file"""
        try:
            with open('config/openai_api_key', 'r') as file:
                api_key = file.read().strip()
            if not api_key:
                raise ValueError("API key not found in config file")
            return api_key
        except Exception as e:
            print(f"Error loading API key: {str(e)}")
            return None
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for list of texts using OpenAI API
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not self.api_key:
            raise ValueError("OpenAI API key not available")
        
        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        all_embeddings = []
        batch_size = 100  # OpenAI API limit
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            data = {
                "input": batch_texts,
                "model": "text-embedding-ada-002"
            }
            
            try:
                response = requests.post(url, headers=headers, json=data)
                response.raise_for_status()
                
                result = response.json()
                embeddings = [item["embedding"] for item in result["data"]]
                all_embeddings.extend(embeddings)
                
                print(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                
            except Exception as e:
                print(f"Error generating embeddings for batch {i//batch_size + 1}: {e}")
                # Add zero vectors for failed batch
                all_embeddings.extend([[0.0] * 1536 for _ in batch_texts])
        
        return all_embeddings
    
    def add_documents(self, documents: List[Dict[str, any]]):
        """
        Add documents to vector store
        
        Args:
            documents: List of document chunks with text and metadata
        """
        if not documents:
            return
        
        print(f"Adding {len(documents)} documents to vector store...")
        
        # Extract texts for embedding
        texts = [doc['text'] for doc in documents]
        
        # Generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Convert to numpy arrays
        embedding_vectors = np.array(embeddings)
        
        # Add to vector store
        self.vector_store.add_vectors(embedding_vectors, documents)
        
        print(f"Added {len(documents)} documents to vector store")
        
        # Save updated store
        self.save_store()
    
    def search_similar_documents(self, query: str, k: int = 5, min_score: float = 0.7) -> List[Dict[str, any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query text
            k: Number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of similar documents with scores
        """
        if not self.vector_store.vectors:
            return []
        
        # Generate embedding for query
        query_embeddings = self.generate_embeddings([query])
        if not query_embeddings:
            return []
        
        query_vector = np.array(query_embeddings[0])
        
        # Search similar documents
        results = self.vector_store.search(query_vector, k)
        
        # Filter by minimum score and format results
        filtered_results = []
        for metadata, score in results:
            if score >= min_score:
                result = metadata.copy()
                result['similarity_score'] = score
                filtered_results.append(result)
        
        return filtered_results
    
    def get_relevant_context(self, query: str, max_context_length: int = 3000) -> str:
        """
        Get relevant context for query
        
        Args:
            query: Search query
            max_context_length: Maximum length of combined context
            
        Returns:
            Combined relevant text chunks
        """
        similar_docs = self.search_similar_documents(query, k=10, min_score=0.6)
        
        if not similar_docs:
            return ""
        
        # Combine relevant chunks, respecting max length
        context_parts = []
        current_length = 0
        
        for doc in similar_docs:
            text = doc['text']
            source = doc.get('source_file', 'Unknown')
            
            # Add source information
            chunk_text = f"[Source: {source}] {text}"
            
            if current_length + len(chunk_text) <= max_context_length:
                context_parts.append(chunk_text)
                current_length += len(chunk_text)
            else:
                # Add partial text if it fits
                remaining_space = max_context_length - current_length
                if remaining_space > 100:  # Only add if meaningful space left
                    partial_text = chunk_text[:remaining_space-3] + "..."
                    context_parts.append(partial_text)
                break
        
        return "\n\n".join(context_parts)
    
    def save_store(self):
        """Save vector store to disk"""
        try:
            self.vector_store.save(self.store_path)
            print(f"Vector store saved to {self.store_path}")
        except Exception as e:
            print(f"Error saving vector store: {e}")
    
    def get_store_info(self) -> Dict[str, any]:
        """Get information about the vector store"""
        return {
            'total_vectors': len(self.vector_store.vectors),
            'dimension': self.vector_store.dimension,
            'store_path': self.store_path,
            'sources': list(set(meta.get('source_file', 'Unknown') 
                               for meta in self.vector_store.metadata))
        }
    
    def clear_store(self):
        """Clear all vectors from store"""
        self.vector_store = SimpleVectorStore()
        if os.path.exists(self.store_path):
            os.remove(self.store_path)
        print("Vector store cleared") 