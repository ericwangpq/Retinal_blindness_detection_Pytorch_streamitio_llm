"""
RAG Service - Retrieval Augmented Generation
Integrates document processing, vector search, and LLM generation
"""

import os
from typing import List, Dict, Optional
from .document_processor import DocumentProcessor
from .vector_store import VectorStoreService
import requests
import json

class RAGService:
    def __init__(self, 
                 documents_dir: str = "rag_resources",
                 vector_store_path: str = "medical_vector_store.json",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize RAG service
        
        Args:
            documents_dir: Directory containing medical documents
            vector_store_path: Path to vector store file
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.documents_dir = documents_dir
        self.doc_processor = DocumentProcessor(chunk_size, chunk_overlap)
        self.vector_store = VectorStoreService(vector_store_path)
        self.api_key = self._get_api_key()
        
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
    
    def initialize_knowledge_base(self, force_rebuild: bool = False) -> bool:
        """
        Initialize the medical knowledge base from documents
        
        Args:
            force_rebuild: Whether to rebuild even if vector store exists
            
        Returns:
            True if successful, False otherwise
        """
        # Check if vector store already exists and has content
        store_info = self.vector_store.get_store_info()
        if store_info['total_vectors'] > 0 and not force_rebuild:
            print(f"Knowledge base already initialized with {store_info['total_vectors']} vectors")
            print(f"Sources: {store_info['sources']}")
            return True
        
        print("Initializing medical knowledge base...")
        
        # Check if documents directory exists
        if not os.path.exists(self.documents_dir):
            print(f"Documents directory {self.documents_dir} not found!")
            return False
        
        try:
            # Process all PDF documents
            print(f"Processing documents from {self.documents_dir}...")
            all_chunks = self.doc_processor.process_documents_directory(self.documents_dir)
            
            if not all_chunks:
                print("No documents were processed successfully!")
                return False
            
            # Filter for medical content
            print("Filtering for medical content...")
            medical_chunks = self.doc_processor.filter_medical_content(all_chunks)
            
            print(f"Found {len(medical_chunks)} relevant medical text chunks")
            
            if not medical_chunks:
                print("No relevant medical content found!")
                return False
            
            # Add to vector store
            print("Adding documents to vector store...")
            self.vector_store.add_documents(medical_chunks)
            
            # Print summary
            store_info = self.vector_store.get_store_info()
            print(f"\n✅ Knowledge base initialized successfully!")
            print(f"Total vectors: {store_info['total_vectors']}")
            print(f"Sources: {store_info['sources']}")
            
            return True
            
        except Exception as e:
            print(f"Error initializing knowledge base: {e}")
            return False
    
    def get_relevant_medical_context(self, query: str, max_context_length: int = 3000) -> str:
        """
        Retrieve relevant medical context for a query
        
        Args:
            query: User question or analysis context
            max_context_length: Maximum length of context
            
        Returns:
            Relevant medical literature context
        """
        try:
            context = self.vector_store.get_relevant_context(query, max_context_length)
            return context
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return ""
    
    def enhanced_medical_analysis(self, 
                                original_prompt: str,
                                patient_context: str = "",
                                prediction_results: Dict = None) -> str:
        """
        Provide enhanced medical analysis using RAG
        
        Args:
            original_prompt: Original analysis prompt
            patient_context: Patient information context
            prediction_results: AI prediction results
            
        Returns:
            Enhanced analysis with medical literature support
        """
        if not self.api_key:
            return "Error: OpenAI API key not available"
        
        # Create search query for relevant literature
        search_query = f"diabetic retinopathy {original_prompt}"
        if prediction_results:
            search_query += f" grade {prediction_results.get('value', '')} {prediction_results.get('class', '')}"
        
        # Get relevant medical context
        medical_context = self.get_relevant_medical_context(search_query, max_context_length=2500)
        
        # Construct enhanced prompt
        enhanced_prompt = f"""You are an expert ophthalmologist providing analysis based on current medical literature and best practices.

{patient_context}

RELEVANT MEDICAL LITERATURE:
{medical_context}

ORIGINAL ANALYSIS REQUEST:
{original_prompt}

Please provide a comprehensive analysis that:
1. Incorporates insights from the relevant medical literature above
2. Follows evidence-based best practices
3. Cites specific findings from the literature when relevant
4. Provides both technical medical assessment and patient-friendly explanation
5. Includes appropriate recommendations based on current guidelines

**IMPORTANT FORMATTING INSTRUCTIONS:**
- When referencing literature or research findings, use the format: ***According to the literature, [finding]*** or ***Research indicates that [finding]*** or ***Studies show that [finding]***
- Make all literature citations bold and italic using ***text*** format
- This will help patients easily identify evidence-based information
- Example: ***According to recent studies, patients with moderate diabetic retinopathy have a 25% risk of progression within one year***

When referencing the literature, mention the source papers to add credibility to your analysis."""

        return self._call_gpt_with_enhanced_prompt(enhanced_prompt)
    
    def enhanced_question_answering(self, 
                                  question: str,
                                  previous_analysis: str = "",
                                  patient_context: str = "",
                                  prediction_results: Dict = None) -> str:
        """
        Answer questions using RAG-enhanced responses
        
        Args:
            question: User question
            previous_analysis: Previous analysis context
            patient_context: Patient information
            prediction_results: AI prediction results
            
        Returns:
            Enhanced answer with literature support
        """
        if not self.api_key:
            return "Error: OpenAI API key not available"
        
        # Create search query for relevant literature
        search_query = f"diabetic retinopathy {question}"
        if prediction_results:
            search_query += f" {prediction_results.get('class', '')}"
        
        # Get relevant medical context
        medical_context = self.get_relevant_medical_context(search_query, max_context_length=2000)
        
        # Construct enhanced prompt
        enhanced_prompt = f"""You are an expert ophthalmologist answering patient questions based on current medical literature and best practices.

PATIENT CONTEXT:
{patient_context}

PREVIOUS ANALYSIS:
{previous_analysis}

RELEVANT MEDICAL LITERATURE:
{medical_context}

PATIENT QUESTION: {question}

Please provide a comprehensive answer that:
1. Directly addresses the patient's question
2. Incorporates relevant insights from the medical literature
3. Uses evidence-based information
4. Is accessible and empathetic for patient understanding
5. Cites specific research findings when helpful
6. Provides actionable recommendations when appropriate

**IMPORTANT FORMATTING INSTRUCTIONS:**
- When referencing literature or research findings, use the format: ***According to the literature, [finding]*** or ***Research shows that [finding]*** or ***Studies indicate that [finding]***
- Make all literature citations bold and italic using ***text*** format
- This helps patients easily identify evidence-based information
- Example: ***According to clinical guidelines, early detection can reduce vision loss by up to 95%***

Balance medical accuracy with patient-friendly communication."""

        return self._call_gpt_with_enhanced_prompt(enhanced_prompt)
    
    def _call_gpt_with_enhanced_prompt(self, prompt: str) -> str:
        """
        Call GPT API with enhanced prompt
        
        Args:
            prompt: Enhanced prompt with medical literature context
            
        Returns:
            GPT response
        """
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert ophthalmologist specializing in diabetic retinopathy. You have access to current medical literature and provide evidence-based, accurate medical insights while being accessible to patients. Always use ***bold italic*** formatting when citing literature or research findings to make evidence-based information clearly visible to patients."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 1500
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            return f"Error generating enhanced response: {str(e)}"
    
    def get_knowledge_base_stats(self) -> Dict[str, any]:
        """Get statistics about the knowledge base"""
        return self.vector_store.get_store_info()
    
    def search_medical_literature(self, query: str, k: int = 5) -> List[Dict[str, any]]:
        """
        Search medical literature directly
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant documents with similarity scores
        """
        return self.vector_store.search_similar_documents(query, k)
    
    def rebuild_knowledge_base(self) -> bool:
        """Rebuild the knowledge base from scratch"""
        print("Rebuilding knowledge base...")
        self.vector_store.clear_store()
        return self.initialize_knowledge_base(force_rebuild=True) 