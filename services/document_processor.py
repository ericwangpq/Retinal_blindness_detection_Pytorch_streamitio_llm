"""
Document processing service for RAG system
Handles PDF extraction, text chunking, and preprocessing
"""

import os
import fitz  # PyMuPDF
import re
from typing import List, Dict, Tuple
from pathlib import Path

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Overlap between chunks for context preservation
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += page.get_text()
            
            doc.close()
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess extracted text
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep medical terminology
        text = re.sub(r'[^\w\s\-\.\,\:\;\(\)\%\+\=\<\>]', '', text)
        
        # Remove page numbers and headers/footers patterns
        text = re.sub(r'\b\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove excessive newlines
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()
    
    def chunk_text(self, text: str) -> List[Dict[str, str]]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks with metadata
        """
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append({
                    'text': current_chunk.strip(),
                    'length': current_length
                })
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + " " + sentence
                current_length = len(current_chunk)
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_length += sentence_length
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'length': current_length
            })
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """
        Get overlap text from the end of current chunk
        
        Args:
            text: Current chunk text
            
        Returns:
            Overlap text for next chunk
        """
        if len(text) <= self.chunk_overlap:
            return text
        
        # Get last chunk_overlap characters, but try to break at sentence boundary
        overlap_text = text[-self.chunk_overlap:]
        
        # Find the last sentence boundary in overlap
        last_sentence = overlap_text.rfind('.')
        if last_sentence > 0:
            overlap_text = overlap_text[last_sentence + 1:].strip()
        
        return overlap_text
    
    def process_pdf_file(self, pdf_path: str) -> List[Dict[str, any]]:
        """
        Process a single PDF file into chunks with metadata
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of processed chunks with metadata
        """
        # Extract and clean text
        raw_text = self.extract_text_from_pdf(pdf_path)
        if not raw_text:
            return []
        
        cleaned_text = self.clean_text(raw_text)
        
        # Create chunks
        chunks = self.chunk_text(cleaned_text)
        
        # Add metadata to each chunk
        filename = Path(pdf_path).stem
        processed_chunks = []
        
        for i, chunk in enumerate(chunks):
            processed_chunks.append({
                'text': chunk['text'],
                'source_file': filename,
                'chunk_id': i,
                'file_path': pdf_path,
                'length': chunk['length']
            })
        
        return processed_chunks
    
    def process_documents_directory(self, directory_path: str) -> List[Dict[str, any]]:
        """
        Process all PDF files in a directory
        
        Args:
            directory_path: Path to directory containing PDFs
            
        Returns:
            List of all processed chunks from all PDFs
        """
        all_chunks = []
        pdf_files = list(Path(directory_path).glob("*.pdf"))
        
        print(f"Processing {len(pdf_files)} PDF files...")
        
        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file.name}")
            chunks = self.process_pdf_file(str(pdf_file))
            all_chunks.extend(chunks)
            print(f"  - Generated {len(chunks)} chunks")
        
        print(f"Total chunks generated: {len(all_chunks)}")
        return all_chunks
    
    def filter_medical_content(self, chunks: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """
        Filter chunks to keep only medically relevant content
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Filtered chunks containing medical content
        """
        medical_keywords = [
            'retinal', 'retinopathy', 'diabetic', 'diabetes', 'ophthalmology',
            'fundus', 'macula', 'vitreous', 'hemorrhage', 'exudate', 'neovascularization',
            'microaneurysm', 'proliferative', 'non-proliferative', 'ischemia',
            'edema', 'vision', 'visual', 'blindness', 'screening', 'treatment'
        ]
        
        filtered_chunks = []
        
        for chunk in chunks:
            text_lower = chunk['text'].lower()
            
            # Check if chunk contains medical keywords
            if any(keyword in text_lower for keyword in medical_keywords):
                chunk['relevance_score'] = sum(1 for keyword in medical_keywords if keyword in text_lower)
                filtered_chunks.append(chunk)
        
        # Sort by relevance score (descending)
        filtered_chunks.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return filtered_chunks 