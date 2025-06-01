#!/usr/bin/env python3
"""
RAG System Initialization Script
This script helps initialize and manage the medical literature knowledge base
"""

import os
import sys
import argparse
from pathlib import Path

# Add services directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'services'))

try:
    from services.rag_service import RAGService
    from services.document_processor import DocumentProcessor
    from services.vector_store import VectorStoreService
except ImportError as e:
    print(f"Error importing services: {e}")
    print("Please ensure all required packages are installed.")
    sys.exit(1)

def check_dependencies():
    """Check if required dependencies are available"""
    required_packages = ['numpy', 'requests', 'pathlib']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install " + ' '.join(missing_packages))
        return False
    
    return True

def check_api_key():
    """Check if OpenAI API key is configured"""
    config_path = Path("config/openai_api_key")
    
    if not config_path.exists():
        print("❌ OpenAI API key not found!")
        print("Please create the file 'config/openai_api_key' and add your OpenAI API key.")
        print("You can get an API key from: https://platform.openai.com/api-keys")
        return False
    
    try:
        with open(config_path, 'r') as f:
            api_key = f.read().strip()
        
        if not api_key:
            print("❌ OpenAI API key file is empty!")
            return False
        
        print("✅ OpenAI API key found")
        return True
        
    except Exception as e:
        print(f"❌ Error reading API key: {e}")
        return False

def check_documents():
    """Check if medical documents are available"""
    docs_path = Path("rag_resources")
    
    if not docs_path.exists():
        print("❌ Documents directory 'rag_resources' not found!")
        return False
    
    pdf_files = list(docs_path.glob("*.pdf"))
    
    if not pdf_files:
        print("❌ No PDF files found in 'rag_resources' directory!")
        return False
    
    print(f"✅ Found {len(pdf_files)} PDF documents")
    return True

def initialize_rag(force_rebuild=False):
    """Initialize the RAG system"""
    print("🚀 Initializing RAG System for Medical Literature...")
    print("=" * 50)
    
    # Check dependencies
    print("1. Checking dependencies...")
    if not check_dependencies():
        return False
    
    # Check API key
    print("\n2. Checking OpenAI API key...")
    if not check_api_key():
        return False
    
    # Check documents
    print("\n3. Checking medical documents...")
    if not check_documents():
        return False
    
    # Initialize RAG service
    print("\n4. Initializing RAG service...")
    try:
        rag_service = RAGService()
        
        print("\n5. Processing medical documents and building knowledge base...")
        print("⚠️  Note: This may take several minutes and will use OpenAI API credits.")
        
        user_input = input("Do you want to continue? (y/N): ")
        if user_input.lower() not in ['y', 'yes']:
            print("❌ Initialization cancelled by user.")
            return False
        
        success = rag_service.initialize_knowledge_base(force_rebuild=force_rebuild)
        
        if success:
            print("\n🎉 RAG System initialized successfully!")
            
            # Display statistics
            stats = rag_service.get_knowledge_base_stats()
            print(f"\n📊 Knowledge Base Statistics:")
            print(f"   • Total vectors: {stats.get('total_vectors', 0)}")
            print(f"   • Sources: {len(stats.get('sources', []))}")
            print(f"   • Vector store: {stats.get('store_path', 'Unknown')}")
            
            return True
        else:
            print("❌ Failed to initialize RAG system.")
            return False
            
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        return False

def show_status():
    """Show current RAG system status"""
    print("📊 RAG System Status")
    print("=" * 30)
    
    try:
        rag_service = RAGService()
        stats = rag_service.get_knowledge_base_stats()
        
        if stats.get('total_vectors', 0) > 0:
            print("✅ RAG system is initialized and ready")
            print(f"   • Total vectors: {stats.get('total_vectors', 0)}")
            print(f"   • Sources: {len(stats.get('sources', []))}")
            print(f"   • Vector store: {stats.get('store_path', 'Unknown')}")
            
            sources = stats.get('sources', [])
            if sources:
                print(f"\n📚 Document Sources ({len(sources)}):")
                for i, source in enumerate(sources[:10], 1):
                    print(f"   {i}. {source}")
                if len(sources) > 10:
                    print(f"   ... and {len(sources) - 10} more")
        else:
            print("⚠️  RAG system not initialized")
            print("   Use 'python initialize_rag.py --init' to initialize")
            
    except Exception as e:
        print(f"❌ Error checking status: {e}")

def search_literature(query, k=5):
    """Search medical literature"""
    print(f"🔍 Searching for: '{query}'")
    print("=" * 50)
    
    try:
        rag_service = RAGService()
        results = rag_service.search_medical_literature(query, k)
        
        if not results:
            print("No results found.")
            return
        
        for i, result in enumerate(results, 1):
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                continue
                
            print(f"\n📄 Result {i}:")
            print(f"   Source: {result.get('source_file', 'Unknown')}")
            print(f"   Similarity: {result.get('similarity_score', 0):.2%}")
            print(f"   Content: {result.get('text', '')[:200]}...")
            
    except Exception as e:
        print(f"❌ Error searching literature: {e}")

def main():
    parser = argparse.ArgumentParser(description="RAG System Management Tool")
    parser.add_argument('--init', action='store_true', help='Initialize RAG system')
    parser.add_argument('--force', action='store_true', help='Force rebuild of knowledge base')
    parser.add_argument('--status', action='store_true', help='Show RAG system status')
    parser.add_argument('--search', type=str, help='Search medical literature')
    parser.add_argument('--results', type=int, default=5, help='Number of search results (default: 5)')
    
    args = parser.parse_args()
    
    if args.init:
        initialize_rag(force_rebuild=args.force)
    elif args.status:
        show_status()
    elif args.search:
        search_literature(args.search, args.results)
    else:
        print("RAG System Management Tool")
        print("=" * 30)
        print("Usage:")
        print("  python initialize_rag.py --init      # Initialize RAG system")
        print("  python initialize_rag.py --status    # Show system status")
        print("  python initialize_rag.py --search 'query'  # Search literature")
        print("  python initialize_rag.py --force --init    # Force rebuild")
        print("\nFor help: python initialize_rag.py --help")

if __name__ == "__main__":
    main() 