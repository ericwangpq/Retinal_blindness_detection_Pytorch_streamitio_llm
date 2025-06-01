# RAG-Enhanced Diabetic Retinopathy Analysis System

This system has been enhanced with **Retrieval Augmented Generation (RAG)** capabilities that integrate medical literature to provide evidence-based analysis and recommendations for diabetic retinopathy screening.

## 🆕 New RAG Features

### 📚 Medical Literature Integration
- **Vector Database**: Indexes medical literature from `rag_resources/` directory
- **Semantic Search**: Finds relevant research papers for each analysis
- **Evidence-Based Responses**: AI answers backed by current medical literature
- **Literature Search**: Direct search through medical papers

### 🔬 Enhanced Analysis
- **Literature-Supported Diagnoses**: AI analysis includes relevant research citations
- **Evidence-Based Recommendations**: Treatment suggestions based on current guidelines
- **Research-Backed Explanations**: Patient-friendly explanations supported by medical evidence

## 🚀 Getting Started with RAG

### Prerequisites

1. **Install Additional Dependencies** (if not already installed):
   ```bash
   pip install numpy requests pathlib
   pip install PyMuPDF  # For PDF processing
   ```

2. **Set up OpenAI API Key**:
   ```bash
   # Create config directory (if not exists)
   mkdir -p config
   
   # Add your OpenAI API key
   echo "your-openai-api-key-here" > config/openai_api_key
   ```
   
   Get your API key from: https://platform.openai.com/api-keys

3. **Verify Medical Documents**:
   - Ensure PDF files are in the `rag_resources/` directory
   - The system will automatically process these during initialization

### Initialize RAG System

Use the initialization script to set up the medical knowledge base:

```bash
# Initialize RAG system
python initialize_rag.py --init

# Check system status
python initialize_rag.py --status

# Force rebuild (if needed)
python initialize_rag.py --init --force
```

**Note**: Initial setup may take several minutes and will use OpenAI API credits to generate embeddings.

## 📖 How RAG Works

### 1. Document Processing
- **PDF Extraction**: Extracts text from medical literature PDFs
- **Text Chunking**: Splits documents into manageable, overlapping chunks
- **Medical Filtering**: Identifies and prioritizes medically relevant content

### 2. Vector Storage
- **Embeddings Generation**: Creates vector representations using OpenAI embeddings
- **Similarity Search**: Enables semantic search through medical literature
- **Persistent Storage**: Saves processed vectors for future use

### 3. Enhanced Analysis
- **Context Retrieval**: Finds relevant literature for each analysis request
- **Prompt Enhancement**: Augments AI prompts with medical research context
- **Evidence Integration**: Combines AI capabilities with authoritative medical sources

## 🔧 RAG System Management

### Command Line Tools

```bash
# Show help
python initialize_rag.py --help

# Initialize system
python initialize_rag.py --init

# Check status
python initialize_rag.py --status

# Search literature
python initialize_rag.py --search "diabetic retinopathy treatment"

# Force rebuild
python initialize_rag.py --init --force
```

### Web Interface Features

1. **Knowledge Base Status**: View indexed documents in sidebar
2. **Literature Search Tab**: Search medical papers directly
3. **Enhanced Analysis**: Get evidence-based AI explanations
4. **Research-Backed Q&A**: Ask questions with literature support

## 📁 File Structure

```
├── services/
│   ├── document_processor.py    # PDF processing and text chunking
│   ├── vector_store.py         # Vector storage and similarity search
│   ├── rag_service.py          # Main RAG orchestration
│   └── retinal_vqa.py          # Enhanced VQA with RAG integration
├── rag_resources/              # Medical literature PDFs
├── config/
│   ├── README.md               # Configuration instructions
│   └── openai_api_key          # Your OpenAI API key (create this)
├── initialize_rag.py           # RAG system management script
├── medical_vector_store.json   # Vector database (auto-generated)
└── app.py                      # Enhanced Streamlit application
```

## 🔍 Usage Examples

### 1. Basic Analysis with RAG
1. Upload retinal image
2. Click "Analyze Image"
3. Get enhanced analysis with literature citations
4. Review evidence-based recommendations

### 2. Literature Search
1. Go to "Literature Search" tab
2. Enter search query (e.g., "proliferative diabetic retinopathy")
3. Review relevant research papers
4. Click on results for detailed content

### 3. Evidence-Based Q&A
1. After analysis, ask questions in Q&A section
2. Get answers supported by medical literature
3. Receive citations and research-backed recommendations

## ⚙️ Configuration Options

### RAG Service Parameters
- **Chunk Size**: Default 1000 characters (adjustable in code)
- **Chunk Overlap**: Default 200 characters for context preservation
- **Similarity Threshold**: Default 0.6 for relevant document retrieval
- **Max Context Length**: Default 3000 characters for AI prompts

### Vector Store Settings
- **Embedding Model**: `text-embedding-ada-002` (OpenAI)
- **Vector Dimension**: 1536 (OpenAI standard)
- **Storage Format**: JSON for portability
- **Similarity Metric**: Cosine similarity

## 🚨 Troubleshooting

### Common Issues

1. **"Knowledge base not available"**
   - Run: `python initialize_rag.py --init`
   - Check OpenAI API key configuration

2. **"No relevant results found"**
   - Verify documents in `rag_resources/` directory
   - Try broader search terms
   - Check if knowledge base is initialized

3. **API key errors**
   - Verify `config/openai_api_key` file exists
   - Ensure API key is valid and has credits
   - Check file permissions

4. **PDF processing errors**
   - Ensure PDFs are not corrupted
   - Check file permissions in `rag_resources/`
   - Try rebuilding with `--force` flag

### Performance Tips

1. **First-time setup**: Allow 5-10 minutes for full initialization
2. **API usage**: Monitor OpenAI usage dashboard for costs
3. **Storage**: Vector store file may be large (several MB)
4. **Memory**: Large document collections may require more RAM

## 📊 System Monitoring

### Check RAG Status
```bash
python initialize_rag.py --status
```

### View Available Sources
The web interface sidebar shows:
- Number of indexed documents
- Source paper names
- Knowledge base status

### Search Performance
- Similarity scores in search results
- Response time in web interface
- API usage in OpenAI dashboard

## 🔒 Security Considerations

1. **API Key Protection**:
   - Never commit `config/openai_api_key` to version control
   - Use environment variables in production
   - Regularly rotate API keys

2. **Medical Data**:
   - Ensure compliance with healthcare regulations
   - Consider data encryption for sensitive documents
   - Implement proper access controls

3. **Literature Sources**:
   - Verify copyright compliance for medical papers
   - Use only authorized/licensed documents
   - Cite sources appropriately in outputs

## 💡 Tips for Best Results

1. **Document Quality**: Use high-quality, recent medical literature
2. **Search Queries**: Use specific medical terminology for better results
3. **Analysis Context**: Provide detailed patient information when available
4. **Question Formulation**: Ask specific, medically-focused questions

## 🔄 Updates and Maintenance

### Adding New Literature
1. Add PDF files to `rag_resources/` directory
2. Run: `python initialize_rag.py --init --force`
3. Verify new sources in web interface

### Updating System
1. Backup vector store file
2. Update code files
3. Rebuild knowledge base if needed
4. Test functionality with sample queries

---

For technical support or questions about the RAG enhancement, please refer to the service files documentation or create an issue in the project repository. 