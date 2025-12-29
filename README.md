# document-segmenting-engine

An intelligent document processing pipeline that performs AI-driven semantic segmentation of PDF documents and generates vector embeddings for knowledge base applications.

## Overview

This project implements a sophisticated document chunking system that mimics AWS Knowledge Base's semantic chunking approach, using OpenAI's language models to identify natural semantic boundaries in documents, followed by AWS Bedrock Titan embedding generation and storage in AWS S3 Vectors for efficient similarity search and retrieval-augmented generation (RAG) applications.

## Architecture

The system consists of multiple components with support for both OpenAI and Ollama backends:

### 1. **OpenAIConfig** - Configuration Management
- Manages OpenAI API credentials and model settings
- Loads configuration from environment variables using `python-dotenv`
- Required environment variables:
  - `OPENAI_API_KEY`: OpenAI API authentication key
  - `MODEL_ID`: OpenAI model identifier for semantic analysis

### 2. **OllamaAIConfig** - Ollama Configuration Management
- Manages Ollama model settings
- Loads configuration from environment variables
- Required environment variables:
  - `OLLAMA_MODEL_ID`: Ollama model identifier (e.g., "llama3.2", "phi4")
  - `OLLAMA_BASE_URL` (optional): Custom Ollama server URL

### 3. **DocumentSegmenter** - Semantic Chunking Engine (OpenAI)
- **Core Functionality**: Identifies semantic chunk boundaries in document text
- **Technology**: Uses OpenAI's structured output API with custom system prompts
- **Algorithm**:
  - Analyzes documents sentence by sentence
  - Detects topic shifts and semantic transitions
  - Maintains semantic continuity within chunks
  - Targets chunk sizes of ~200-450 tokens
  - Respects structural boundaries (code blocks, tables)
- **Output**: Returns `Breakpoints` (Pydantic model) containing:
  - List of sentence indices where new chunks should begin
  - Explanatory notes for each breakpoint decision

### 4. **OllamaDocumentSegmenter** - Semantic Chunking Engine (Ollama)
- **Core Functionality**: Provides the same semantic chunking as DocumentSegmenter but using Ollama
- **Technology**: Extends `AICore` base class with Ollama chat API
- **Advantages**:
  - Works with local Ollama models
  - No API costs
  - Privacy-preserving (runs locally)
  - Supports various open-source models
- **Output**: Returns `Breakpoints` with JSON parsing from chat responses

### 5. **PdfChunker** - PDF Processing Pipeline (OpenAI)
- **Workflow**:
  1. Extracts text from PDF files using `pypdf.PdfReader`
  2. Splits extracted text into sentences using NLTK's sentence tokenizer
  3. Delegates to `DocumentSegmenter` to determine semantic breakpoints
  4. Applies breakpoints to create final text chunks
- **Features**:
  - Automatic NLTK punkt tokenizer download if missing
  - Robust error handling for file operations
  - Breakpoint validation and sanitization
  - Empty chunk prevention

### 6. **OllamaPdfChunker** - PDF Processing Pipeline (Ollama)
- **Same Functionality**: Identical to PdfChunker but uses `OllamaDocumentSegmenter`
- **Use Case**: For users who prefer local Ollama models over OpenAI
- **Implementation**: Uses the same extraction and tokenization logic

### 7. **TitanEmbeddingGenerator** - Vector Embedding & Storage
- **Purpose**: Generates vector embeddings and stores them in AWS S3 Vectors
- **Technology Stack**:
  - AWS Bedrock Runtime: Invokes Titan Embed Text V2 model
  - AWS S3 Vectors: Stores vector embeddings with metadata
- **Configuration** (currently empty strings, require setup):
  - Region: AWS region for services
  - Bucket: S3 Vectors bucket name
  - Vector Index: Name of the vector index
  - Model ID: Bedrock embedding model identifier
- **Process**:
  1. Generates embeddings for each text chunk via Bedrock
  2. Packages vectors with rich metadata:
     - Unique chunk key
     - Text preview (first 500 characters)
     - Character length and word count
  3. Stores vectors in S3 Vectors for similarity search

### 8. **Streamlit Web Application** (`app.py`)
- **User Interface**: Interactive web app for document processing
- **Features**:
  - PDF file upload
  - Two processing modes:
    - **Segment Only**: Extract and display text chunks
    - **Segment & Vectorize**: Full pipeline with embedding storage
  - Real-time progress indicators
  - Expandable chunk previews with word counts
  - Processing metrics and summaries
- **Session Management**: Stores chunks for potential reprocessing

## Key Features

### Semantic Boundary Detection
The system intelligently identifies where new chunks should begin based on:
- **Topic Changes**: Detects shifts in subject matter
- **Discourse Markers**: Recognizes transition phrases
- **Semantic Continuity**: Groups related sentences together
- **Size Optimization**: Balances semantic cohesion with target chunk sizes

### Structured Output Parsing
Uses Pydantic models for type-safe data handling:
- `Breakpoints` model ensures consistent API responses
- Validates breakpoint indices and explanatory notes
- Enables robust error handling

### Metadata Enrichment
Each vector embedding includes comprehensive metadata:
- Chunk identification
- Text preview for human verification
- Character and word count statistics
- Facilitates debugging and quality control

## Dependencies

```
boto3                              # AWS SDK for Python
openai                             # OpenAI API client
python-dotenv                      # Environment variable management
pypdf                              # PDF text extraction
pydantic                          # Data validation
nltk                              # Natural language processing
streamlit                         # Web application framework
boto3-stubs[bedrock-runtime,s3vectors]  # Type hints for boto3
ollama                            # Ollama Python client for local LLMs
```

## Configuration Requirements

### Environment Variables
Create a `.env` file with:

**For OpenAI Backend:**
```
OPENAI_API_KEY=your_openai_api_key
MODEL_ID=your_openai_model_id
```

**For Ollama Backend:**
```
OLLAMA_MODEL_ID=llama3.2
OLLAMA_BASE_URL=http://localhost:11434  # Optional, defaults to standard Ollama URL
```

### Choosing Between OpenAI and Ollama

**Use OpenAI (DocumentSegmenter/PdfChunker) when:**
- You need the highest quality semantic analysis
- API costs are acceptable
- You prefer managed cloud services

**Use Ollama (OllamaDocumentSegmenter/OllamaPdfChunker) when:**
- You want to run models locally
- You need privacy/data security
- You want to avoid API costs
- You have sufficient local compute resources

### Usage Example

**With OpenAI:**
```python
from document_segmenter import PdfChunker

chunker = PdfChunker()
chunks = chunker.chunk_pdf("document.pdf")
```

**With Ollama:**
```python
from document_segmenter import OllamaPdfChunker

chunker = OllamaPdfChunker()
chunks = chunker.chunk_pdf("document.pdf")
```

### AWS Configuration
- Configure AWS credentials (via AWS CLI, environment variables, or IAM roles)
- Update `TitanEmbeddingGenerator` class variables:
  - `__REGION_NAME`: Your AWS region
  - `__BUCKET_NAME`: Your S3 Vectors bucket
  - `__VECTOR_INDEX_NAME`: Your vector index name
  - `__EMBEDDING_MODEL_ID`: Bedrock model ID (e.g., "amazon.titan-embed-text-v2:0")

## Use Cases

- **Knowledge Base Construction**: Build searchable document repositories
- **RAG Applications**: Prepare documents for retrieval-augmented generation
- **Semantic Search**: Enable meaning-based document retrieval
- **Document Analysis**: Study document structure and semantic flow
- **Content Organization**: Automatically segment long-form content

## Technical Highlights

- **Type Safety**: Extensive type hints throughout the codebase
- **Error Handling**: Comprehensive exception handling with informative messages
- **Modular Design**: Clear separation of concerns with distinct classes
- **Property Encapsulation**: Uses Python properties for controlled attribute access
- **Documentation**: Detailed docstrings for all public methods and classes
- **Cloud Integration**: Seamless AWS service integration via boto3
