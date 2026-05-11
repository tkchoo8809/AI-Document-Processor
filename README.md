# Document Intelligence Visual RAG

An open-source prototype solution for extracting structured data from scanned documents using fine-tuned vision language models and Retrieval-Augmented Generation (RAG).

## 🧩 Problem Statement

Scanned documents often contain invoices, receipts, and forms that are difficult to parse with traditional OCR alone. This project addresses the challenges of noisy scans, varied layouts, and incomplete text by combining vision-language models with retrieval and structured data extraction.

The goal is to build an open-source, cost-free pipeline that can be deployed without proprietary APIs, enabling document processing for research and production use while minimizing external dependency costs.

## 📋 Project Overview

This project delivers an end-to-end prototype for document intelligence, combining vision-language modeling, retrieval, and structured agent evaluation.

- **Document Processing**: Extract structured text, invoice fields, and tables from images and PDFs using vision-language models.
- **Model Fine-tuning**: Train and adapt vision models with Unsloth and Hugging Face tooling for domain-specific document understanding.
- **Retrieval-Augmented Generation**: Use Qdrant to store and search document embeddings for fast retrieval and contextual reasoning.
- **Data Augmentation**: Create realistic augmented document data to improve model robustness across scan quality and layout variation.
- **Inference & Evaluation**: Compare models, validate extraction accuracy, and assess agent decision-making with LLM-based evaluation.

## 🚀 Features

- **Multi-Model Support**:
  - Vision-Language: `unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit`, `unsloth/Qwen2-VL-2B-Instruct-bnb-4bit`, `unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit`
  - Retrieval: `vidore/colqwen2-v1.0-hf`, `vidore/colpali-v1.3-hf`, `jinaai/jina-embeddings-v4`
  
- **Efficient Training**:
  - 4-bit quantization with BitsAndBytes
  - LoRA fine-tuning with Unsloth
  - CUDA acceleration support
  
- **Structured Data Extraction**:
  - Pydantic-based schema validation
  - JSON output with automatic field mapping
  - Custom validation rules for currency, dates, and other formats
  
- **RAG System**:
  - Qdrant cloud/local deployment support
  - Payload indexing for efficient retrieval
  - Multi-modal document indexing and retrieval
  
## 📁 Project Structure

```
├── visual_rag.py               # Core RAG system with Qdrant integration
├── model_training.py           # Fine-tuning test for vision models
├── model_inference.py          # Inference test for trained models
├── inference_comparison.py     # Model result testing and evaluation
├── augment_image.py            # Data augmentation utilities for testing
├── pydantic_classes.py         # Schema definitions for structured extraction
├── pydantic_llm_judge.py       # LLM-based evaluation schemas
├── pydantic_orchestrator.py    # Agent orchestration and tool calling
├── helper.py                   # Utility functions and dataset loaders
├── requirements.txt            # Python dependencies
├── credentials.env             # Environment variables (API keys, URLs)
└── README.md                   # This file
```

## 🔧 Installation

### Prerequisites
- Python 3.10+
- CUDA 12.1 compatible GPU (recommended for faster training/inference)
- 16GB+ VRAM for model inference (depending on model size)

### Setup

1. **Clone and navigate to project directory**:
```bash
cd /path/to/Capstone
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/Scripts/activate  # Windows
# or
source venv/bin/activate      # Linux/Mac
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

**Note**: For PyTorch with CUDA 12.8 support, the requirements.txt includes a pinned PyTorch version. If you need a different CUDA version, install PyTorch separately:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

4. **Configure credentials**:
Create a `credentials.env` file with your configuration:
```env
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
HF_TOKEN=your_huggingface_token
```

## 📚 Core Modules

### `visual_rag.py`
Main module for Retrieval-Augmented Generation with Qdrant vector database and vision language models.

**Key Classes**:

#### `Qdrant`
Manages vector database operations with Qdrant (cloud or local).

**Methods**:
- `__init__(collection_name, url, api_key)` - Initialize cloud connection to Qdrant
- `_setup_collection()` - Create collection with dual vector fields for efficient retrieval:
  - `page_summary`: 128D dense vectors for fast pre-filtering (Stage 1)
  - `colqwen_multivector`: Multi-vector format for precise re-ranking (Stage 2)
- `initialise_payload_indices()` - Create indices for metadata filtering (dates, invoice numbers, seller/buyer names)
- `upsert(ids, multivectors, summary_vectors, metadata)` - Upload documents with embeddings to Qdrant in batches

#### `VisualRAG`
Complete RAG pipeline combining document retrieval, extraction, and generation.

**Initialization**:
- Loads dual models: Vision-Language (Qwen2.5-VL) for extraction and ColQwen2 for document retrieval
- Initializes Qdrant cloud connection and local storage directory

**Core Methods**:

*Model Loading*:
- `initialise_unsloth_inference_model()` - Load 4-bit quantized Qwen2.5-VL for efficient inference
- `initialise_hf_retriever_model()` - Load ColQwen2 or ColPali for multi-vector document retrieval

*Memory Management*:
- `clear_vram(label, reset_stats)` - Free GPU memory, sync CUDA, and report memory usage
- `print_gpu_memory_usage(model)` - Display current VRAM allocation

*Document Embedding & Retrieval*:
- `embed(inputs, mode)` - Generate embeddings for images or text queries with automatic tensor cleanup
- `build_filters(date, seller, invoice_id, buyer)` - Create metadata filters for targeted search
- `search(query_text, k, filter_conditions)` - Hybrid search: semantic vector search + metadata filtering using two-stage re-ranking

*Data Extraction*:
- `extract_metadata(pil_image)` - Extract structured invoice data (header, items, totals) with Pydantic validation
- `evaluate_json(extracted, ground_truth)` - Compare extraction accuracy against ground truth with per-field scoring

*Document Indexing*:
- `upload_and_index(file_path)` - Main entry point for indexing images or PDFs
- `generate_deterministic_id(file_path)` - Create consistent UUIDs from file contents (handles deduplication)
- `_handle_image(path_obj)` - Extract metadata, embed, and upsert single image
- `_handle_pdf(path_obj)` - Process multi-page PDFs: extract, embed, and index each page separately

*Inference*:
- `inference(user_query, images, k, json_format)` - Run vision model on documents:
  - JSON mode: Extract structured data with schema enforcement
  - Q&A mode: Answer questions about document content

**Key Features**:
- Two-stage retrieval: Fast semantic pre-filtering + precise multi-vector re-ranking
- Metadata filtering with indexed fields for exact and partial matching
- PDF support with page-level indexing and header inheritance across pages
- Automatic deduplication using content-based deterministic IDs
- Memory-efficient inference with 4-bit quantization and VRAM cleanup
- Structured JSON extraction with Pydantic validation

**Usage Example**:
```python
from visual_rag import VisualRAG
import os

# Initialize
rag = VisualRAG(
    qdrant_url=os.getenv("QDRANT_URL"),
    qdrant_api_key=os.getenv("QDRANT_API_KEY")
)

# Upload and index documents
rag.upload_and_index("path/to/invoice.pdf")
rag.upload_and_index("path/to/image.jpg")

# Search with filters
results = rag.search(
    query_text="Invoice amount",
    k=3,
    filter_conditions=rag.build_filters(invoice_id="12345")
)

# Extract structured data
from PIL import Image
image = Image.open("invoice.jpg")
data = rag.extract_metadata(image)

# Answer questions about documents
answer = rag.inference(
    user_query="What is the total amount?",
    images=[image],
    json_format=False
)
``` 

### `model_training.py`
Fine-tuning vision models using the Unsloth framework for efficient training.

**Main Function**: `train_unsloth_qwen(dataset_location="augmented_data")`

**Features**:
- Loads JSONL-formatted training data
- 4-bit quantization for reduced memory usage
- LoRA adapters for parameter-efficient fine-tuning
- Supports multiple model architectures (Qwen2-VL, Llama-3.2-Vision, Pixtral)

**Usage**:
```python
from model_training import train_unsloth_qwen

train_unsloth_qwen(dataset_location="augmented_data")
```

**JSONL Training Data Format**
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "image_path.jpg"},
        {"type": "text", "text": "Extract structured data..."}
      ]
    },
    {
      "role": "assistant",
      "content": "{...extracted_json...}"
    }
  ]
}
```

**Extraction Output Format**
```json
{
  "field_1": "value1",
  "field_2": "value2",
  "monetary_field": 1000.00,
  "date_field": "01-01-2024"
}
```

### `model_inference.py`
Run inference using fine-tuned or pre-trained vision models.

**Main Function**: `unsloth_inference(image_path, model_path)`

**Features**:
- Schema-guided extraction using Pydantic models
- JSON output validation
- Support for local and HuggingFace models
- 4-bit inference for memory efficiency

**Usage**:
```python
from model_inference import unsloth_inference

result = unsloth_inference(
    image_path="path/to/image.jpg",
    model_path="unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit"
)
```

### `pydantic_classes.py`
Defines schemas for structured data extraction from documents.

**Key Classes**:
- `UniversalExtraction`: Base schema for document information extraction
- Custom validators for:
  - Currency formatting (removes symbols, converts to float)
  - Date standardization (multiple format support)
  - Field-level validation rules

**Usage**:
```python
from pydantic_classes import UniversalExtraction

# Automatically validates extracted data against schema
extraction = UniversalExtraction(**extracted_json)
```

### `augment_image.py`
Generate augmented training data from original documents.

**Features**:
- Image augmentation techniques (rotation, scaling, noise, etc.)
- Batch processing for large datasets
- Maintains data consistency during augmentation

### `helper.py`
Utility functions and dataset loaders.

**Key Classes**:
- `JSONLDataset`: PyTorch-compatible dataset loader for JSONL training data
- Helper functions for image processing, data validation, etc.

### `pydantic_orchestrator.py`
AI-powered agent orchestrator using Pydantic AI for conversational document management with local LLM backend (Ollama).

**Key Components**:

#### `RagDeps` (Dataclass)
Dependency container that holds the `VisualRAG` instance for use across agent tools.

#### Local LLM Configuration
- **Backend**: Ollama (local inference server)
- **Models Supported**: `qwen2.5:7b`, `qwen2:7b`, `llama3.1:8b` (running on CPU or GPU)
- **API**: OpenAI-compatible interface at `http://localhost:11434/v1`
- **Advantage**: No API costs, full privacy, runs locally

**Setup**:
```bash
# Download Ollama from https://ollama.com/download
ollama pull qwen2.5:7b     # or preferred model
```

#### Agent Tools

**1. `search_documents(user_query, date, seller, buyer, invoice_id)`**
- Search for documents using natural language queries
- Supports optional metadata filters (date, seller/buyer names, invoice ID)
- Returns matching documents from Qdrant database as JSON
- **Smart Feature**: Won't trigger unnecessary tool calls if user is reformatting previous responses

**2. `upload_document(file_path)`**
- Upload and index a single image or PDF file
- Automatically handles format detection and extraction
- Returns status with point ID or error message

**3. `batch_upload_documents(pattern)`**
- Upload multiple files matching a glob pattern or directory
- Examples: `data/*.pdf`, `./invoices/`, `processed_documents/*.jpg`
- Returns summary with success/failure counts
- Error handling with sample of failed uploads

#### Utility Functions

**`print_memory_usage(model_name, host_url)`**
- Monitor Ollama model's memory footprint
- Displays System RAM and GPU VRAM usage
- Useful for performance tuning

**`run_chatbot()`**
- Main event loop for the conversational agent
- Handles user input and maintains chat history
- Manages tool calls and responses
- Graceful exit with 'quit', 'exit', or 'q'
- Chat memory reset with 'clear' command

#### System Prompt Features
- Guides the AI on when to use each tool
- Prevents redundant tool calls when reformatting previous responses
- Encourages using chat history for memory-based operations

**Usage Example**:
```python
import asyncio
from pydantic_orchestrator import run_chatbot

# Start the interactive agent
asyncio.run(run_chatbot())

# In chat:
# User: "Upload all invoices from data/*.pdf"
# Agent: Uses batch_upload_documents tool
#
# User: "What's the total amount in invoice #12345?"
# Agent: Uses search_documents with invoice_id filter
#
# User: "Make that JSON format"
# Agent: Reformats previous response WITHOUT calling tools
```

**Key Features**:
- Conversational interface for document management
- Natural language understanding for complex queries
- Asynchronous tool execution for non-blocking operations
- Memory-aware prompt preventing tool call abuse
- Local-first architecture (no external API required beyond HuggingFace models)
- Integrates seamlessly with VisualRAG backend

## ⚙️ Configuration

### Environment Variables
Set these in `credentials.env`:

```env
# Qdrant Configuration
QDRANT_URL=https://your-cloud-instance.qdrant.io
QDRANT_API_KEY=your_api_key_here

# HuggingFace Configuration
HF_TOKEN=your_huggingface_token
HF_HUB_ENABLE_HF_TRANSFER=1
HF_HUB_DOWNLOAD_TIMEOUT=1000
```

### Model Selection
Models are specified in individual scripts. Supported models:

**Vision-Language Models**:
- `unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit`
- `unsloth/Qwen2-VL-2B-Instruct-bnb-4bit`
- `unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit`
- `unsloth/Pixtral-12B-2409-bnb-4bit`

**Retrieval Models**:
- `ColQwen2ForRetrieval`
- `ColPaliForRetrieval`


## 🔍 Key Libraries

- **Transformers**: HuggingFace model loading and inference
- **Unsloth**: Efficient vision model fine-tuning
- **Pydantic**: Data validation and schema definition
- **Qdrant**: Vector database for RAG
- **PyTorch**: Deep learning framework
- **PIL/Pillow**: Image processing
- **PyPDFium2**: PDF processing

See `requirements.txt` for complete dependency list with versions.

## 🚨 Troubleshooting

### Common Issues

**Windows File Lock Issues During Training**:
```python
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
```

**Model Download Timeouts**:
```python
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "1200"  # Increase timeout in seconds
os.environ["HF_HUB_ETAG_TIMEOUT"] = "1200"
```

**CUDA Out of Memory**:
- Reduce batch size in training
- Use 4-bit quantization
- Enable gradient checkpointing

**Qdrant Connection Issues**:
- Verify URL and API key in `credentials.env`
- Check network connectivity to Qdrant Cloud
- Ensure proper timeout settings

## 📈 Performance Optimization

1. **Memory Usage**:
   - 4-bit quantization with BitsAndBytes
   - Gradient checkpointing
   - Flash Attention (if available)

2. **Inference Speed**:
   - Use smaller model variants (2B vs 7B)
   - Batch processing
   - GPU acceleration

3. **Training Efficiency**:
   - LoRA adapters (parameter-efficient fine-tuning)
   - Unsloth optimization
   - Data augmentation for smaller datasets

## 📝 Notes

- All models support both inference and fine-tuning
- Vision models require 4-8GB VRAM for inference, 16GB+ for training
- Qdrant supports both cloud and self-hosted deployments
- JSON schema validation is enforced at extraction time

## � Future Implementation

Planned extensions include a FastAPI backend to expose the document processing pipeline as REST APIs, and a Streamlit front end for easy upload, search, and visualization of extraction results.

- **FastAPI Backend**:
  - Serve document upload and indexing endpoints
  - Provide search and retrieval APIs for Qdrant-backed documents
  - Offer JSON output endpoints for structured extraction and reasoning

- **Streamlit Front End**:
  - User-facing interface for uploading scans and PDFs
  - Instant display of extracted metadata, tables, and agent responses
  - Interactive filtering, search, and document review workflows
