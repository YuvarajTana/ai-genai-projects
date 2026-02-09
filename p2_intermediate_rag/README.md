# P2: Intermediate RAG System

An enhanced Retrieval-Augmented Generation (RAG) system with semantic chunking, hybrid search, cross-encoder reranking, and a modern web interface.

## 🚀 Features

### Improvements over P1
| Feature | P1 Simple | P2 Intermediate |
|---------|-----------|-----------------|
| Chunking | Character-based | Semantic + Recursive |
| Search | Vector only | Hybrid (Vector + BM25) |
| Ranking | Distance only | Cross-encoder reranking |
| Interface | CLI only | CLI + REST API + Web UI |
| Response | Blocking | Streaming support |
| File Types | 4 | 5+ (with HTML) |

### Key Components

- **Semantic Chunking**: Splits documents at natural breakpoints based on sentence similarity
- **Hybrid Search**: Combines dense vector search with sparse BM25 keyword matching
- **Cross-Encoder Reranking**: Uses a second-stage ranker for more accurate results
- **FastAPI Backend**: REST API with streaming support
- **Modern Web UI**: Interactive chat interface with real-time status

## 📁 Project Structure

```
p2_intermediate_rag/
├── src/
│   ├── loaders/           # Document loading (TXT, PDF, DOCX, HTML)
│   ├── chunking/          # Text chunking strategies
│   ├── embeddings/        # Embedding models
│   ├── retrieval/         # Vector store, BM25, reranking
│   ├── generation/        # LLM backends and prompts
│   └── api/               # FastAPI application
├── frontend/              # Web interface
├── data/                  # Place your documents here
├── tests/                 # Unit tests
├── config.yaml            # Configuration
├── main.py                # CLI entry point
└── requirements.txt       # Dependencies
```

## 🛠️ Installation

### 1. Create Virtual Environment

```bash
cd p2_intermediate_rag
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

Option A - Install in editable mode (recommended, enables proper imports):

```bash
pip install -e .
```

Option B - Install from requirements only:

```bash
pip install -r requirements.txt
```

### 3. Download spaCy Model (optional, for better sentence splitting)

```bash
python -m spacy download en_core_web_sm
```

### 4. Start Ollama

Make sure Ollama is running with a model:

```bash
ollama serve
ollama pull llama3.2
```

## 🐳 Docker Quick Start

```bash
# Start everything (Ollama + API + Frontend)
docker-compose up -d

# Pull a model for Ollama (first time)
docker compose exec ollama ollama pull llama3.2

# API: http://localhost:8000
# Frontend: http://localhost:3000
```

Place documents in `./data` before indexing.

## 📖 Usage

### CLI Mode

```bash
# Index documents
python main.py --index

# Start interactive chat
python main.py --chat

# Index then chat
python main.py --index --chat
```

### API Server

```bash
# Start the API server
python main.py --serve
```

API will be available at `http://localhost:8000`

**Endpoints:**
- `GET /health` - Health check
- `GET /stats` - System statistics
- `POST /index` - Index documents
- `POST /query` - Query the RAG system
- `POST /query/stream` - Query with streaming response
- `DELETE /index` - Clear the index

### Web Interface

1. Start the API server: `python main.py --serve`
2. Open `frontend/index.html` in your browser
3. Or serve it with a simple HTTP server:

```bash
cd frontend
python -m http.server 3000
```

Then open `http://localhost:3000`

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
# Chunking
chunking:
  strategy: "semantic"  # semantic, recursive, or character
  chunk_size: 1000
  chunk_overlap: 200

# Retrieval
retrieval:
  top_k: 6
  hybrid:
    enabled: true
    vector_weight: 0.7
    bm25_weight: 0.3
  reranking:
    enabled: true
    model: "cross-encoder/ms-marco-MiniLM-L-6-v2"

# LLM
llm:
  provider: "ollama"
  ollama:
    model: "llama3.2"
    temperature: 0.1
```

## 📊 Chunking Strategies

### 1. Semantic Chunking
Splits text based on semantic similarity between sentences. Creates chunks at natural topic boundaries.

### 2. Recursive Chunking
Hierarchically splits text using separators (paragraphs → sentences → words).

### 3. Character Chunking
Simple fixed-size chunks with overlap (from P1, included as fallback).

## 🔍 Retrieval Pipeline

```
Query
  │
  ├─► Vector Search (semantic similarity)
  │         │
  │         ├─► Top-k candidates
  │         │
  ├─► BM25 Search (keyword matching)
  │         │
  │         ├─► Top-k candidates
  │         │
  └─► Score Fusion (weighted combination)
            │
            ├─► Combined candidates
            │
            └─► Cross-Encoder Reranking
                      │
                      └─► Final top-k results
```

## 🧪 Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## 📈 Evaluation

Compare with P1 using these metrics:
- **Retrieval Precision@k**: How many retrieved documents are relevant
- **Answer Faithfulness**: Is the answer grounded in the context
- **Response Latency**: Time to generate answer

## 🔜 Next Steps (P3)

The next project will add:
- Multi-modal support (images, tables)
- OCR for scanned documents
- Advanced table extraction
- CLIP embeddings for images

## 📝 License

MIT License

## 🙏 Acknowledgments

- [Sentence Transformers](https://www.sbert.net/)
- [ChromaDB](https://www.trychroma.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Ollama](https://ollama.ai/)
