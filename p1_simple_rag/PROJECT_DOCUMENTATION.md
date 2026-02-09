# P1: Simple RAG System

## Project Overview

This project implements a basic **Retrieval-Augmented Generation (RAG)** system that combines document retrieval with Large Language Model (LLM) generation to answer questions based on your own documents.

### What is RAG?

RAG is an AI framework that enhances LLM responses by:
1. **Retrieving** relevant information from a knowledge base
2. **Augmenting** the prompt with this context
3. **Generating** accurate, grounded responses

This approach solves the "hallucination" problem of LLMs by grounding responses in actual documents.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     INDEXING PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   📄 Documents     ✂️ Chunking      🧮 Embedding     💾 Storage  │
│   ─────────────→  ─────────────→  ─────────────→  ───────────  │
│   ./data/         900 chars       MiniLM-L6-v2    ChromaDB     │
│   TXT,PDF,DOCX    120 overlap     384-dim vectors              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ❓ Query  →  🧮 Embed  →  🔍 Search  →  📝 Prompt  →  🤖 LLM  │
│   User         Vector       Top-k         Context       Ollama  │
│   Question     384-dim      Chunks        Building      Answer  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Document Loaders
| Format | Library | Function |
|--------|---------|----------|
| `.txt`, `.md` | Built-in | `load_txt()` |
| `.pdf` | pypdf | `load_pdf()` |
| `.docx` | python-docx | `load_docx()` |

### 2. Text Chunking
- **Method**: Character-based sliding window
- **Chunk Size**: 900 characters
- **Overlap**: 120 characters
- **Purpose**: Split documents into manageable pieces while preserving context

### 3. Embedding Model
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Type**: Sentence embeddings (normalized)

### 4. Vector Database
- **Database**: ChromaDB (persistent storage)
- **Location**: `./chroma_db/`
- **Collection**: `docs`

### 5. LLM Backend
- **Service**: Ollama (local)
- **Model**: `gpt-oss:20b`
- **Endpoint**: `http://localhost:11434/api/generate`

---

## Usage

### Index Documents
```bash
python rag.py --index
```
Place your documents in the `./data/` directory first.

### Start Chat
```bash
python rag.py --chat
```

### Both Operations
```bash
python rag.py --index --chat
```

---

## Configuration

| Parameter | Value | Location |
|-----------|-------|----------|
| `DATA_DIR` | `./data` | Line 14 |
| `DB_DIR` | `./chroma_db` | Line 15 |
| `COLLECTION_NAME` | `docs` | Line 16 |
| `EMBED_MODEL_NAME` | `all-MiniLM-L6-v2` | Line 17 |
| `OLLAMA_URL` | `localhost:11434` | Line 19 |
| `OLLAMA_MODEL` | `gpt-oss:20b` | Line 20 |
| `chunk_size` | 900 | Line 53 |
| `overlap` | 120 | Line 53 |
| `k` (retrieval) | 4 | Line 134 |

---

## Limitations

### 1. 📄 Document Processing Limitations

| Limitation | Impact | Description |
|------------|--------|-------------|
| **Fixed Chunk Size** | Poor context handling | 900 chars may split sentences mid-way, losing meaning |
| **No Semantic Chunking** | Lost structure | Doesn't respect document structure (sections, paragraphs) |
| **Limited File Types** | Narrow coverage | Only TXT, PDF, DOCX, MD supported |
| **No OCR Support** | Missed content | Scanned PDFs and images cannot be processed |
| **No Table Extraction** | Data loss | Tables in PDFs/DOCX are poorly handled |
| **No Metadata Extraction** | Missing context | Document titles, dates, authors not captured |

### 2. 🧮 Embedding Limitations

| Limitation | Impact | Description |
|------------|--------|-------------|
| **Single Embedding Model** | Quality ceiling | MiniLM is fast but not the most accurate |
| **No Query Expansion** | Missed relevant docs | Single query vector may miss synonyms/related terms |
| **No Fine-tuning** | Domain mismatch | Generic embeddings may not suit specialized domains |
| **English-centric** | Language barrier | Model optimized for English text only |

### 3. 🔍 Retrieval Limitations

| Limitation | Impact | Description |
|------------|--------|-------------|
| **Fixed Top-k** | Rigid results | Always returns 4 chunks regardless of relevance |
| **No Reranking** | Suboptimal order | Chunks ordered only by embedding distance |
| **No Hybrid Search** | Missed matches | Pure vector search misses exact keyword matches |
| **No Filtering** | No precision control | Cannot filter by source, date, or other metadata |
| **No Relevance Threshold** | Noise in results | May return irrelevant chunks if nothing matches |

### 4. 🤖 Generation Limitations

| Limitation | Impact | Description |
|------------|--------|-------------|
| **No Streaming** | Poor UX | User waits for complete response |
| **Single LLM** | No flexibility | Tied to Ollama and specific model |
| **Basic Prompt** | Suboptimal responses | Simple prompt template, no few-shot examples |
| **No Memory** | No conversation context | Each query is independent, no chat history |
| **No Source Verification** | Trust issues | Cannot verify if answer actually uses context |

### 5. 🏗️ System Limitations

| Limitation | Impact | Description |
|------------|--------|-------------|
| **CLI Only** | Limited accessibility | No web interface or API |
| **No Incremental Indexing** | Slow updates | Must re-index everything when adding documents |
| **No Duplicate Detection** | Wasted storage | Same document can be indexed multiple times |
| **No Error Recovery** | Fragile system | Failures during indexing lose all progress |
| **No Evaluation Metrics** | Unknown quality | No way to measure retrieval/generation quality |
| **No Caching** | Slow repeated queries | Same query re-embeds and re-generates |

---

## Roadmap: P2 Intermediate RAG

The next project (`p2_intermediate_rag`) will address these limitations with advanced features:

### Phase 1: Enhanced Document Processing
- [ ] **Semantic Chunking** — Split by sentences/paragraphs, respect document structure
- [ ] **Recursive Chunking** — Hierarchical splitting for better context
- [ ] **More File Types** — Add HTML, JSON, CSV, Excel support
- [ ] **OCR Integration** — Process scanned documents using Tesseract
- [ ] **Table Extraction** — Properly handle tabular data
- [ ] **Metadata Extraction** — Capture document titles, dates, authors

### Phase 2: Advanced Embedding & Retrieval
- [ ] **Multiple Embedding Models** — Support OpenAI, Cohere, custom models
- [ ] **Hybrid Search** — Combine vector search with BM25 keyword search
- [ ] **Query Expansion** — Generate multiple query variations
- [ ] **Reranking** — Use cross-encoder for better result ordering
- [ ] **Filtering** — Filter by metadata, date ranges, sources
- [ ] **Relevance Threshold** — Minimum similarity score to return results
- [ ] **Parent-Child Retrieval** — Return larger context for small matches

### Phase 3: Improved Generation
- [ ] **Streaming Responses** — Real-time token streaming
- [ ] **Multiple LLM Backends** — Support OpenAI, Anthropic, local models
- [ ] **Advanced Prompting** — Chain-of-thought, few-shot examples
- [ ] **Conversation Memory** — Maintain chat history across turns
- [ ] **Citation Verification** — Highlight which sources support each claim
- [ ] **Response Evaluation** — Check faithfulness to context

### Phase 4: System Improvements
- [ ] **Web Interface** — FastAPI backend + React/Vue frontend
- [ ] **REST API** — Programmatic access to all features
- [ ] **Incremental Indexing** — Add/update/delete individual documents
- [ ] **Duplicate Detection** — Skip already-indexed content
- [ ] **Batch Processing** — Parallel document processing
- [ ] **Caching Layer** — Cache embeddings and responses
- [ ] **Evaluation Framework** — Measure retrieval precision/recall, answer quality

### Phase 5: Production Features
- [ ] **Authentication** — User management and access control
- [ ] **Multi-tenancy** — Separate document collections per user/team
- [ ] **Monitoring** — Logging, metrics, alerting
- [ ] **Rate Limiting** — Protect against abuse
- [ ] **Cost Tracking** — Monitor API usage and costs

---

## P2 Project Structure (Proposed)

```
p2_intermediate_rag/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── loaders/
│   │   ├── __init__.py
│   │   ├── base.py            # Base loader interface
│   │   ├── pdf_loader.py      # Enhanced PDF with OCR
│   │   ├── docx_loader.py
│   │   ├── html_loader.py
│   │   └── table_loader.py    # Table extraction
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── semantic.py        # Sentence-based chunking
│   │   ├── recursive.py       # Hierarchical chunking
│   │   └── character.py       # Simple chunking (fallback)
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── base.py            # Embedding interface
│   │   ├── sentence_transformer.py
│   │   ├── openai.py
│   │   └── cohere.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_search.py   # ChromaDB/Pinecone
│   │   ├── bm25_search.py     # Keyword search
│   │   ├── hybrid.py          # Combined search
│   │   └── reranker.py        # Cross-encoder reranking
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── base.py            # LLM interface
│   │   ├── ollama.py
│   │   ├── openai.py
│   │   └── prompts.py         # Prompt templates
│   ├── memory/
│   │   ├── __init__.py
│   │   └── conversation.py    # Chat history management
│   └── api/
│       ├── __init__.py
│       ├── main.py            # FastAPI app
│       └── routes.py          # API endpoints
├── frontend/                   # Web UI (optional)
│   ├── index.html
│   └── app.js
├── tests/
│   ├── test_loaders.py
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   └── test_generation.py
├── data/                       # Document storage
├── chroma_db/                  # Vector database
├── config.yaml                 # Configuration file
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Technology Stack Comparison

| Component | P1 (Simple) | P2 (Intermediate) |
|-----------|-------------|-------------------|
| **Document Loading** | Basic loaders | LangChain/LlamaIndex loaders |
| **Chunking** | Character-based | Semantic + Recursive |
| **Embeddings** | MiniLM only | Multiple providers |
| **Vector DB** | ChromaDB | ChromaDB + Pinecone option |
| **Search** | Vector only | Hybrid (Vector + BM25) |
| **Reranking** | None | Cross-encoder |
| **LLM** | Ollama only | Multiple backends |
| **Interface** | CLI | Web + API + CLI |
| **Memory** | None | Conversation history |
| **Caching** | None | Redis/in-memory |

---

## Key Libraries for P2

```txt
# Core
langchain>=0.1.0          # Document processing, chains
llama-index>=0.9.0        # Alternative RAG framework
chromadb>=0.4.0           # Vector database
pinecone-client>=3.0.0    # Cloud vector DB (optional)

# Embeddings
sentence-transformers>=2.2.0
openai>=1.0.0
cohere>=4.0.0

# Document Processing
pypdf>=3.0.0
python-docx>=0.8.0
beautifulsoup4>=4.12.0    # HTML parsing
pytesseract>=0.3.0        # OCR
pandas>=2.0.0             # Table handling
openpyxl>=3.1.0           # Excel support

# Retrieval
rank-bm25>=0.2.0          # BM25 keyword search
sentence-transformers     # Cross-encoder reranking

# API & Web
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0

# Caching & Storage
redis>=4.0.0
diskcache>=5.6.0

# Evaluation
ragas>=0.1.0              # RAG evaluation
deepeval>=0.20.0          # LLM evaluation
```

---

## Success Metrics for P2

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Retrieval Precision@4** | >80% | RAGAS evaluation |
| **Answer Faithfulness** | >90% | RAGAS evaluation |
| **Response Latency** | <3s | API timing |
| **Throughput** | >10 req/s | Load testing |
| **Document Types** | 8+ | Feature completeness |
| **Test Coverage** | >80% | pytest-cov |

---

## Getting Started with P2

1. **Create project structure**
   ```bash
   mkdir -p p2_intermediate_rag/src/{loaders,chunking,embeddings,retrieval,generation,memory,api}
   mkdir -p p2_intermediate_rag/{tests,data,frontend}
   ```

2. **Install dependencies**
   ```bash
   cd p2_intermediate_rag
   pip install -r requirements.txt
   ```

3. **Start with document processing improvements**
   - Implement semantic chunking
   - Add more file type support
   - Extract metadata

4. **Then enhance retrieval**
   - Add hybrid search
   - Implement reranking

5. **Build the API**
   - FastAPI endpoints
   - Web interface

---

## References

- [LangChain Documentation](https://python.langchain.com/)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [RAGAS - RAG Evaluation](https://docs.ragas.io/)

---

*Document Version: 1.0*  
*Last Updated: January 2026*  
*Next Project: p2_intermediate_rag*
