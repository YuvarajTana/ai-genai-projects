# P1: Simple RAG

A foundational Retrieval-Augmented Generation system in a single Python file (~210 lines). Indexes your documents and answers questions using local LLM inference via Ollama.

## Architecture

```
INDEXING:  Documents (./data)  -->  Chunk (900 chars)  -->  Embed (MiniLM-L6-v2)  -->  Store (ChromaDB)

QUERY:    Question  -->  Embed  -->  Vector Search (top-4)  -->  Build Prompt  -->  Ollama LLM  -->  Answer
```

## Features

- Document loaders for TXT, PDF, DOCX, and Markdown
- Character-based chunking with 120-char overlap
- Sentence Transformer embeddings (384 dimensions)
- ChromaDB persistent vector storage
- Ollama local LLM generation
- CLI chat with source citations

## Quick Start

### 1. Setup

```bash
cd p1_simple_rag
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start Ollama

```bash
ollama serve
ollama pull llama3.2
```

### 3. Add Documents

Place your files (`.txt`, `.pdf`, `.docx`, `.md`) in the `./data/` directory.

### 4. Run

```bash
# Index documents
python rag.py --index

# Start interactive chat
python rag.py --chat

# Both at once
python rag.py --index --chat
```

## Configuration

All configuration is via constants at the top of `rag.py`:

| Parameter          | Default            | Description                   |
|--------------------|--------------------|------------------------------ |
| `DATA_DIR`         | `./data`           | Documents directory           |
| `DB_DIR`           | `./chroma_db`      | ChromaDB storage              |
| `EMBED_MODEL_NAME` | `all-MiniLM-L6-v2` | Embedding model               |
| `OLLAMA_URL`       | `localhost:11434`  | Ollama API endpoint           |
| `OLLAMA_MODEL`     | `gpt-oss:20b`      | LLM model name                |
| `chunk_size`       | 900                | Characters per chunk          |
| `overlap`          | 120                | Overlap between chunks        |
| `k`                | 4                  | Number of results to retrieve |

## Project Structure

```
p1_simple_rag/
├── rag.py                      # Complete RAG system (single file)
├── requirements.txt            # Python dependencies
├── PROJECT_DOCUMENTATION.md    # Detailed architecture docs
├── rag_visualization.html      # Interactive flow diagram
├── data/                       # Your documents go here
└── chroma_db/                  # ChromaDB persistent storage
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | ChromaDB (persistent) |
| LLM | Ollama (local inference) |
| PDF parsing | pypdf |
| DOCX parsing | python-docx |

## Known Limitations

- Fixed chunk size (may split mid-sentence)
- Single embedding model, no fine-tuning
- Vector-only search (no keyword matching)
- No reranking of results
- No streaming responses
- No conversation memory
- CLI only (no API or web UI)
- No evaluation metrics

These are addressed in the next project: [P2 Intermediate RAG](../p2_intermediate_rag/).

## License

MIT
