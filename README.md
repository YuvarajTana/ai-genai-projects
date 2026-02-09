# AI Generative AI Projects

A portfolio of **Retrieval-Augmented Generation (RAG)** systems built progressively from basic to production-grade. Each project builds on the previous one, adding new capabilities and improving accuracy.

## Overview

| Project | Status | Description |
|---------|--------|-------------|
| **P1: Simple RAG** | ✅ Complete | Basic RAG with character chunking, ChromaDB, and Ollama |
| **P2: Intermediate RAG** | 🚧 Current | Semantic chunking, hybrid search, reranking, FastAPI + Web UI |
| **P3–P10** | Planned | See [RAG_PROJECTS_ROADMAP.md](RAG_PROJECTS_ROADMAP.md) |

## Repository Structure

```
ai-generative-ai-projects/
├── p1_simple_rag/           # Basic RAG (single-file, CLI)
├── p2_intermediate_rag/     # Enhanced RAG (API, hybrid search, Docker)
├── RAG_PROJECTS_ROADMAP.md  # Full 10-project roadmap
├── CODE_REVIEW_INSIGHTS.md  # Architecture and improvement notes
└── README.md                # This file
```

## Quick Start

### P1: Simple RAG

```bash
cd p1_simple_rag
pip install -r requirements.txt
python rag.py --index    # Index documents from ./data
python rag.py --chat     # Interactive chat
```

### P2: Intermediate RAG

```bash
cd p2_intermediate_rag
pip install -e .
ollama serve && ollama pull llama3.2   # Start Ollama
python main.py --index                 # Index documents
python main.py --serve                 # Start API (http://localhost:8000)
```

Or with Docker:

```bash
cd p2_intermediate_rag
docker compose up -d
# API: http://localhost:8000  |  Frontend: http://localhost:3000
```

## Key Documents

- **[RAG_PROJECTS_ROADMAP.md](RAG_PROJECTS_ROADMAP.md)** — Vision, project details, timeline, and technology stack for all 10 projects
- **[CODE_REVIEW_INSIGHTS.md](CODE_REVIEW_INSIGHTS.md)** — Architecture patterns, identified issues, and improvement recommendations

## Technology Stack

- **Vector DB**: ChromaDB
- **Embeddings**: Sentence Transformers
- **LLM**: Ollama (local)
- **API**: FastAPI
- **Frontend**: HTML/CSS/JS (P2)

## License

MIT
