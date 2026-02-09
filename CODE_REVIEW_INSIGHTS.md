# Code Review & Insights: RAG Projects Portfolio

**Reviewed by:** Claude (AI Code Review)
**Date:** February 6, 2026
**Scope:** P1 Simple RAG, P2 Intermediate RAG, and Roadmap

---

## Executive Summary

You're building an ambitious 10-project RAG portfolio that progressively increases in complexity from basic vector search to enterprise-grade production systems. P1 is complete (a clean ~210-line single-file RAG system), and P2 is structurally complete but has several bugs and gaps that should be fixed before moving to P3.

The **evolution from P1 to P2 is impressive** — you went from a monolithic script to a properly layered, extensible architecture with factory patterns, abstract base classes, and a full API+frontend. That said, P2 has some issues that would undermine it as a portfolio piece if left unaddressed.

---

## Pattern 1: Strong Architectural Growth (P1 → P2)

Your biggest strength is how dramatically the architecture matured between projects.

**P1** is a single 213-line file with everything inline — loaders, chunking, vector DB, generation, and CLI all in `rag.py`. Hardcoded constants at the top, no configuration system, no error boundaries between components.

**P2** explodes this into a proper layered architecture with 6 packages, 25+ source files, abstract base classes, factory patterns, YAML configuration with Pydantic validation, and a FastAPI server with a polished dark-theme frontend. The dependency flow is strictly one-directional: `Loaders → Chunking → Embeddings → Retrieval → Generation → API`.

This shows genuine growth in software design thinking. Each layer has its own base class, concrete implementations, and clean `__init__.py` exports. The factory pattern lets you swap chunking strategies or loaders without touching calling code.

---

## Pattern 2: Bugs in the Most Complex Components

The simpler components (text loader, character chunker) are solid. But the most complex, highest-value components have bugs:

**Semantic Chunker — broken minimum chunk size logic:** In `semantic_chunker.py`, the `_create_semantic_chunks` method uses a `for` loop with manual `i += 1` to merge small chunks. In Python, modifying the loop variable inside a `for` loop has **no effect** on the next iteration — the `for` loop resets `i` to the next value from `range()`. This means the minimum chunk size enforcement is completely non-functional, and very small chunks will be emitted regardless.

**Recursive Chunker — ordering issue:** In `_recursive_split`, the `good_splits` (small pieces that don't need further splitting) are appended to `final_chunks` *after* all recursive sub-splits. This means the output order may not match the original text, which can confuse the LLM's context window.

**Chunk ID collisions:** `Chunk.id` uses `filename::chunk_N`, which isn't unique when two files share the same name across different directories (e.g., two `README.md` files). ChromaDB will silently overwrite data.

---

## Pattern 3: Frontend-Backend Contract Mismatch

The web frontend sends a `use_reranker` toggle to the API, but the API endpoint **completely ignores it**. Users clicking the reranking checkbox in the UI see no actual change in behavior. The `QueryRequest` model accepts the field, but the `query()` handler never reads it. This is the kind of gap that would be caught by integration tests — which don't exist yet.

Similarly, the model name is hardcoded as `'llama3.2'` in the frontend rather than being read from the API's `/stats` endpoint.

---

## Pattern 4: Security and Robustness Gaps

**XSS vulnerability:** The frontend's `addMessage` function uses `innerHTML` to inject content directly from LLM responses. If the LLM generates output containing HTML or script tags (which can happen with certain prompts), it will execute in the user's browser. This should use `textContent` or a sanitization library.

**Blocking async calls:** The FastAPI handlers call synchronous LLM generation directly, which blocks the event loop. During the 5-30 seconds an LLM takes to respond, the server can't handle any other requests. The `agenerate()` default implementation wraps sync calls without `asyncio.to_thread()`, which defeats the purpose of async.

**CORS wildcard:** The API config includes `"*"` in `cors_origins`, which allows any domain to make requests. Fine for development but should be documented as something to lock down.

---

## Pattern 5: Test Coverage is Dangerously Low

Estimated functional coverage is roughly **20-25%**. Here's what is and isn't tested:

**Tested:** TextLoader, CharacterChunker, RecursiveChunker, ChunkerFactory, LoaderFactory basics.

**Not tested:** PDFLoader, DocxLoader, HTMLLoader, SemanticChunker (the most complex chunker), VectorStore, BM25Search, HybridRetriever, Reranker, OllamaLLM, PromptBuilder, and all API endpoints.

The most complex and bug-prone components have zero tests. The semantic chunker bug described above would have been caught by a simple test asserting minimum chunk size.

---

## Priority Fix List

### Critical (fix before P3)

| # | Issue | File | Impact |
|---|-------|------|--------|
| 1 | Semantic chunker `for` loop bug — minimum chunk size enforcement broken | `semantic_chunker.py:195` | Produces many tiny, low-quality chunks |
| 2 | Chunk ID collisions from duplicate filenames | `chunking/base.py` | Silent data loss in ChromaDB |
| 3 | XSS vulnerability via `innerHTML` | `frontend/index.html:609` | Security risk in any demo/portfolio context |
| 4 | `use_reranker` API parameter ignored | `api/main.py` | Frontend/backend contract violation |

### Important (fix for portfolio quality)

| # | Issue | File | Impact |
|---|-------|------|--------|
| 5 | Blocking sync LLM calls in async FastAPI | `generation/base.py`, `api/main.py` | Server can only handle one request at a time |
| 6 | BM25 tokenizer lacks stopword removal/stemming | `bm25_search.py` | Lower quality keyword search results |
| 7 | Position tracking uses fragile `text.find(content[:50])` | `recursive_chunker.py`, `semantic_chunker.py` | Incorrect chunk positions in metadata |
| 8 | HTMLLoader hardcodes `lxml` parser without fallback | `html_loader.py` | Runtime crash if lxml not installed |
| 9 | No tests for SemanticChunker, retrieval, generation, or API | `tests/` | Bugs go undetected |

### Nice-to-have

| # | Issue | File | Impact |
|---|-------|------|--------|
| 10 | `sys.path.insert` hack in API | `api/main.py:18` | Fragile; use `pip install -e .` instead |
| 11 | Semantic chunker loads its own SentenceTransformer | `semantic_chunker.py` | Double memory usage for embedding model |
| 12 | Prompt template uses `.format()` — user input with `{` causes errors | `prompts.py` | Edge case UX bug |
| 13 | `extract_images` parameter in PDFLoader accepted but unused | `pdf_loader.py` | Dead code / misleading API |

---

## What's Working Well

These deserve recognition as genuine strengths:

- **Layered architecture** — Clean separation of concerns with one-directional dependencies. This is how production RAG systems are actually built.
- **Configuration system** — Pydantic-validated YAML config with sensible defaults. Very professional.
- **Factory + Strategy patterns** — Swappable chunking strategies and loaders without modifying calling code.
- **Lazy loading** — Heavy models (embeddings, cross-encoder) are loaded only on first use, keeping startup fast.
- **Documentation quality** — Excellent roadmap, consistent docstrings with Args/Returns/Raises, good README for P2.
- **Progressive complexity plan** — The 10-project roadmap is well-thought-out with clear accuracy targets and technology progression.

---

## Recommendations for P3 Readiness

1. **Fix the 4 critical bugs** listed above before starting P3. These undermine P2's credibility.
2. **Add tests for SemanticChunker and HybridRetriever** — these are your differentiating features over P1.
3. **Add a `setup.py` or `pyproject.toml`** — eliminate the `sys.path.insert` hacks and make the project properly installable.
4. **Wire up the reranker toggle** — it's a 5-line fix in `api/main.py` and it makes the demo much more impressive.
5. **Consider adding a simple evaluation script** — even a basic precision@k measurement on a small test set would validate your retrieval improvements over P1.

---

*This review covers architecture, code quality, bugs, and portfolio readiness. It does not cover runtime performance or accuracy benchmarks, which would require running the system against test data.*
