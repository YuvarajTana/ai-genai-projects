"""
FastAPI Application
===================
Main API application with endpoints.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool

from src.config import get_config
from src.loaders import load_document, load_documents, LoaderFactory
from src.chunking import chunk_documents, ChunkerFactory
from src.embeddings import SentenceTransformerEmbedder
from src.retrieval import HybridRetriever
from src.generation import OllamaLLM, PromptBuilder, GenerationConfig


# Pydantic models for API
class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., description="The question to ask")
    k: int = Field(4, description="Number of results to retrieve")
    use_reranker: bool = Field(True, description="Whether to use reranking")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str
    sources: List[Dict[str, Any]]


class IndexRequest(BaseModel):
    """Request model for indexing."""
    directory: Optional[str] = Field(None, description="Directory to index")


class IndexResponse(BaseModel):
    """Response model for indexing."""
    status: str
    chunks_indexed: int


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    components: Dict[str, bool]


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI app
    """
    config = get_config()
    
    # Setup logging
    from src.logging_config import setup_logging
    setup_logging(
        level=config.logging.level,
        log_file=config.logging.file,
    )
    
    app = FastAPI(
        title=config.app.name,
        version=config.app.version,
        description="Intermediate RAG System with Hybrid Search and Reranking",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize components (lazy loaded)
    app.state.embedder = None
    app.state.retriever = None
    app.state.llm = None
    app.state.prompt_builder = None
    
    def get_embedder():
        if app.state.embedder is None:
            app.state.embedder = SentenceTransformerEmbedder(
                model_name=config.embeddings.model_name,
                normalize=config.embeddings.normalize,
            )
        return app.state.embedder
    
    def get_retriever():
        if app.state.retriever is None:
            app.state.retriever = HybridRetriever(
                embedder=get_embedder(),
                collection_name=config.vector_db.collection_name,
                persist_directory=config.paths.db_dir,
                use_bm25=config.retrieval.hybrid.enabled,
                use_reranker=config.retrieval.reranking.enabled,
                vector_weight=config.retrieval.hybrid.vector_weight,
                bm25_weight=config.retrieval.hybrid.bm25_weight,
            )
        return app.state.retriever
    
    def get_llm():
        if app.state.llm is None:
            app.state.llm = OllamaLLM(
                model=config.llm.ollama.model,
                base_url=config.llm.ollama.base_url,
                timeout=config.llm.ollama.timeout,
            )
        return app.state.llm
    
    def get_prompt_builder():
        if app.state.prompt_builder is None:
            app.state.prompt_builder = PromptBuilder(
                system_template=config.prompts.system_template,
                qa_template=config.prompts.qa_template,
            )
        return app.state.prompt_builder
    
    # Health check endpoint
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Check the health of system components."""
        llm = get_llm()
        retriever = get_retriever()
        
        return HealthResponse(
            status="healthy",
            components={
                "embedder": True,
                "vector_store": retriever.count() >= 0,
                "ollama": llm.is_available(),
            }
        )
    
    # Index documents endpoint
    @app.post("/index", response_model=IndexResponse)
    async def index_documents(request: IndexRequest = None):
        """Index documents from the data directory."""
        config = get_config()
        data_dir = request.directory if request and request.directory else config.paths.data_dir
        logger.info("Index request for directory: {}", data_dir)
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            raise HTTPException(
                status_code=400,
                detail=f"Data directory created at {data_dir}. Please add documents and try again."
            )
        
        try:
            # Load documents
            documents = load_documents(data_dir)
            
            if not documents:
                raise HTTPException(
                    status_code=400,
                    detail=f"No supported documents found in {data_dir}"
                )
            
            # Chunk documents
            chunks = chunk_documents(
                documents,
                strategy=config.chunking.strategy,
                chunk_size=config.chunking.chunk_size,
                chunk_overlap=config.chunking.chunk_overlap,
            )
            
            # Index chunks
            retriever = get_retriever()
            retriever.index(chunks)
            logger.info("Indexed {} chunks successfully", len(chunks))
            
            return IndexResponse(
                status="success",
                chunks_indexed=len(chunks),
            )
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # Query endpoint
    @app.post("/query", response_model=QueryResponse)
    async def query(request: QueryRequest):
        """Query the RAG system."""
        logger.info("Query: {} (k={}, use_reranker={})", request.query[:50], request.k, request.use_reranker)
        try:
            retriever = get_retriever()
            llm = get_llm()
            prompt_builder = get_prompt_builder()
            
            # Check if index has documents
            if retriever.count() == 0:
                raise HTTPException(
                    status_code=400,
                    detail="No documents indexed. Please index documents first using POST /index"
                )
            
            # Retrieve relevant chunks
            results = retriever.retrieve(
                query=request.query,
                k=request.k,
                use_reranker=request.use_reranker,
            )
            
            if not results:
                return QueryResponse(
                    answer="I couldn't find any relevant information in the documents.",
                    sources=[],
                )
            
            # Build prompt
            prompt = prompt_builder.build_prompt(
                question=request.query,
                results=results,
            )
            
            # Generate answer
            config = get_config()
            gen_config = GenerationConfig(
                temperature=config.llm.ollama.temperature,
                max_tokens=config.llm.ollama.max_tokens,
            )
            
            answer = await llm.agenerate(prompt, gen_config)
            
            # Format sources
            sources = [
                {
                    "source": r.get("metadata", {}).get("source", "Unknown"),
                    "chunk": r.get("metadata", {}).get("chunk_index", 0),
                    "score": r.get("rerank_score", r.get("combined_score", r.get("distance", 0))),
                    "text_preview": r.get("text", "")[:200] + "...",
                }
                for r in results
            ]
            
            return QueryResponse(
                answer=answer.strip(),
                sources=sources,
            )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Query failed: {}", e)
            raise HTTPException(status_code=500, detail=str(e))
    
    # Streaming query endpoint
    @app.post("/query/stream")
    async def query_stream(request: QueryRequest):
        """Query with streaming response."""
        try:
            retriever = get_retriever()
            llm = get_llm()
            prompt_builder = get_prompt_builder()
            
            if retriever.count() == 0:
                raise HTTPException(
                    status_code=400,
                    detail="No documents indexed."
                )
            
            # Retrieve
            results = retriever.retrieve(
                query=request.query,
                k=request.k,
                use_reranker=request.use_reranker,
            )
            
            if not results:
                async def empty_response():
                    yield "I couldn't find any relevant information."
                return StreamingResponse(
                    empty_response(),
                    media_type="text/plain",
                )
            
            # Build prompt
            prompt = prompt_builder.build_prompt(
                question=request.query,
                results=results,
            )
            
            # Stream response (run sync generator in threadpool)
            def generate_sync():
                config = get_config()
                gen_config = GenerationConfig(
                    temperature=config.llm.ollama.temperature,
                    max_tokens=config.llm.ollama.max_tokens,
                )
                for chunk in llm.generate_stream(prompt, gen_config):
                    yield chunk
            
            return StreamingResponse(
                iterate_in_threadpool(generate_sync()),
                media_type="text/plain",
            )
        
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # Get stats endpoint
    @app.get("/stats")
    async def get_stats():
        """Get system statistics."""
        config = get_config()
        retriever = get_retriever()
        return {
            "chunks_indexed": retriever.count(),
            "supported_extensions": LoaderFactory.get_supported_extensions(),
            "chunking_strategies": ChunkerFactory.get_available_strategies(),
            "llm_model": config.llm.ollama.model,
        }
    
    # Clear index endpoint
    @app.delete("/index")
    async def clear_index():
        """Clear all indexed documents."""
        try:
            retriever = get_retriever()
            retriever.clear()
            logger.info("Index cleared")
            return {"status": "success", "message": "Index cleared"}
        except Exception as e:
            logger.error("Clear index failed: {}", e)
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


# Create default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    config = get_config()
    uvicorn.run(
        "main:app",
        host=config.api.host,
        port=config.api.port,
        reload=True,
    )
