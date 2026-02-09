"""
P2 Intermediate RAG - Main Entry Point
======================================

Usage:
    # Index documents
    python main.py --index
    
    # Start API server
    python main.py --serve
    
    # Interactive CLI chat
    python main.py --chat
    
    # All options
    python main.py --index --serve
"""

import argparse
import os
from pathlib import Path

from loguru import logger

from src.config import get_config, reload_config
from src.loadging_config import setup_logging
from src.loaders import load_documents
from src.chunking import chunk_documents
from src.embeddings import SentenceTransformerEmbedder
from src.retrieval import HybridRetriever
from src.generation import OllamaLLM, PromptBuilder, GenerationConfig


def index_documents(config):
    """Index documents from the data directory."""
    logger.info("Indexing documents...")
    
    data_dir = config.paths.data_dir
    
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        logger.warning("Created {}. Please add documents and re-run.", data_dir)
        return None
    
    # Load documents
    logger.info("Loading from {}...", data_dir)
    documents = load_documents(data_dir)
    
    if not documents:
        logger.warning("No supported documents found in {}", data_dir)
        logger.info("Supported formats: {}", config.documents.supported_extensions)
        return None
    
    logger.info("Loaded {} document(s)", len(documents))
    
    # Chunk documents
    logger.info("Chunking with {} strategy...", config.chunking.strategy)
    chunks = chunk_documents(
        documents,
        strategy=config.chunking.strategy,
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
    )
    logger.info("Created {} chunk(s)", len(chunks))
    
    # Initialize embedder
    logger.info("Loading embedding model: {}", config.embeddings.model_name)
    embedder = SentenceTransformerEmbedder(
        model_name=config.embeddings.model_name,
        normalize=config.embeddings.normalize,
    )
    
    # Initialize retriever and index
    logger.info("Indexing chunks...")
    retriever = HybridRetriever(
        embedder=embedder,
        collection_name=config.vector_db.collection_name,
        persist_directory=config.paths.db_dir,
        use_bm25=config.retrieval.hybrid.enabled,
        use_reranker=config.retrieval.reranking.enabled,
    )
    
    retriever.index(chunks)
    
    logger.info("Indexing complete! {} chunks indexed.", len(chunks))
    return retriever


def chat_loop(config, retriever=None):
    """Interactive chat loop."""
    logger.info("Starting chat interface...")
    logger.info("Type 'exit' or 'quit' to stop.")
    
    # Initialize components if not provided
    if retriever is None:
        embedder = SentenceTransformerEmbedder(
            model_name=config.embeddings.model_name,
            normalize=config.embeddings.normalize,
        )
        retriever = HybridRetriever(
            embedder=embedder,
            collection_name=config.vector_db.collection_name,
            persist_directory=config.paths.db_dir,
            use_bm25=config.retrieval.hybrid.enabled,
            use_reranker=config.retrieval.reranking.enabled,
        )
    
    # Check if we have indexed documents
    if retriever.count() == 0:
        logger.warning("No documents indexed. Run with --index first.")
        return
    
    logger.info("{} chunks available for retrieval.", retriever.count())
    
    # Initialize LLM
    llm = OllamaLLM(
        model=config.llm.ollama.model,
        base_url=config.llm.ollama.base_url,
        timeout=config.llm.ollama.timeout,
    )
    
    if not llm.is_available():
        logger.warning("Ollama not available at {}", config.llm.ollama.base_url)
        logger.info("Make sure Ollama is running: ollama serve")
        return
    
    logger.info("Using model: {}", config.llm.ollama.model)
    
    # Initialize prompt builder
    prompt_builder = PromptBuilder(
        system_template=config.prompts.system_template,
        qa_template=config.prompts.qa_template,
    )
    
    gen_config = GenerationConfig(
        temperature=config.llm.ollama.temperature,
        max_tokens=config.llm.ollama.max_tokens,
    )
    
    logger.info("=" * 60)
    
    while True:
        try:
            query = input("\n🧑 You: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ("exit", "quit", "q"):
                logger.info("Goodbye!")
                break
            
            # Retrieve
            results = retriever.retrieve(
                query=query,
                k=config.retrieval.top_k,
            )
            
            if not results:
                logger.info("Assistant: I couldn't find any relevant information.")
                continue
            
            # Build prompt
            prompt = prompt_builder.build_prompt(
                question=query,
                results=results,
            )
            
            # Generate response
            print("\n🤖 Assistant: ", end="", flush=True)
            
            # Stream response
            for chunk in llm.generate_stream(prompt, gen_config):
                print(chunk, end="", flush=True)
            
            print()
            
            # Show sources
            print("\n📖 Sources:")
            for i, r in enumerate(results[:3], 1):
                source = r.get("metadata", {}).get("source", "Unknown")
                chunk_idx = r.get("metadata", {}).get("chunk_index", 0)
                score = r.get("rerank_score", r.get("combined_score", 0))
                print(f"   {i}. {source} (chunk {chunk_idx}, score: {score:.3f})")
        
        except KeyboardInterrupt:
            logger.info("Goodbye!")
            break
        except Exception as e:
            logger.error("Error: {}", e)


def start_server(config):
    """Start the FastAPI server."""
    logger.info("Starting API server...")
    logger.info("Host: {} Port: {}", config.api.host, config.api.port)
    logger.info("API docs: http://localhost:{}/docs", config.api.port)
    logger.info("Health: http://localhost:{}/health", config.api.port)
    logger.info("Press Ctrl+C to stop.")
    
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.app.debug,
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="P2 Intermediate RAG System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --index              Index documents from ./data
  python main.py --chat               Start interactive chat
  python main.py --serve              Start API server
  python main.py --index --chat       Index then chat
  python main.py --config my.yaml     Use custom config
        """
    )
    
    parser.add_argument(
        "--index",
        action="store_true",
        help="Index documents from data directory",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Start interactive chat",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start FastAPI server",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = reload_config(args.config)
    
    # Setup logging from config
    setup_logging(
        level=config.logging.level,
        log_file=config.logging.file,
    )
    
    logger.info("{} v{}", config.app.name, config.app.version)
    
    # No arguments - show help
    if not (args.index or args.chat or args.serve):
        parser.print_help()
        return
    
    retriever = None
    
    # Index documents
    if args.index:
        retriever = index_documents(config)
    
    # Start chat
    if args.chat:
        chat_loop(config, retriever)
    
    # Start server
    if args.serve:
        start_server(config)


if __name__ == "__main__":
    main()
