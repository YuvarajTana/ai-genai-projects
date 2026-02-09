"""
Configuration Management
========================
Loads and validates configuration from YAML file and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class PathsConfig(BaseModel):
    """File paths configuration."""
    data_dir: str = "./data"
    db_dir: str = "./chroma_db"
    cache_dir: str = "./cache"


class SemanticChunkingConfig(BaseModel):
    """Semantic chunking settings."""
    breakpoint_threshold: float = 0.3
    min_chunk_size: int = 100


class RecursiveChunkingConfig(BaseModel):
    """Recursive chunking settings."""
    separators: List[str] = ["\n\n", "\n", ". ", " ", ""]


class ChunkingConfig(BaseModel):
    """Chunking configuration."""
    strategy: str = "semantic"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    semantic: SemanticChunkingConfig = SemanticChunkingConfig()
    recursive: RecursiveChunkingConfig = RecursiveChunkingConfig()


class EmbeddingsConfig(BaseModel):
    """Embedding model configuration."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize: bool = True
    batch_size: int = 32


class VectorDBConfig(BaseModel):
    """Vector database configuration."""
    collection_name: str = "docs"
    distance_metric: str = "cosine"


class HybridSearchConfig(BaseModel):
    """Hybrid search settings."""
    enabled: bool = True
    vector_weight: float = 0.7
    bm25_weight: float = 0.3


class RerankingConfig(BaseModel):
    """Reranking configuration."""
    enabled: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 4


class FilteringConfig(BaseModel):
    """Filtering configuration."""
    enabled: bool = True
    default_filters: Dict[str, Any] = {}


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""
    top_k: int = 6
    hybrid: HybridSearchConfig = HybridSearchConfig()
    reranking: RerankingConfig = RerankingConfig()
    filtering: FilteringConfig = FilteringConfig()


class OllamaConfig(BaseModel):
    """Ollama LLM settings."""
    base_url: str = "http://localhost:11434"
    model: str = "llama3.2"
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout: int = 300


class OpenAIConfig(BaseModel):
    """OpenAI LLM settings."""
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 2048


class LLMConfig(BaseModel):
    """LLM configuration."""
    provider: str = "ollama"
    ollama: OllamaConfig = OllamaConfig()
    openai: OpenAIConfig = OpenAIConfig()


class PromptsConfig(BaseModel):
    """Prompt templates."""
    system_template: str = "You are a helpful AI assistant."
    qa_template: str = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


class APIConfig(BaseModel):
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = ["*"]


class CacheConfig(BaseModel):
    """Caching configuration."""
    enabled: bool = True
    ttl: int = 3600
    max_size: int = 1000


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None


class AppConfig(BaseModel):
    """Application metadata."""
    name: str = "P2 Intermediate RAG"
    version: str = "1.0.0"
    debug: bool = False


class DocumentsConfig(BaseModel):
    """Document processing configuration."""
    supported_extensions: List[str] = [".txt", ".md", ".pdf", ".docx", ".html"]


class Config(BaseModel):
    """Main configuration class."""
    app: AppConfig = AppConfig()
    paths: PathsConfig = PathsConfig()
    documents: DocumentsConfig = DocumentsConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    vector_db: VectorDBConfig = VectorDBConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    llm: LLMConfig = LLMConfig()
    prompts: PromptsConfig = PromptsConfig()
    api: APIConfig = APIConfig()
    cache: CacheConfig = CacheConfig()
    logging: LoggingConfig = LoggingConfig()


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Config object with all settings
    """
    config_file = Path(config_path)
    
    if config_file.exists():
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)
        cfg = Config(**config_dict)
    else:
        cfg = Config()

    # Override Ollama URL from environment (e.g. for Docker: OLLAMA_HOST=http://ollama:11434)
    ollama_host = os.environ.get("OLLAMA_HOST")
    if ollama_host:
        cfg.llm.ollama.base_url = ollama_host.rstrip("/")
    return cfg


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config(config_path: str = "config.yaml") -> Config:
    """Reload configuration from file."""
    global _config
    _config = load_config(config_path)
    return _config
