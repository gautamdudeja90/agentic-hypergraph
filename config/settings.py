"""Configuration for HyperGraphRAG multi-agent system."""

from pydantic import BaseModel, Field
from typing import Optional


class HyperGraphRAGConfig(BaseModel):
    """Complete configuration for production-ready HyperGraphRAG."""

    # ===== Construction Settings =====
    chunk_token_size: int = 1200
    chunk_overlap: int = 100
    entity_extract_max_gleaning: int = 2

    # ===== Retrieval Settings =====
    top_k_entities: int = 20
    top_k_relationships: int = 20
    top_k_texts: int = 10
    context_max_tokens: int = 4000

    # ===== LLM Settings =====
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 100
    embedding_dimensions: int = 1536

    # ===== Storage Backends (Open Source) =====
    kv_storage_type: str = "JsonKVStorage"
    vector_storage_type: str = "NanoVectorDBStorage"
    graph_storage_type: str = "NetworkXStorage"

    # Storage paths
    kv_storage_path: str = "./data/kv_storage"
    vector_storage_path: str = "./data/vector_storage"
    graph_storage_path: str = "./data/graph_storage"

    # ===== Caching =====
    enable_embedding_cache: bool = True
    enable_query_cache: bool = True
    enable_llm_cache: bool = True

    # ===== Concurrency =====
    max_concurrent_llm_calls: int = 16
    max_concurrent_embeddings: int = 64

    # ===== ADK Production Features =====
    # Session persistence
    session_db_path: str = "./data/sessions.db"
    enable_session_persistence: bool = True

    # Artifact management
    artifact_base_path: str = "./data/artifacts"
    enable_artifact_versioning: bool = True

    # Resumability & fault tolerance
    enable_resumability: bool = True
    max_llm_calls_per_session: int = 500

    # Event compaction for long sessions
    event_compaction_enabled: bool = True
    event_sliding_window_size: int = 100
    event_overlap_size: int = 10

    # Monitoring & observability
    enable_logging_plugin: bool = True
    enable_error_recovery: bool = True
    max_tool_retries: int = 3
    log_level: str = "INFO"

    # Streaming
    streaming_mode: str = "SSE"  # "SSE", "BIDI", or "NONE"

    # OpenAI API Key (required)
    openai_api_key: Optional[str] = None


# Default configuration instance
default_config = HyperGraphRAGConfig()
