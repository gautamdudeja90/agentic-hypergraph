"""Schemas for construction agents (document → hypergraph)."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from schemas.common import Entity, Hyperedge, TextChunk, GraphNode, GraphEdge, EmbeddingItem, EmbeddingResult


# ===== DocumentIngestionAgent Schemas =====

class DocumentInput(BaseModel):
    """Input for document ingestion."""
    content: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ChunkedDocument(BaseModel):
    """Output from document ingestion."""
    doc_id: str
    chunks: List[TextChunk]


# ===== ExtractionAgent Schemas =====

class ChunkExtractionInput(BaseModel):
    """Input for chunk extraction."""
    chunks: List[TextChunk]
    max_gleaning_rounds: int = 2


class ExtractionResult(BaseModel):
    """Output from entity/relationship extraction."""
    entities: List[Entity]
    hyperedges: List[Hyperedge]
    chunks_processed: int = 0


# ===== GraphBuilderAgent Schemas =====

class GraphConstructionInput(BaseModel):
    """Input for graph construction."""
    entities: List[Entity]
    hyperedges: List[Hyperedge]
    merge_strategy: str = "weight_sum"  # or "description_concat"


class GraphStructure(BaseModel):
    """Output from graph construction."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]


# ===== EmbeddingAgent Schemas =====

class EmbeddingInput(BaseModel):
    """Input for embedding."""
    items: List[EmbeddingItem]
    batch_size: int = 100


class EmbeddingOutput(BaseModel):
    """Output from embedding."""
    results: List[EmbeddingResult]
    total_embedded: int
    cache_hits: int = 0


# ===== StorageCoordinatorAgent Schemas =====

class StorageRequest(BaseModel):
    """Request for storage operation."""
    operation: str  # "write", "read", "delete"
    storage_type: str  # "kv", "vector", "graph"
    data: Dict[str, Any]


class StorageResponse(BaseModel):
    """Response from storage operation."""
    success: bool
    operation_id: str
    errors: Optional[List[str]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
