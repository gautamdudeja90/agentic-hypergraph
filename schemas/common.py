"""Common schemas shared across construction and retrieval agents."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class Entity(BaseModel):
    """Entity extracted from text."""
    name: str
    type: str
    description: str
    source_chunk_ids: List[str] = Field(default_factory=list)
    weight: float = 1.0


class Hyperedge(BaseModel):
    """Hyperedge representing n-ary relationship between entities."""
    entities: List[str]  # Entity names
    relationship: str
    description: str
    weight: float = 1.0
    source_chunk_ids: List[str] = Field(default_factory=list)


class TextChunk(BaseModel):
    """Text chunk with metadata."""
    chunk_id: str
    content: str
    tokens: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    doc_id: Optional[str] = None


class GraphNode(BaseModel):
    """Node in the hypergraph."""
    node_id: str
    node_type: str  # "entity" or "hyperedge"
    name: str
    description: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    weight: float = 1.0


class GraphEdge(BaseModel):
    """Edge connecting nodes in the hypergraph."""
    source_id: str
    target_id: str
    edge_type: str  # e.g., "connected_to", "part_of"
    weight: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EmbeddingItem(BaseModel):
    """Item to be embedded."""
    item_id: str
    item_type: str  # "entity", "hyperedge", "chunk"
    text: str


class EmbeddingResult(BaseModel):
    """Result of embedding operation."""
    item_id: str
    embedding: List[float]
    cached: bool = False
