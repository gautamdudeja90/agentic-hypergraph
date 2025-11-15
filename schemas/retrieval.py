"""Schemas for retrieval agents (query → response)."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


# ===== QueryPlannerAgent Schemas =====

class QueryInput(BaseModel):
    """Input for query planning."""
    query: str
    mode: Optional[str] = "hybrid"  # "local", "global", "hybrid"
    top_k: int = 20


class RetrievalTask(BaseModel):
    """Task for retrieval agent."""
    agent: str  # "entity", "relationship", "text"
    parameters: Dict[str, Any]


class QueryPlan(BaseModel):
    """Output from query planner."""
    query_id: str
    keywords: List[str]
    mode: str
    retrieval_tasks: List[RetrievalTask]
    cached_result: Optional[str] = None


# ===== EntityRetrievalAgent Schemas =====

class EntityRetrievalInput(BaseModel):
    """Input for entity retrieval."""
    keywords: List[str]
    top_k: int = 20
    include_neighbors: bool = True


class RetrievedEntity(BaseModel):
    """Retrieved entity with metadata."""
    entity_id: str
    name: str
    type: str
    description: str
    relevance_score: float
    degree: int = 0
    neighbors: Optional[List[str]] = Field(default_factory=list)
    related_edges: Optional[List[str]] = Field(default_factory=list)


class EntityRetrievalResult(BaseModel):
    """Output from entity retrieval."""
    entities: List[RetrievedEntity]


# ===== RelationshipRetrievalAgent Schemas =====

class RelationshipRetrievalInput(BaseModel):
    """Input for relationship retrieval."""
    keywords: List[str]
    top_k: int = 20
    include_entities: bool = True


class RetrievedRelationship(BaseModel):
    """Retrieved relationship with metadata."""
    hyperedge_id: str
    description: str
    connected_entities: List[str]
    relevance_score: float
    source_chunk_ids: List[str] = Field(default_factory=list)
    weight: float = 1.0


class RelationshipRetrievalResult(BaseModel):
    """Output from relationship retrieval."""
    relationships: List[RetrievedRelationship]


# ===== TextRetrievalAgent Schemas =====

class TextRetrievalInput(BaseModel):
    """Input for text retrieval."""
    entity_ids: List[str]
    hyperedge_ids: List[str]
    top_k: int = 10
    max_tokens: int = 2000


class TextUnit(BaseModel):
    """Retrieved text unit."""
    chunk_id: str
    content: str
    tokens: int
    relevance_score: float
    source_entities: List[str] = Field(default_factory=list)


class TextRetrievalResult(BaseModel):
    """Output from text retrieval."""
    text_units: List[TextUnit]
    total_tokens: int


# ===== ContextAssemblyAgent Schemas =====

class ContextAssemblyInput(BaseModel):
    """Input for context assembly."""
    entities: List[RetrievedEntity]
    relationships: List[RetrievedRelationship]
    texts: List[TextUnit]
    mode: str
    max_tokens: int = 4000


class AssembledContext(BaseModel):
    """Output from context assembly."""
    entity_context: str  # CSV formatted
    relationship_context: str  # CSV formatted
    text_context: str
    total_tokens: int
    mode: str


# ===== ResponseGeneratorAgent Schemas =====

class ResponseGenerationInput(BaseModel):
    """Input for response generation."""
    query: str
    context: AssembledContext
    model: str = "gpt-4o-mini"
    stream: bool = False


class ResponseOutput(BaseModel):
    """Output from response generation."""
    response: str
    query_id: str
    tokens_used: int
    cached: bool = False
