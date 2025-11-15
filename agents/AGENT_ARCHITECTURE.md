# HyperGraphRAG Multi-Agent Architecture Plan

## Overview

This document outlines the complete plan for converting HyperGraphRAG into a Google ADK-based multi-agent application. The system is divided into **Construction Agents** (building hypergraphs) and **Retrieval Agents** (querying hypergraphs).

---

## Source Analysis Summary

### Original HyperGraphRAG Components

**Core Components:**
- `HyperGraphRAG`: Central orchestrator managing the entire pipeline
- `BaseVectorStorage`: Semantic search over entities, hyperedges, chunks
- `BaseKVStorage`: Key-value storage for documents and chunks
- `BaseGraphStorage`: Graph storage for entity-relationship network
- `extract_entities`: LLM-based entity/relationship extraction
- `kg_query`: Query orchestration and retrieval

**Construction Pipeline:**
1. Document insertion → Hashing & deduplication
2. Chunking → Token-based splitting (1200 tokens, 100 overlap)
3. Entity extraction → LLM multi-round gleaning
4. Hyperedge construction → N-ary relationship creation
5. Entity processing → Merge duplicates, weight aggregation
6. Graph storage → Insert nodes/edges into graph DB
7. Vector indexing → Embed and store in vector DB

**Retrieval Pipeline:**
1. Query parsing → Extract keywords via LLM
2. Context building → Route to local/global/hybrid mode
3. Entity retrieval → Vector search + neighbor fetching
4. Relationship retrieval → Hyperedge search + entity connections
5. Text unit retrieval → Trace to source chunks
6. Context assembly → Format as CSV with token limits
7. Response generation → LLM with assembled context

---

## Agent Architecture

### Construction Agents (Document → Hypergraph)

#### 1. DocumentIngestionAgent
**Responsibilities:**
- Accept raw documents as input
- Generate document IDs via MD5 hashing
- Check for duplicates in KV storage
- Chunk documents by token size (default: 1200 tokens, 100 overlap)
- Store chunks in KV storage

**Input Schema:**
```python
class DocumentInput(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None
```

**Output Schema:**
```python
class ChunkedDocument(BaseModel):
    doc_id: str
    chunks: List[TextChunk]

class TextChunk(BaseModel):
    chunk_id: str
    content: str
    tokens: int
    metadata: Dict[str, Any]
```

**Key Functions:**
- `hash_document()` - Generate unique ID
- `check_duplicate()` - Query KV storage
- `chunk_by_tokens()` - Split using tiktoken
- `store_chunks()` - Save to KV storage

---

#### 2. ExtractionAgent
**Responsibilities:**
- Extract entities from text chunks using LLM
- Extract hyperedges (n-ary relationships) from chunks
- Perform multi-round gleaning (iterative refinement)
- Parse and validate extraction output

**Input Schema:**
```python
class ChunkExtractionInput(BaseModel):
    chunks: List[TextChunk]
    max_gleaning_rounds: int = 2
```

**Output Schema:**
```python
class ExtractionResult(BaseModel):
    entities: List[Entity]
    hyperedges: List[Hyperedge]

class Entity(BaseModel):
    name: str
    type: str
    description: str
    source_chunk_ids: List[str]

class Hyperedge(BaseModel):
    entities: List[str]  # Entity names
    relationship: str
    description: str
    weight: float = 1.0
    source_chunk_ids: List[str]
```

**Key Functions:**
- `extract_entities_llm()` - Initial LLM extraction
- `gleaning_round()` - Iterative refinement
- `parse_extraction_output()` - Validate and structure
- `batch_extract()` - Parallel processing of chunks

**LLM Prompts:**
- Entity extraction prompt with examples
- Relationship extraction prompt
- Gleaning refinement prompt

---

#### 3. GraphBuilderAgent
**Responsibilities:**
- Merge duplicate entities across chunks
- Aggregate entity weights and descriptions
- Construct hyperedge nodes in graph
- Build edges connecting hyperedges to entities
- Manage entity summarization (if description exceeds token limit)

**Input Schema:**
```python
class GraphConstructionInput(BaseModel):
    entities: List[Entity]
    hyperedges: List[Hyperedge]
    merge_strategy: str = "weight_sum"  # or "description_concat"
```

**Output Schema:**
```python
class GraphStructure(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]

class GraphNode(BaseModel):
    node_id: str
    node_type: str  # "entity" or "hyperedge"
    name: str
    description: str
    metadata: Dict[str, Any]

class GraphEdge(BaseModel):
    source_id: str
    target_id: str
    edge_type: str
    weight: float
```

**Key Functions:**
- `merge_entities()` - Deduplicate and merge
- `aggregate_weights()` - Sum or average weights
- `summarize_description()` - LLM-based summarization if needed
- `create_hyperedge_nodes()` - Convert to graph nodes
- `build_edges()` - Connect hyperedges to entities

---

#### 4. EmbeddingAgent
**Responsibilities:**
- Batch embed entities, hyperedges, and chunks
- Manage embedding cache for deduplication
- Store embeddings in vector DB
- Handle rate limiting and concurrency control

**Input Schema:**
```python
class EmbeddingInput(BaseModel):
    items: List[EmbeddingItem]
    batch_size: int = 100

class EmbeddingItem(BaseModel):
    item_id: str
    item_type: str  # "entity", "hyperedge", "chunk"
    text: str
```

**Output Schema:**
```python
class EmbeddingResult(BaseModel):
    item_id: str
    embedding: List[float]
    cached: bool
```

**Key Functions:**
- `batch_embed()` - Parallel embedding with semaphore
- `check_cache()` - Query embedding cache
- `store_vectors()` - Insert into vector DB
- `compute_embedding()` - Call OpenAI/Ollama API

---

#### 5. StorageCoordinatorAgent
**Responsibilities:**
- Orchestrate writes to KV, Vector, and Graph storage
- Ensure consistency across storage backends
- Handle transaction-like operations
- Manage storage backend initialization

**Input Schema:**
```python
class StorageRequest(BaseModel):
    operation: str  # "write", "read", "delete"
    storage_type: str  # "kv", "vector", "graph"
    data: Dict[str, Any]
```

**Output Schema:**
```python
class StorageResponse(BaseModel):
    success: bool
    operation_id: str
    errors: Optional[List[str]]
```

**Key Functions:**
- `coordinate_write()` - Multi-storage transaction
- `rollback_on_failure()` - Consistency guarantee
- `initialize_backends()` - Setup storage connections
- `health_check()` - Monitor storage status

---

### Retrieval Agents (Query → Response)

#### 1. QueryPlannerAgent
**Responsibilities:**
- Parse user query to extract keywords and intent
- Determine retrieval mode (local/global/hybrid)
- Route to appropriate retrieval agents
- Check query cache

**Input Schema:**
```python
class QueryInput(BaseModel):
    query: str
    mode: Optional[str] = "hybrid"  # "local", "global", "hybrid"
    top_k: int = 20
```

**Output Schema:**
```python
class QueryPlan(BaseModel):
    query_id: str
    keywords: List[str]
    mode: str
    retrieval_tasks: List[RetrievalTask]
    cached_result: Optional[str] = None

class RetrievalTask(BaseModel):
    agent: str  # "entity", "relationship", "text"
    parameters: Dict[str, Any]
```

**Key Functions:**
- `extract_keywords_llm()` - LLM-based keyword extraction
- `determine_mode()` - Query classification
- `check_cache()` - Query result cache lookup
- `create_plan()` - Generate retrieval strategy

---

#### 2. EntityRetrievalAgent
**Responsibilities:**
- Vector search for similar entities
- Fetch entity metadata (type, description, degree)
- Retrieve one-hop neighbors in graph
- Rank entities by relevance and degree

**Input Schema:**
```python
class EntityRetrievalInput(BaseModel):
    keywords: List[str]
    top_k: int = 20
    include_neighbors: bool = True
```

**Output Schema:**
```python
class EntityRetrievalResult(BaseModel):
    entities: List[RetrievedEntity]

class RetrievedEntity(BaseModel):
    entity_id: str
    name: str
    type: str
    description: str
    relevance_score: float
    degree: int
    neighbors: Optional[List[str]]
    related_edges: Optional[List[str]]
```

**Key Functions:**
- `vector_search_entities()` - Query vector DB
- `fetch_entity_metadata()` - Get from graph storage
- `compute_node_degree()` - Graph traversal
- `get_neighbors()` - One-hop entity retrieval
- `rank_by_relevance()` - Score and sort

---

#### 3. RelationshipRetrievalAgent
**Responsibilities:**
- Vector search for similar hyperedges
- Fetch entities connected to hyperedges
- Trace back to source text chunks
- Rank relationships by relevance

**Input Schema:**
```python
class RelationshipRetrievalInput(BaseModel):
    keywords: List[str]
    top_k: int = 20
    include_entities: bool = True
```

**Output Schema:**
```python
class RelationshipRetrievalResult(BaseModel):
    relationships: List[RetrievedRelationship]

class RetrievedRelationship(BaseModel):
    hyperedge_id: str
    description: str
    connected_entities: List[str]
    relevance_score: float
    source_chunk_ids: List[str]
    weight: float
```

**Key Functions:**
- `vector_search_hyperedges()` - Query vector DB
- `fetch_connected_entities()` - Graph traversal
- `get_source_chunks()` - Trace to text units
- `rank_relationships()` - Score and sort

---

#### 4. TextRetrievalAgent
**Responsibilities:**
- Trace from entity/relationship IDs to source chunks
- Rank chunks by relevance to query
- Enforce token budget per context type
- Return formatted text units

**Input Schema:**
```python
class TextRetrievalInput(BaseModel):
    entity_ids: List[str]
    hyperedge_ids: List[str]
    top_k: int = 10
    max_tokens: int = 2000
```

**Output Schema:**
```python
class TextRetrievalResult(BaseModel):
    text_units: List[TextUnit]
    total_tokens: int

class TextUnit(BaseModel):
    chunk_id: str
    content: str
    tokens: int
    relevance_score: float
    source_entities: List[str]
```

**Key Functions:**
- `trace_to_chunks()` - Map entities/edges to chunks
- `rank_by_relevance()` - Score chunks
- `enforce_token_budget()` - Limit token usage
- `format_text_units()` - Structure output

---

#### 5. ContextAssemblyAgent
**Responsibilities:**
- Combine entity, relationship, and text retrieval results
- Format as CSV or structured context
- Enforce global token limits
- Handle mode-specific assembly (local/global/hybrid)

**Input Schema:**
```python
class ContextAssemblyInput(BaseModel):
    entities: List[RetrievedEntity]
    relationships: List[RetrievedRelationship]
    texts: List[TextUnit]
    mode: str
    max_tokens: int = 4000
```

**Output Schema:**
```python
class AssembledContext(BaseModel):
    entity_context: str  # CSV formatted
    relationship_context: str  # CSV formatted
    text_context: str
    total_tokens: int
    mode: str
```

**Key Functions:**
- `format_entities_csv()` - Convert to CSV
- `format_relationships_csv()` - Convert to CSV
- `combine_contexts()` - Merge based on mode
- `enforce_token_limits()` - Truncate if needed
- `build_final_context()` - Assemble all parts

---

#### 6. ResponseGeneratorAgent
**Responsibilities:**
- Build LLM prompt with assembled context
- Call LLM for response generation
- Clean up response artifacts
- Cache result for future queries

**Input Schema:**
```python
class ResponseGenerationInput(BaseModel):
    query: str
    context: AssembledContext
    model: str = "gpt-4o-mini"
    stream: bool = False
```

**Output Schema:**
```python
class ResponseOutput(BaseModel):
    response: str
    query_id: str
    tokens_used: int
    cached: bool
```

**Key Functions:**
- `build_system_prompt()` - Create prompt with context
- `call_llm()` - Generate response
- `clean_response()` - Remove artifacts
- `cache_result()` - Store for reuse

---

### Shared Service Agents

#### 1. LLMServiceAgent
**Responsibilities:**
- Manage OpenAI/Ollama API calls
- Implement retry logic with exponential backoff
- Handle streaming responses
- Pool and reuse connections

**Key Functions:**
- `async_completion()` - Single completion call
- `async_batch_completion()` - Batch processing
- `stream_completion()` - Streaming support
- `handle_retry()` - Tenacity-based retry

---

#### 2. CacheAgent
**Responsibilities:**
- Manage embedding cache (deduplication)
- Handle query result cache
- Implement cache invalidation
- Persist cache to disk

**Key Functions:**
- `get_cached_embedding()` - Lookup by text hash
- `set_cached_embedding()` - Store embedding
- `get_cached_query()` - Lookup by query hash
- `invalidate_cache()` - Clear specific entries

---

## Agent Communication Patterns

### Construction Flow
```
User Document
    ↓
DocumentIngestionAgent
    ├→ Chunks → KV Storage
    ├→ Message: ChunkedDocument → ExtractionAgent (parallel per chunk)
    ↓
ExtractionAgent (parallel workers)
    ├→ Entities, Hyperedges
    ├→ Message: ExtractionResult → GraphBuilderAgent
    ↓
GraphBuilderAgent
    ├→ Merge & Build Graph Structure
    ├→ Message: GraphStructure → StorageCoordinatorAgent
    ├→ Message: EmbeddingItems → EmbeddingAgent
    ↓
[Parallel]
    EmbeddingAgent → Vector Storage
    StorageCoordinatorAgent → Graph Storage
    ↓
Construction Complete
```

### Retrieval Flow
```
User Query
    ↓
QueryPlannerAgent
    ├→ Check Cache (hit? → return)
    ├→ Extract Keywords
    ├→ Determine Mode
    ├→ Create QueryPlan
    ↓
[Parallel Retrieval]
    ├→ EntityRetrievalAgent → Entities
    ├→ RelationshipRetrievalAgent → Relationships
    └→ TextRetrievalAgent → Text Units
    ↓
ContextAssemblyAgent
    ├→ Combine Results
    ├→ Format Context
    ↓
ResponseGeneratorAgent
    ├→ Build Prompt
    ├→ Call LLM
    ├→ Cache Result
    ↓
Response to User
```

---

## Google ADK Production Features & Implementation Guidelines

### Production-Ready Features from Google ADK

Based on analysis of `/Users/gautamdudeja/repo/adk-python/src/google/adk`, our system will leverage:

#### 1. **Long-Running & Stateful Operations**
- **Session Persistence**: `SqliteSessionService` for SQLite-backed state (no Google Cloud)
- **Agent State Management**: Pydantic-based `BaseAgentState` for structured state
- **Event-Driven Architecture**: Event streaming with replay capability
- **Branch Tracking**: Isolated conversation histories for multi-path exploration

#### 2. **Resumability & Fault Tolerance**
- **Resumability Config**: `ResumabilityConfig` enables pause/resume on long operations
- **Event Replay**: Resume from last checkpoint using event history
- **LLM Call Limits**: `max_llm_calls` prevents runaway loops (default: 500)
- **Idempotent Operations**: Required for reliable resumption

#### 3. **Artifact Management**
- **Filesystem Backend**: `FileArtifactService` with automatic versioning
- **Version Control**: Monotonic version IDs for all artifacts
- **Session/User Scoping**: Isolated artifact namespaces
- **Metadata Support**: Custom metadata per version

#### 4. **Memory & Context Management**
- **Session Memory**: Cross-conversation context via `InMemoryMemoryService`
- **Event Compaction**: `EventsCompactionConfig` with sliding window for long sessions
- **Context Caching**: Token optimization for repeated context (Gemini)

#### 5. **Multi-Agent Orchestration**
- **SequentialAgent**: Staged pipeline execution (ingestion → extraction → storage)
- **ParallelAgent**: Concurrent retrieval from multiple sources
- **LoopAgent**: Iterative refinement (gleaning, summarization)
- **Transfer Pattern**: Dynamic agent delegation

#### 6. **Open-Source Stack (No Google Cloud)**
- **Session Storage**: SQLite (`SqliteSessionService` + `aiosqlite`)
- **Artifact Storage**: Local filesystem (`FileArtifactService`)
- **Memory**: In-memory service (`InMemoryMemoryService`)
- **LLM Integration**: Any provider via `LiteLlm` (OpenAI, Anthropic, local models)
- **Code Execution**: Local sandbox (`BuiltInCodeExecutor`)

#### 7. **Observability & Monitoring**
- **Plugin System**: Global callbacks for logging, error handling, analytics
- **LoggingPlugin**: Comprehensive event logging
- **OpenTelemetry**: Distributed tracing support
- **Agent Callbacks**: Per-agent hooks for monitoring

#### 8. **Production Tooling**
- **Retry Logic**: Built into LLM and tool execution
- **Rate Limiting**: Configurable for API calls
- **Streaming Support**: SSE and bidirectional streaming
- **Error Recovery**: `ReflectRetryToolPlugin` for automatic retry with reflection

---

### ADK Implementation Patterns for HyperGraphRAG

#### Agent Structure Template
```python
from google.adk import LlmAgent, SequentialAgent, ParallelAgent
from google.adk.artifacts import FileArtifactService
from google.adk.sessions import SqliteSessionService
from google.adk.core import BaseAgentState, ResumabilityConfig
from pydantic import BaseModel, Field

class MyAgentState(BaseAgentState):
    # Agent-specific state fields with persistence
    processed_chunks: List[str] = Field(default_factory=list)
    current_phase: str = "init"
    retry_count: int = 0

class MyAgentInput(BaseModel):
    # Input schema
    content: str
    metadata: Dict[str, Any] = {}

class MyAgentOutput(BaseModel):
    # Output schema
    result: str
    artifacts: List[str] = []

# LLM-based agent with tools
my_agent = LlmAgent(
    name="my_agent",
    model="gpt-4o-mini",  # Or any LiteLlm-supported model
    instruction="You are an agent that...",
    state_type=MyAgentState,
    tools=[tool1, tool2],  # Custom tools
    resumable=True,  # Enable resumability
)
```

#### Hierarchical Agent Composition
```python
from google.adk import SequentialAgent, ParallelAgent, LlmAgent, FunctionTool

# Construction Pipeline (Sequential)
construction_pipeline = SequentialAgent(
    name="hypergraph_construction",
    agents=[
        document_ingestion_agent,  # Chunking
        extraction_agent,          # Entity extraction (with gleaning loop)
        graph_builder_agent,       # Graph construction
        ParallelAgent(             # Parallel storage operations
            name="storage_parallel",
            agents=[embedding_agent, storage_coordinator_agent]
        )
    ],
    state_type=ConstructionState,
    resumable=True,
)

# Retrieval Pipeline (Parallel → Sequential)
retrieval_pipeline = SequentialAgent(
    name="hypergraph_retrieval",
    agents=[
        query_planner_agent,
        ParallelAgent(
            name="multi_source_retrieval",
            agents=[
                entity_retrieval_agent,
                relationship_retrieval_agent,
                text_retrieval_agent,
            ]
        ),
        context_assembly_agent,
        response_generator_agent,
    ],
    state_type=RetrievalState,
    resumable=True,
)
```

#### Storage Integration with Tools
```python
from google.adk import FunctionTool
from hypergraphrag.base import BaseGraphStorage, BaseVectorStorage, BaseKVStorage

# Wrap storage operations as ADK tools
class StorageTools:
    def __init__(
        self,
        graph_storage: BaseGraphStorage,
        vector_storage: BaseVectorStorage,
        kv_storage: BaseKVStorage
    ):
        self.graph = graph_storage
        self.vector = vector_storage
        self.kv = kv_storage

    async def write_graph(self, nodes: List[dict], edges: List[dict]) -> dict:
        """Write nodes and edges to graph storage."""
        await self.graph.upsert_nodes(nodes)
        await self.graph.upsert_edges(edges)
        return {"status": "success", "nodes": len(nodes), "edges": len(edges)}

    async def search_vectors(self, query_embedding: List[float], top_k: int) -> List[dict]:
        """Semantic search in vector storage."""
        return await self.vector.query(query_embedding, top_k=top_k)

    async def get_kv(self, key: str) -> dict:
        """Retrieve from key-value storage."""
        return await self.kv.get(key)

# Create ADK FunctionTools from storage methods
storage_tools = StorageTools(graph_storage, vector_storage, kv_storage)
write_graph_tool = FunctionTool(storage_tools.write_graph)
search_vectors_tool = FunctionTool(storage_tools.search_vectors)
get_kv_tool = FunctionTool(storage_tools.get_kv)
```

#### Session & Artifact Configuration
```python
from google.adk import Application
from google.adk.sessions import SqliteSessionService
from google.adk.artifacts import FileArtifactService
from google.adk.core import ResumabilityConfig, EventsCompactionConfig

# Configure production-ready app (no Google Cloud)
app = Application(
    name="hypergraph_rag",
    agent=construction_pipeline,  # or retrieval_pipeline

    # SQLite-backed sessions for persistence
    session_service=SqliteSessionService(
        db_path="./data/sessions.db",
        compaction_config=EventsCompactionConfig(
            enabled=True,
            sliding_window_size=100,  # Keep last 100 events
            overlap_size=10,           # 10 event overlap for context
        )
    ),

    # Filesystem artifacts with versioning
    artifact_service=FileArtifactService(
        base_path="./data/artifacts"
    ),

    # Enable resumability for long-running operations
    resumability_config=ResumabilityConfig(is_resumable=True),

    # LLM call limits for safety
    max_llm_calls=500,

    # Streaming support
    streaming_mode="SSE",  # or "BIDI" or "NONE"
)
```

#### Plugin System for Observability
```python
from google.adk.plugins import LoggingPlugin, ReflectRetryToolPlugin
from google.adk.core import BasePlugin

# Custom monitoring plugin
class HypergraphMonitoringPlugin(BasePlugin):
    async def on_agent_start(self, agent_name: str, state: dict):
        # Track agent execution start
        print(f"Agent {agent_name} started with state: {state}")

    async def after_tool_callback(self, tool_name: str, result: Any):
        # Track tool execution metrics
        if "graph" in tool_name:
            print(f"Graph operation: {result}")

# Add plugins to application
app.add_plugin(LoggingPlugin(log_level="INFO"))
app.add_plugin(ReflectRetryToolPlugin(max_retries=3))
app.add_plugin(HypergraphMonitoringPlugin())
```

#### Iterative Gleaning with LoopAgent
```python
from google.adk import LoopAgent, LlmAgent, FunctionTool

# Multi-round entity extraction gleaning
gleaning_agent = LoopAgent(
    name="entity_gleaning",
    agent=LlmAgent(
        name="entity_extractor",
        model="gpt-4o-mini",
        instruction="Extract entities and relationships from text...",
        tools=[parse_extraction_tool, merge_entities_tool],
    ),
    max_iterations=2,  # entity_extract_max_gleaning
    state_type=GleaningState,
    resumable=True,
)
```

#### Artifact Versioning for Graph Snapshots
```python
# Store graph snapshots as versioned artifacts
async def save_graph_snapshot(session_id: str, graph_data: dict):
    artifact_uri = await app.artifact_service.create_artifact(
        session_id=session_id,
        artifact_name="graph_snapshot",
        content=json.dumps(graph_data).encode(),
        metadata={"nodes": len(graph_data["nodes"]), "edges": len(graph_data["edges"])}
    )
    # Returns: hypergraph_rag/sessions/{session_id}/artifacts/graph_snapshot/v1
    return artifact_uri

# Load specific version
async def load_graph_snapshot(session_id: str, version: int = None):
    artifact_data = await app.artifact_service.get_artifact(
        session_id=session_id,
        artifact_name="graph_snapshot",
        version=version  # None = latest
    )
    return json.loads(artifact_data.content.decode())
```

#### State Management for Long-Running Construction
```python
from google.adk.core import BaseAgentState
from typing import List, Dict, Optional

class ConstructionState(BaseAgentState):
    """Persistent state for hypergraph construction."""
    session_id: str
    document_ids: List[str] = Field(default_factory=list)
    processed_chunks: int = 0
    total_chunks: int = 0
    extracted_entities: int = 0
    constructed_hyperedges: int = 0
    current_phase: str = "init"  # init, chunking, extraction, building, embedding
    errors: List[Dict[str, str]] = Field(default_factory=list)

    # Resumability tracking
    last_checkpoint: Optional[str] = None
    checkpoint_timestamp: Optional[float] = None

class RetrievalState(BaseAgentState):
    """Persistent state for hypergraph retrieval."""
    query_id: str
    query_text: str
    mode: str = "hybrid"
    retrieved_entities: List[str] = Field(default_factory=list)
    retrieved_relationships: List[str] = Field(default_factory=list)
    context_tokens: int = 0
    cache_hit: bool = False
```

---

## Preserving Original Functionality

### Critical Features to Maintain

1. **Multi-Round Gleaning**
   - Keep `entity_extract_max_gleaning` parameter (default: 2)
   - Implement iterative refinement in ExtractionAgent
   - Use same LLM prompts for consistency

2. **Chunking Parameters**
   - Token size: 1200 (configurable)
   - Overlap: 100 tokens (configurable)
   - Use tiktoken for accurate counting

3. **Retrieval Modes**
   - Local: Entity-centric retrieval
   - Global: Hyperedge-centric retrieval
   - Hybrid: Combine both (default)

4. **Caching**
   - Embedding cache (deduplication)
   - Query result cache
   - LLM response cache

5. **Rate Limiting**
   - Async semaphore for concurrent LLM calls
   - Exponential backoff with tenacity
   - Batch size limits for embedding

6. **Storage Backends**
   - Support all existing backends:
     - KV: JSON, MongoDB, Oracle
     - Vector: NanoVectorDB, Milvus, Chroma
     - Graph: NetworkX, Neo4j, TiDB
   - Keep abstraction layer (Base classes)

7. **Entity Merging**
   - Weight aggregation (sum)
   - Description concatenation
   - Summarization if exceeding token limit

8. **Token Budget Enforcement**
   - Per-context-type limits
   - Global context limit
   - Truncation strategies

---

## Implementation Priority

### Phase 1: Core Construction (Week 1-2)
**Goal:** Agents can ingest documents and build hypergraphs

1. **DocumentIngestionAgent**
   - Implement chunking
   - KV storage integration
   - Test with sample documents

2. **ExtractionAgent**
   - LLM-based entity extraction
   - Multi-round gleaning
   - Parallel chunk processing

3. **GraphBuilderAgent**
   - Entity merging
   - Hyperedge construction
   - Graph storage integration

4. **EmbeddingAgent**
   - Batch embedding
   - Cache management
   - Vector storage integration

5. **StorageCoordinatorAgent**
   - Multi-storage orchestration
   - Transaction coordination

**Deliverable:** End-to-end construction pipeline working

---

### Phase 2: Core Retrieval (Week 3-4)
**Goal:** Agents can query hypergraphs and generate responses

1. **QueryPlannerAgent**
   - Keyword extraction
   - Mode determination
   - Query caching

2. **EntityRetrievalAgent**
   - Vector search
   - Graph traversal
   - Neighbor fetching

3. **RelationshipRetrievalAgent**
   - Hyperedge search
   - Entity connection

4. **TextRetrievalAgent**
   - Chunk tracing
   - Relevance ranking

5. **ContextAssemblyAgent**
   - CSV formatting
   - Token limit enforcement
   - Mode-specific assembly

6. **ResponseGeneratorAgent**
   - Prompt building
   - LLM integration
   - Response caching

**Deliverable:** End-to-end retrieval pipeline working

---

### Phase 3: Optimization & Services (Week 5-6)
**Goal:** Production-ready with performance optimization

1. **LLMServiceAgent**
   - Connection pooling
   - Streaming support
   - Advanced retry logic

2. **CacheAgent**
   - Persistent cache
   - Invalidation strategies
   - Analytics

3. **Performance Tuning**
   - Parallel execution optimization
   - Batch size tuning
   - Memory management

4. **Monitoring & Logging**
   - Agent health checks
   - Performance metrics
   - Error tracking

5. **Testing & Validation**
   - Unit tests for each agent
   - Integration tests for flows
   - Validation against original HyperGraphRAG

**Deliverable:** Production-ready multi-agent system

---

## Configuration Management

### Production Configuration Schema
```python
from pydantic import BaseModel, Field
from typing import Optional

class HyperGraphRAGConfig(BaseModel):
    # === Construction Settings ===
    chunk_token_size: int = 1200
    chunk_overlap: int = 100
    entity_extract_max_gleaning: int = 2

    # === Retrieval Settings ===
    top_k_entities: int = 20
    top_k_relationships: int = 20
    top_k_texts: int = 10
    context_max_tokens: int = 4000

    # === LLM Settings ===
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 100

    # === Storage Backends (Open Source) ===
    kv_storage_type: str = "JsonKVStorage"
    vector_storage_type: str = "NanoVectorDBStorage"
    graph_storage_type: str = "NetworkXStorage"

    # === Caching ===
    enable_embedding_cache: bool = True
    enable_query_cache: bool = True
    enable_llm_cache: bool = True

    # === Concurrency ===
    max_concurrent_llm_calls: int = 16
    max_concurrent_embeddings: int = 64

    # === ADK Production Features ===
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
```

### Application Initialization with Production Features
```python
from google.adk import Application, LlmAgent, SequentialAgent, ParallelAgent
from google.adk.sessions import SqliteSessionService
from google.adk.artifacts import FileArtifactService
from google.adk.core import ResumabilityConfig, EventsCompactionConfig
from google.adk.plugins import LoggingPlugin, ReflectRetryToolPlugin

def create_hypergraph_app(config: HyperGraphRAGConfig) -> Application:
    """Create production-ready HyperGraphRAG application with ADK features."""

    # Build agent hierarchy
    construction_pipeline = build_construction_pipeline(config)
    retrieval_pipeline = build_retrieval_pipeline(config)

    # Create main application
    app = Application(
        name="hypergraph_rag",
        agent=construction_pipeline,  # Can switch to retrieval_pipeline

        # SQLite sessions (no Google Cloud)
        session_service=SqliteSessionService(
            db_path=config.session_db_path,
            compaction_config=EventsCompactionConfig(
                enabled=config.event_compaction_enabled,
                sliding_window_size=config.event_sliding_window_size,
                overlap_size=config.event_overlap_size,
            )
        ) if config.enable_session_persistence else None,

        # Filesystem artifacts with versioning
        artifact_service=FileArtifactService(
            base_path=config.artifact_base_path
        ) if config.enable_artifact_versioning else None,

        # Resumability for long-running operations
        resumability_config=ResumabilityConfig(
            is_resumable=config.enable_resumability
        ),

        # Safety limits
        max_llm_calls=config.max_llm_calls_per_session,

        # Streaming
        streaming_mode=config.streaming_mode,
    )

    # Add production plugins
    if config.enable_logging_plugin:
        app.add_plugin(LoggingPlugin(log_level=config.log_level))

    if config.enable_error_recovery:
        app.add_plugin(ReflectRetryToolPlugin(max_retries=config.max_tool_retries))

    return app
```

---

## Testing Strategy

### Unit Tests
- Each agent with mock inputs/outputs
- Storage abstraction tests
- LLM call mocking

### Integration Tests
- Construction flow end-to-end
- Retrieval flow end-to-end
- Agent communication

### Validation Tests
- Compare output with original HyperGraphRAG
- Performance benchmarks
- Accuracy metrics (if ground truth available)

---

## Project Structure
```
agentic-hypergraph/
├── agents/
│   ├── __init__.py
│   ├── construction/
│   │   ├── __init__.py
│   │   ├── document_ingestion.py
│   │   ├── extraction.py
│   │   ├── graph_builder.py
│   │   ├── embedding.py
│   │   └── storage_coordinator.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── query_planner.py
│   │   ├── entity_retrieval.py
│   │   ├── relationship_retrieval.py
│   │   ├── text_retrieval.py
│   │   ├── context_assembly.py
│   │   └── response_generator.py
│   └── services/
│       ├── __init__.py
│       ├── llm_service.py
│       └── cache.py
├── storage/
│   ├── __init__.py
│   ├── base.py (from HyperGraphRAG)
│   ├── kv_storage.py
│   ├── vector_storage.py
│   └── graph_storage.py
├── schemas/
│   ├── __init__.py
│   ├── construction.py (Input/Output models)
│   ├── retrieval.py (Input/Output models)
│   └── common.py (Shared models)
├── config/
│   ├── __init__.py
│   └── settings.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── validation/
├── examples/
│   ├── construction_example.py
│   └── retrieval_example.py
└── README.md
```

---

## Dependencies

### Core
- `google-adk` - Agent framework
- `pydantic` - Schema validation
- `asyncio` - Async operations

### From HyperGraphRAG
- `openai` - LLM and embeddings
- `tiktoken` - Token counting
- `tenacity` - Retry logic
- `networkx` - Graph operations
- `nano-vectordb` - Vector storage

### Optional Backends
- `neo4j` - Neo4j graph storage
- `pymongo` - MongoDB storage
- `pymilvus` - Milvus vector DB
- `chromadb` - Chroma vector DB

---

## Next Steps

1. **Review this plan** - Validate architecture decisions
2. **Set up project structure** - Create directories and base files
3. **Implement Phase 1** - Start with DocumentIngestionAgent
4. **Iterate and test** - Build incrementally with testing

---

**Document Version:** 1.0
**Last Updated:** 2025-11-15
**Maintained by:** Development Team
