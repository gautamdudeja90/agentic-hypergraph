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

## Google ADK Implementation Guidelines

### Agent Structure Template
```python
from google.adk import agent, AgentState, Message
from pydantic import BaseModel

class MyAgentState(AgentState):
    # Agent-specific state fields
    pass

class MyAgentInput(BaseModel):
    # Input schema
    pass

class MyAgentOutput(BaseModel):
    # Output schema
    pass

@agent(
    name="my_agent",
    state_type=MyAgentState,
    input_type=MyAgentInput,
    output_type=MyAgentOutput
)
class MyAgent:
    def __init__(self, config: dict):
        # Initialize agent
        pass

    async def process(self, input: MyAgentInput, state: MyAgentState) -> MyAgentOutput:
        # Main processing logic
        pass

    async def on_message(self, message: Message, state: MyAgentState):
        # Handle incoming messages from other agents
        pass

    # Tool functions
    async def tool_function(self, params):
        # Call external services (storage, LLM, etc.)
        pass
```

### Storage Integration
```python
# Keep existing storage abstractions
from hypergraphrag.base import BaseGraphStorage, BaseVectorStorage, BaseKVStorage

# Wrap as tools for StorageCoordinatorAgent
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

    async def write_graph(self, nodes, edges):
        # Graph write operations
        pass

    async def search_vectors(self, query_embedding, top_k):
        # Vector search
        pass

    async def get_kv(self, key):
        # KV retrieval
        pass
```

### Message Passing
```python
# Agent-to-agent communication
from google.adk import send_message

# From DocumentIngestionAgent to ExtractionAgent
await send_message(
    to_agent="extraction_agent",
    message_type="chunked_document",
    payload=chunked_doc.dict()
)

# From ExtractionAgent to GraphBuilderAgent
await send_message(
    to_agent="graph_builder_agent",
    message_type="extraction_result",
    payload=extraction_result.dict()
)
```

### Event-Driven Coordination
```python
from google.adk import event_handler

@event_handler("extraction_complete")
async def on_extraction_complete(event, state):
    # Trigger graph building and embedding in parallel
    await send_message(to_agent="graph_builder_agent", ...)
    await send_message(to_agent="embedding_agent", ...)
```

### Parallel Execution
```python
import asyncio

# Parallel chunk extraction
extraction_tasks = [
    extraction_agent.process(chunk)
    for chunk in chunks
]
results = await asyncio.gather(*extraction_tasks)

# Parallel retrieval
retrieval_tasks = [
    entity_agent.process(query),
    relationship_agent.process(query),
    text_agent.process(query)
]
entity_result, rel_result, text_result = await asyncio.gather(*retrieval_tasks)
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

### Agent Configuration Schema
```python
class HyperGraphRAGConfig(BaseModel):
    # Construction settings
    chunk_token_size: int = 1200
    chunk_overlap: int = 100
    entity_extract_max_gleaning: int = 2

    # Retrieval settings
    top_k_entities: int = 20
    top_k_relationships: int = 20
    top_k_texts: int = 10
    context_max_tokens: int = 4000

    # LLM settings
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 100

    # Storage backends
    kv_storage_type: str = "JsonKVStorage"
    vector_storage_type: str = "NanoVectorDBStorage"
    graph_storage_type: str = "NetworkXStorage"

    # Caching
    enable_embedding_cache: bool = True
    enable_query_cache: bool = True
    enable_llm_cache: bool = True

    # Concurrency
    max_concurrent_llm_calls: int = 16
    max_concurrent_embeddings: int = 64
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
