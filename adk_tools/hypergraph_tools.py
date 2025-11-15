"""ADK FunctionTools for HyperGraph operations.

These tools wrap our storage and processing operations as ADK-compatible functions.
"""

from google.adk.tools import FunctionTool
from typing import Dict, Any, List
import asyncio

# Import our existing infrastructure
from config.settings import HyperGraphRAGConfig
from storage.implementations import create_all_storage
from agents.services.llm_service import create_llm_service, create_embedding_service
from agents.construction.document_ingestion import DocumentIngestionAgent
from agents.construction.extraction import ExtractionAgent
from agents.construction.graph_builder import GraphBuilderAgent
from agents.construction.embedding import EmbeddingAgent
from schemas.construction import DocumentInput, ChunkExtractionInput
from schemas.common import TextChunk


# Global state for storage and services
_storage_initialized = False
_kv_storage = None
_vector_storage = None
_graph_storage = None
_embedding_service = None
_llm_service = None
_agents = {}


def initialize_storage(config: HyperGraphRAGConfig):
    """Initialize storage backends and services."""
    global _storage_initialized, _kv_storage, _vector_storage, _graph_storage
    global _embedding_service, _llm_service, _agents

    if _storage_initialized:
        return

    # Initialize services
    _embedding_service = create_embedding_service(
        api_key=config.openai_api_key,
        model=config.embedding_model,
        dimensions=config.embedding_dimensions,
    )

    _llm_service = create_llm_service(
        api_key=config.openai_api_key,
        model=config.llm_model,
    )

    # Initialize storage
    _kv_storage, _vector_storage, _graph_storage = create_all_storage(
        namespace="adk_hypergraph",
        working_dir="./data/adk_hypergraph",
        embedding_func=_embedding_service,
    )

    # Initialize agents
    _agents['ingestion'] = DocumentIngestionAgent(
        kv_storage=_kv_storage,
        chunk_token_size=config.chunk_token_size,
        chunk_overlap=config.chunk_overlap,
    )

    _agents['extraction'] = ExtractionAgent(
        llm_service=_llm_service,
        max_gleaning_rounds=config.entity_extract_max_gleaning,
    )

    _agents['graph_builder'] = GraphBuilderAgent(
        graph_storage=_graph_storage,
        llm_service=_llm_service,
    )

    _agents['embedding'] = EmbeddingAgent(
        vector_storage=_vector_storage,
        embedding_service=_embedding_service,
    )

    _storage_initialized = True


# ===== Tool Functions =====

async def ingest_document_func(content: str, metadata: Dict[str, Any] = None) -> str:
    """Ingest a document and chunk it for processing.

    Args:
        content: The document text content
        metadata: Optional metadata dict

    Returns:
        Summary of ingestion results
    """
    doc_input = DocumentInput(content=content, metadata=metadata or {})
    result = await _agents['ingestion'].process(doc_input)

    return f"✅ Document ingested successfully!\n- Document ID: {result.doc_id}\n- Chunks created: {len(result.chunks)}\n- Total tokens: {sum(c.tokens for c in result.chunks)}"


async def extract_entities_func(doc_id: str) -> str:
    """Extract entities and hyperedges from a document.

    Args:
        doc_id: The document ID to process

    Returns:
        Summary of extraction results
    """
    # Get chunks for this doc
    all_keys = await _kv_storage.all_keys()
    chunk_ids = [k for k in all_keys if k.startswith(doc_id)]
    chunks_data = await _kv_storage.get_by_ids(chunk_ids)

    # Convert to TextChunk objects
    chunks = [
        TextChunk(
            chunk_id=cid,
            content=cdata['content'],
            tokens=cdata['tokens'],
            metadata=cdata,
            doc_id=doc_id
        )
        for cid, cdata in zip(chunk_ids, chunks_data) if cdata
    ]

    if not chunks:
        return f"❌ No chunks found for document {doc_id}"

    extraction_input = ChunkExtractionInput(chunks=chunks)
    result = await _agents['extraction'].process(extraction_input)

    # Also build the graph
    from schemas.construction import GraphConstructionInput
    graph_input = GraphConstructionInput(
        entities=result.entities,
        hyperedges=result.hyperedges,
    )
    graph_result = await _agents['graph_builder'].process(graph_input)

    # Embed the nodes
    await _agents['embedding'].embed_nodes(graph_result.nodes)
    await _agents['embedding'].embed_chunks(chunks)

    entity_names = [e.name for e in result.entities[:5]]
    hyperedge_descs = [h.description[:50] + "..." for h in result.hyperedges[:3]]

    return f"""✅ Extraction complete!
- Entities: {len(result.entities)} (e.g., {', '.join(entity_names)})
- Hyperedges: {len(result.hyperedges)}
- Graph nodes: {len(graph_result.nodes)}
- Example relationships: {hyperedge_descs}
- All entities and hyperedges have been embedded and stored in the hypergraph."""


async def query_hypergraph_func(query: str, mode: str = "hybrid", top_k: int = 10) -> str:
    """Query the hypergraph to find relevant information.

    Args:
        query: The query text
        mode: Retrieval mode (local/global/hybrid)
        top_k: Number of results to return

    Returns:
        Retrieved context from hypergraph
    """
    # Query vector storage
    results = await _vector_storage.query(query, top_k=top_k)

    if not results:
        return "❌ No results found in the hypergraph."

    # Format results
    context_parts = []
    context_parts.append(f"📊 Found {len(results)} relevant items:\n")

    for i, result in enumerate(results[:5], 1):
        item_type = result.get('entity_type', 'unknown')
        description = result.get('description', '')[:150]
        score = result.get('distance', 0.0)

        context_parts.append(f"{i}. [{item_type.upper()}] {description}... (relevance: {score:.3f})")

    return "\n".join(context_parts)


async def list_graph_stats_func() -> str:
    """Get statistics about the current hypergraph.

    Returns:
        Statistics summary
    """
    all_keys = await _kv_storage.all_keys()

    # Count documents and chunks
    doc_ids = set()
    chunk_count = 0
    for key in all_keys:
        if '_chunk_' in key:
            chunk_count += 1
            doc_id = key.split('_chunk_')[0]
            doc_ids.add(doc_id)

    return f"""📈 Hypergraph Statistics:
- Documents ingested: {len(doc_ids)}
- Text chunks: {chunk_count}
- Hypergraph is ready for querying!

Use query_hypergraph to search for information."""


# ===== Create ADK FunctionTools =====

ingest_document_tool = FunctionTool(
    func=ingest_document_func,
    name="ingest_document",
    description="Ingest a document and chunk it for hypergraph construction"
)

extract_entities_tool = FunctionTool(
    func=extract_entities_func,
    name="extract_entities",
    description="Extract entities and hyperedges from an ingested document and build the hypergraph"
)

query_hypergraph_tool = FunctionTool(
    func=query_hypergraph_func,
    name="query_hypergraph",
    description="Query the hypergraph to find relevant entities, relationships, and context"
)

list_stats_tool = FunctionTool(
    func=list_graph_stats_func,
    name="list_graph_stats",
    description="Get statistics about the current hypergraph"
)
