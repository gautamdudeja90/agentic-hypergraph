"""Google ADK Application for Agentic HyperGraph RAG.

This integrates our hypergraph agents with Google ADK framework using:
- Agent for LLM-powered agents
- Runner for execution
- FunctionTool for storage operations
- No Google Cloud dependencies (using LiteLLM)
"""

import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any

# Google ADK imports
from google.adk import Agent, Runner
from google.adk.tools.function_tool import FunctionTool
from google.adk.models.litellm_model import LiteLlmModel

from dotenv import load_dotenv
load_dotenv()

# Our imports
from config.settings import HyperGraphRAGConfig
from storage.implementations import create_all_storage
from agents.services.llm_service import create_embedding_service
from schemas.construction import DocumentInput
from agents.construction.document_ingestion import DocumentIngestionAgent
from agents.construction.extraction import ExtractionAgent
from agents.construction.graph_builder import GraphBuilderAgent
from agents.construction.embedding import EmbeddingAgent


# ===== Storage Tools (wrapped as ADK FunctionTools) =====

class HypergraphTools:
    """Tools for hypergraph operations."""

    def __init__(self, config: HyperGraphRAGConfig):
        """Initialize storage and services."""
        self.config = config

        # Initialize embedding service
        self.embedding_service = create_embedding_service(
            api_key=config.openai_api_key,
            model=config.embedding_model,
            dimensions=config.embedding_dimensions,
        )

        # Initialize storage
        self.kv_storage, self.vector_storage, self.graph_storage = create_all_storage(
            namespace="adk_hypergraph",
            working_dir=config.kv_storage_path,
            embedding_func=self.embedding_service,
        )

        # Initialize agents
        from agents.services.llm_service import create_llm_service
        llm_service = create_llm_service(
            api_key=config.openai_api_key,
            model=config.llm_model,
        )

        self.ingestion_agent = DocumentIngestionAgent(
            kv_storage=self.kv_storage,
            chunk_token_size=config.chunk_token_size,
            chunk_overlap=config.chunk_overlap,
        )

        self.extraction_agent = ExtractionAgent(
            llm_service=llm_service,
            max_gleaning_rounds=config.entity_extract_max_gleaning,
        )

        self.graph_builder_agent = GraphBuilderAgent(
            graph_storage=self.graph_storage,
            llm_service=llm_service,
        )

        self.embedding_agent = EmbeddingAgent(
            vector_storage=self.vector_storage,
            embedding_service=self.embedding_service,
        )

    async def ingest_document(self, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Ingest and chunk a document.

        Args:
            content: Document text content
            metadata: Optional metadata

        Returns:
            Dictionary with doc_id and number of chunks
        """
        doc_input = DocumentInput(content=content, metadata=metadata or {})
        result = await self.ingestion_agent.process(doc_input)
        return {
            "doc_id": result.doc_id,
            "num_chunks": len(result.chunks),
            "chunks": [{"id": c.chunk_id, "tokens": c.tokens} for c in result.chunks]
        }

    async def extract_entities(self, doc_id: str) -> Dict[str, Any]:
        """Extract entities and hyperedges from a document's chunks.

        Args:
            doc_id: Document ID to process

        Returns:
            Dictionary with extraction results
        """
        # Get chunks for this doc
        all_keys = await self.kv_storage.all_keys()
        chunk_ids = [k for k in all_keys if k.startswith(doc_id)]
        chunks_data = await self.kv_storage.get_by_ids(chunk_ids)

        # Convert to TextChunk objects
        from schemas.common import TextChunk
        from schemas.construction import ChunkExtractionInput

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

        extraction_input = ChunkExtractionInput(chunks=chunks)
        result = await self.extraction_agent.process(extraction_input)

        return {
            "num_entities": len(result.entities),
            "num_hyperedges": len(result.hyperedges),
            "entities": [{"name": e.name, "type": e.type} for e in result.entities[:10]],
            "hyperedges": [{"description": h.description[:100]} for h in result.hyperedges[:5]]
        }

    async def build_graph(self, doc_id: str) -> Dict[str, Any]:
        """Build hypergraph from extracted entities.

        Args:
            doc_id: Document ID to process

        Returns:
            Dictionary with graph statistics
        """
        # This is simplified - in reality you'd coordinate the full pipeline
        return {
            "status": "Graph building initiated",
            "doc_id": doc_id,
            "message": "Use the full construction pipeline for complete graph building"
        }

    async def query_hypergraph(self, query: str, mode: str = "hybrid", top_k: int = 10) -> Dict[str, Any]:
        """Query the hypergraph and return relevant context.

        Args:
            query: Query text
            mode: Retrieval mode (local/global/hybrid)
            top_k: Number of results

        Returns:
            Dictionary with retrieved context
        """
        # Simplified query - returns vector search results
        results = await self.vector_storage.query(query, top_k=top_k)

        return {
            "query": query,
            "mode": mode,
            "num_results": len(results),
            "results": [
                {
                    "id": r.get("id", "unknown"),
                    "content": r.get("description", "")[:200],
                    "score": r.get("distance", 0.0)
                }
                for r in results[:5]
            ]
        }


# ===== ADK Agent Configuration =====

def create_hypergraph_agent(config: HyperGraphRAGConfig) -> Agent:
    """Create ADK agent for hypergraph operations.

    Returns:
        Configured ADK Agent
    """
    # Initialize tools
    tools_instance = HypergraphTools(config)

    # Wrap methods as ADK FunctionTools
    tools = [
        FunctionTool(tools_instance.ingest_document),
        FunctionTool(tools_instance.extract_entities),
        FunctionTool(tools_instance.build_graph),
        FunctionTool(tools_instance.query_hypergraph),
    ]

    # Create LiteLLM model (OpenAI-compatible, no Gemini)
    model = LiteLlmModel(
        model_name=config.llm_model,
        api_key=config.openai_api_key,
    )

    # Create ADK Agent
    agent = Agent(
        model=model,
        name="hypergraph_agent",
        instruction="""You are a HyperGraph RAG assistant. You can:

1. Ingest documents and create knowledge hypergraphs
2. Extract entities and relationships from text
3. Build hypergraph structures with entities and hyperedges
4. Query the hypergraph to answer questions

Use the available tools to process documents and answer queries based on the hypergraph knowledge base.

When a user provides a document, use ingest_document, then extract_entities.
When a user asks a question, use query_hypergraph to find relevant information.
""",
        tools=tools,
    )

    return agent


# ===== Main Application =====

async def main():
    """Run the ADK-based hypergraph application."""

    print("="*80)
    print("  Google ADK - Agentic HyperGraph RAG")
    print("="*80)

    # Load configuration
    config = HyperGraphRAGConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        llm_model="gpt-4o-mini",
    )

    # Create agent
    print("\n[Setup] Creating ADK agent...")
    agent = create_hypergraph_agent(config)

    # Create runner
    runner = Runner(agent=agent)

    # Example: Ingest a document
    print("\n" + "="*80)
    print("Example 1: Ingesting a Document")
    print("="*80)

    document_text = """
    Artificial intelligence and machine learning are transforming healthcare.
    Deep learning models can detect diseases from medical images with high accuracy.
    Natural language processing helps analyze electronic health records.
    AI-powered systems assist doctors in diagnosis and treatment planning.
    """

    user_message = f"Please ingest this document and extract entities from it:\n\n{document_text}"

    print(f"\nUser: {user_message[:100]}...")
    print("\nAgent:")

    # Run agent
    response = await runner.run(user_message=user_message)
    print(response)

    # Example: Query the hypergraph
    print("\n" + "="*80)
    print("Example 2: Querying the Hypergraph")
    print("="*80)

    query_message = "What are the applications of AI in healthcare?"

    print(f"\nUser: {query_message}")
    print("\nAgent:")

    response = await runner.run(user_message=query_message)
    print(response)

    print("\n" + "="*80)
    print("  ADK Integration Complete!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
