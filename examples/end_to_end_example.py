"""Complete End-to-End Example - Construction + Retrieval

This example demonstrates the full RAG system:
1. Build hypergraph from documents (Construction Pipeline)
2. Query the hypergraph (Retrieval Pipeline)
3. Generate answers using assembled context
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from config.settings import HyperGraphRAGConfig
from storage.implementations import create_all_storage
from agents.services.llm_service import create_llm_service, create_embedding_service
from agents.construction.pipeline import HypergraphConstructionPipeline
from agents.retrieval.pipeline import HypergraphRetrievalPipeline
from schemas.construction import DocumentInput


async def main():
    """Run complete end-to-end example."""

    print("="*80)
    print("  AGENTIC HYPERGRAPH - Complete End-to-End Demo")
    print("  Construction + Retrieval")
    print("="*80)

    # Configuration
    config = HyperGraphRAGConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Initialize services
    print("\n[Setup] Initializing services...")
    llm_service = create_llm_service(
        api_key=config.openai_api_key,
        model=config.llm_model,
    )
    embedding_service = create_embedding_service(
        api_key=config.openai_api_key,
        model=config.embedding_model,
        dimensions=config.embedding_dimensions,
    )

    # Initialize storage
    print("[Setup] Initializing storage...")
    working_dir = "./data/end_to_end_demo"
    kv_storage, vector_storage, graph_storage = create_all_storage(
        namespace="demo",
        working_dir=working_dir,
        embedding_func=embedding_service,
    )

    # === PART 1: CONSTRUCTION ===
    print("\n" + "="*80)
    print("  PART 1: CONSTRUCTION (Building Hypergraph)")
    print("="*80)

    construction_pipeline = HypergraphConstructionPipeline(
        kv_storage=kv_storage,
        vector_storage=vector_storage,
        graph_storage=graph_storage,
        llm_service=llm_service,
        embedding_service=embedding_service,
    )

    # Sample documents
    documents = [
        DocumentInput(
            content="""
            Machine learning is a subset of artificial intelligence that enables systems
            to learn and improve from experience without being explicitly programmed.
            Deep learning, a branch of machine learning, uses neural networks with multiple
            layers to analyze complex patterns in data. Applications include computer vision,
            natural language processing, and speech recognition. Popular frameworks include
            TensorFlow, PyTorch, and scikit-learn.
            """,
            metadata={"topic": "AI/ML"}
        ),
        DocumentInput(
            content="""
            Python is a versatile programming language widely used in data science,
            web development, and automation. Its extensive ecosystem includes libraries
            like NumPy for numerical computing, Pandas for data manipulation, and
            Matplotlib for visualization. Python's readability and simplicity make it
            ideal for beginners and experts alike. The language supports multiple
            programming paradigms including object-oriented and functional programming.
            """,
            metadata={"topic": "Programming"}
        ),
    ]

    print(f"\nProcessing {len(documents)} documents...")
    construction_results = await construction_pipeline.process_documents(documents)

    print("\n[Construction Summary]")
    for i, result in enumerate(construction_results):
        print(f"  Document {i+1}:")
        print(f"    - Chunks: {result['chunks']}")
        print(f"    - Entities: {result['entities']}")
        print(f"    - Hyperedges: {result['hyperedges']}")

    # === PART 2: RETRIEVAL ===
    print("\n" + "="*80)
    print("  PART 2: RETRIEVAL (Querying Hypergraph)")
    print("="*80)

    retrieval_pipeline = HypergraphRetrievalPipeline(
        kv_storage=kv_storage,
        vector_storage=vector_storage,
        graph_storage=graph_storage,
        llm_service=llm_service,
        embedding_service=embedding_service,
    )

    # Sample queries
    queries = [
        "What is machine learning and how does it work?",
        "What are the popular Python libraries for data science?",
        "How are deep learning and neural networks related?",
    ]

    print(f"\nProcessing {len(queries)} queries...\n")

    for i, query in enumerate(queries):
        print(f"\n{'─'*80}")
        print(f"Query {i+1}: {query}")
        print(f"{'─'*80}")

        response = await retrieval_pipeline.query(
            query=query,
            mode="hybrid",
            top_k=15,
        )

        print(f"\n[RESPONSE]")
        print(response.response)
        print(f"\n[Metadata]")
        print(f"  - Query ID: {response.query_id[:16]}...")
        print(f"  - Tokens: {response.tokens_used}")
        print(f"  - Cached: {response.cached}")

    # === SUMMARY ===
    print("\n" + "="*80)
    print("  DEMO COMPLETE!")
    print("="*80)

    total_entities = sum(r['entities'] for r in construction_results)
    total_hyperedges = sum(r['hyperedges'] for r in construction_results)

    print(f"\n[Final Statistics]")
    print(f"  Construction:")
    print(f"    - Documents processed: {len(documents)}")
    print(f"    - Total entities: {total_entities}")
    print(f"    - Total hyperedges: {total_hyperedges}")
    print(f"\n  Retrieval:")
    print(f"    - Queries processed: {len(queries)}")
    print(f"    - Hypergraph queried successfully")

    print(f"\n[Data Location]")
    print(f"  {working_dir}/")
    print(f"    ├── kv_store_demo.json       (documents & chunks)")
    print(f"    ├── vdb_demo.json            (vector embeddings)")
    print(f"    └── graph_demo.graphml       (hypergraph structure)")

    print("\n" + "="*80)
    print("  System is ready for production use!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
