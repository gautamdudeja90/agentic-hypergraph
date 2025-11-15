"""Complete Construction Example - Demonstrates the full hypergraph construction pipeline.

This example shows how to:
1. Set up storage backends (KV, Vector, Graph)
2. Initialize LLM and embedding services (OpenAI-compatible)
3. Create the construction pipeline
4. Process documents to build a hypergraph
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our components
from config.settings import HyperGraphRAGConfig
from storage.implementations import create_all_storage
from agents.services.llm_service import create_llm_service, create_embedding_service
from agents.construction.pipeline import HypergraphConstructionPipeline
from schemas.construction import DocumentInput


async def main():
    """Run the complete construction example."""

    print("="*80)
    print("  HYPERGRAPH CONSTRUCTION PIPELINE - Example")
    print("="*80)

    # 1. Configuration
    print("\n1. Loading configuration...")
    config = HyperGraphRAGConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        llm_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        chunk_token_size=1200,
        chunk_overlap=100,
        entity_extract_max_gleaning=2,
    )
    print(f"   ✓ Using LLM: {config.llm_model}")
    print(f"   ✓ Using embeddings: {config.embedding_model}")

    # 2. Initialize services
    print("\n2. Initializing LLM and embedding services...")
    llm_service = create_llm_service(
        api_key=config.openai_api_key,
        model=config.llm_model,
    )
    embedding_service = create_embedding_service(
        api_key=config.openai_api_key,
        model=config.embedding_model,
        dimensions=config.embedding_dimensions,
    )
    print("   ✓ Services initialized (OpenAI-compatible endpoints)")

    # 3. Initialize storage backends
    print("\n3. Initializing storage backends...")
    working_dir = "./data/example_run"
    kv_storage, vector_storage, graph_storage = create_all_storage(
        namespace="example",
        working_dir=working_dir,
        embedding_func=embedding_service,
        embedding_batch_num=config.embedding_batch_size,
    )
    print(f"   ✓ Storage initialized at: {working_dir}")
    print("   ✓ KV: JsonKVStorage")
    print("   ✓ Vector: NanoVectorDB")
    print("   ✓ Graph: NetworkX")

    # 4. Create construction pipeline
    print("\n4. Creating construction pipeline...")
    pipeline = HypergraphConstructionPipeline(
        kv_storage=kv_storage,
        vector_storage=vector_storage,
        graph_storage=graph_storage,
        llm_service=llm_service,
        embedding_service=embedding_service,
        chunk_token_size=config.chunk_token_size,
        chunk_overlap=config.chunk_overlap,
        max_gleaning_rounds=config.entity_extract_max_gleaning,
    )
    print("   ✓ Pipeline ready")

    # 5. Prepare sample documents
    print("\n5. Preparing sample documents...")
    documents = [
        DocumentInput(
            content="""
            Artificial intelligence (AI) is transforming healthcare in unprecedented ways.
            Machine learning algorithms can now detect diseases from medical images with accuracy
            comparable to human experts. Deep learning models analyze patient data to predict
            health outcomes and recommend personalized treatment plans. Natural language processing
            helps extract insights from electronic health records. AI-powered robots assist in
            surgeries with precision beyond human capability. However, challenges remain in areas
            like data privacy, algorithmic bias, and the need for regulatory frameworks to ensure
            safe deployment of AI systems in clinical settings.
            """,
            metadata={"source": "AI in Healthcare article", "category": "technology"}
        ),
        DocumentInput(
            content="""
            Climate change is one of the most pressing challenges facing humanity. Rising global
            temperatures are causing melting ice caps, rising sea levels, and more frequent extreme
            weather events. The primary driver is greenhouse gas emissions from fossil fuel combustion,
            deforestation, and industrial processes. Scientists warn that without immediate action to
            reduce emissions, the consequences could be catastrophic. Renewable energy technologies
            like solar and wind power offer promising solutions. International cooperation through
            agreements like the Paris Accord aims to limit global warming to 1.5 degrees Celsius.
            Individual actions, corporate responsibility, and government policies all play crucial
            roles in addressing this global crisis.
            """,
            metadata={"source": "Climate Science report", "category": "environment"}
        ),
    ]
    print(f"   ✓ Prepared {len(documents)} documents")

    # 6. Process documents
    print("\n6. Processing documents through pipeline...")
    print("   (This will use LLM for extraction, may take a minute...)\n")

    results = await pipeline.process_documents(documents)

    # 7. Summary
    print("\n" + "="*80)
    print("  PIPELINE SUMMARY")
    print("="*80)

    total_chunks = sum(r["chunks"] for r in results)
    total_entities = sum(r["entities"] for r in results)
    total_hyperedges = sum(r["hyperedges"] for r in results)
    total_nodes = sum(r["nodes"] for r in results)
    total_edges = sum(r["edges"] for r in results)

    print(f"\nProcessed {len(documents)} documents:")
    print(f"  • Total chunks: {total_chunks}")
    print(f"  • Total entities extracted: {total_entities}")
    print(f"  • Total hyperedges extracted: {total_hyperedges}")
    print(f"  • Total graph nodes: {total_nodes}")
    print(f"  • Total graph edges: {total_edges}")

    print(f"\nHypergraph stored at: {working_dir}")
    print("\nNext steps:")
    print("  1. Implement retrieval pipeline to query this hypergraph")
    print("  2. Use QueryPlannerAgent to process queries")
    print("  3. Retrieve relevant entities, relationships, and text")
    print("  4. Generate responses using assembled context")

    print("\n" + "="*80)
    print("  Example complete!")
    print("="*80)


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
