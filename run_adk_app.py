"""Google ADK Multi-Agent HyperGraph RAG Application.

This is the main entry point for running the ADK-based multi-agent system.
Uses Google ADK framework with:
- LlmAgent for specialist agents
- AgentTool for agent delegation
- FunctionTool for operations
- Runner for execution
- 100% OpenAI-compatible (no Google Cloud/Gemini)
"""

import os
import asyncio
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Google ADK
from google.adk import Runner

# Our configuration
from config.settings import HyperGraphRAGConfig

# Initialize tools/storage
from adk_tools.hypergraph_tools import initialize_storage

# Import coordinator
from adk_specialists.coordinator import root_agent


async def main():
    """Run the ADK multi-agent application."""

    print("="*80)
    print("  Google ADK - Agentic HyperGraph RAG System")
    print("  Multi-Agent Architecture with Specialist Delegation")
    print("="*80)

    # Configure
    print("\n[Setup] Loading configuration...")
    config = HyperGraphRAGConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        llm_model="gpt-4o-mini",
    )

    # Initialize storage and tools
    print("[Setup] Initializing storage and services...")
    initialize_storage(config)

    # Create runner with coordinator
    print("[Setup] Creating ADK Runner with coordinator agent...")
    runner = Runner(agent=root_agent)

    print("\n✅ System ready!")
    print("\n" + "="*80)
    print("  DEMO: Document Processing + Querying")
    print("="*80)

    # === Example 1: Ingest and Process a Document ===
    print("\n📄 Example 1: Processing a Document")
    print("-" * 80)

    document = """
    Machine learning is revolutionizing artificial intelligence. Deep learning models,
    particularly neural networks, have achieved remarkable success in computer vision
    and natural language processing. TensorFlow and PyTorch are the leading frameworks
    for building these models. Companies like Google, Meta, and OpenAI are pushing
    the boundaries of what's possible with AI.
    """

    user_message = f"I have a document about AI and machine learning. Please ingest it and extract all the entities and relationships:\n\n{document}"

    print(f"\n👤 User: {user_message[:150]}...\n")
    print("🤖 Coordinator: Processing your request...\n")

    response = await runner.run(user_message=user_message)
    print(f"📋 Response:\n{response}\n")

    # === Example 2: Query the Hypergraph ===
    print("\n" + "="*80)
    print("📊 Example 2: Querying the Hypergraph")
    print("-" * 80)

    query = "What are the main AI frameworks and who is using them?"

    print(f"\n👤 User: {query}\n")
    print("🤖 Coordinator: Searching hypergraph...\n")

    response = await runner.run(user_message=query)
    print(f"📋 Response:\n{response}\n")

    # === Example 3: Show Statistics ===
    print("\n" + "="*80)
    print("📈 Example 3: Hypergraph Statistics")
    print("-" * 80)

    stats_query = "Can you show me the current hypergraph statistics?"

    print(f"\n👤 User: {stats_query}\n")
    print("🤖 Coordinator: Fetching statistics...\n")

    response = await runner.run(user_message=stats_query)
    print(f"📋 Response:\n{response}\n")

    # === Summary ===
    print("\n" + "="*80)
    print("  ✅ ADK Multi-Agent System Demo Complete!")
    print("="*80)

    print("\n🎯 What Just Happened:")
    print("1. Coordinator received your document")
    print("2. Delegated to DocumentIngestionSpecialist → chunked the text")
    print("3. Delegated to ExtractionSpecialist → extracted entities & hyperedges")
    print("4. Delegated to QuerySpecialist → searched the hypergraph")
    print("5. All agents worked together seamlessly!")

    print("\n🏗️  Architecture:")
    print("- Coordinator Agent (delegates tasks)")
    print("  ├── DocumentIngestionSpecialist (chunks documents)")
    print("  ├── ExtractionSpecialist (extracts & builds graph)")
    print("  └── QuerySpecialist (searches hypergraph)")

    print("\n💾 Data stored in: ./data/adk_hypergraph/")

    print("\n🚀 Next Steps:")
    print("- Try your own documents and queries")
    print("- Extend with more specialists (summarization, etc.)")
    print("- Add session persistence and artifact versioning")
    print("- Deploy as a production service")

    print("\n" + "="*80)


if __name__ == "__main__":
    asyncio.run(main())
