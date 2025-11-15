"""Extraction Specialist - ADK LlmAgent for entity and hyperedge extraction."""

from google.adk.agents import LlmAgent
from google.adk.models.litellm_model import LiteLlmModel
from adk_tools.hypergraph_tools import extract_entities_tool, list_stats_tool

extraction_model = LiteLlmModel(
    model_name="gpt-4o-mini",
    api_key=None,
)

agent = LlmAgent(
    name="ExtractionSpecialist",
    model=extraction_model,
    description="Extracts entities and hyperedges from documents using LLM-based multi-round gleaning, then builds the hypergraph structure.",
    instruction=(
        "You are an entity extraction specialist. Your responsibilities:\n\n"
        "1. Take a document ID and extract entities and relationships from it\n"
        "2. Use the `extract_entities` tool which:\n"
        "   - Performs multi-round gleaning to extract entities comprehensively\n"
        "   - Identifies hyperedges (n-ary relationships between entities)\n"
        "   - Builds the graph structure with nodes and edges\n"
        "   - Embeds everything for semantic search\n\n"
        "3. Report what was extracted clearly\n\n"
        "The extraction process uses advanced LLM techniques to ensure high-quality entity and relationship extraction."
    ),
    tools=[extract_entities_tool, list_stats_tool]
)
