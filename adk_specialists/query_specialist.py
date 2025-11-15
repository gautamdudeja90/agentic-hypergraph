"""Query Specialist - ADK LlmAgent for querying the hypergraph."""

from google.adk.agents import LlmAgent
from google.adk.models.litellm_model import LiteLlmModel
from adk_tools.hypergraph_tools import query_hypergraph_tool, list_stats_tool

query_model = LiteLlmModel(
    model_name="gpt-4o-mini",
    api_key=None,
)

agent = LlmAgent(
    name="QuerySpecialist",
    model=query_model,
    description="Queries the hypergraph to find relevant entities, relationships, and context to answer user questions.",
    instruction=(
        "You are a hypergraph query specialist. Your job is to:\n\n"
        "1. Take user questions and query the hypergraph knowledge base\n"
        "2. Use `query_hypergraph` tool to search for relevant:\n"
        "   - Entities (people, places, concepts)\n"
        "   - Hyperedges (relationships connecting multiple entities)\n"
        "   - Source text chunks\n\n"
        "3. Synthesize the retrieved information into a clear answer\n"
        "4. Cite the sources from the hypergraph\n\n"
        "The hypergraph uses semantic search to find the most relevant information.\n"
        "You can use different modes:\n"
        "- 'local' for entity-centric retrieval\n"
        "- 'global' for relationship-centric retrieval  \n"
        "- 'hybrid' for balanced retrieval (default)"
    ),
    tools=[query_hypergraph_tool, list_stats_tool]
)
