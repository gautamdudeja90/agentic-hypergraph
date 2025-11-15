"""Coordinator Agent - Delegates tasks to specialist agents."""

from google.adk.agents import LlmAgent
from google.adk.models.litellm_model import LiteLlmModel
from google.adk.tools import agent_tool

# Import specialist agents
from adk_specialists.ingestion_specialist import agent as ingestion_agent
from adk_specialists.extraction_specialist import agent as extraction_agent
from adk_specialists.query_specialist import agent as query_agent

# Import tools
from adk_tools.hypergraph_tools import list_stats_tool

coordinator_model = LiteLlmModel(
    model_name="gpt-4o-mini",
    api_key=None,
)

agent = LlmAgent(
    name="HypergraphCoordinator",
    model=coordinator_model,
    description="Master coordinator for the Agentic HyperGraph RAG system. Delegates tasks to specialized agents for document processing and querying.",
    instruction=(
        "You are the HyperGraph RAG Coordinator. You manage a multi-agent system for building and querying knowledge hypergraphs.\n\n"
        "**YOUR SPECIALISTS:**\n"
        "1. **DocumentIngestionSpecialist**: Ingests and chunks documents\n"
        "2. **ExtractionSpecialist**: Extracts entities/hyperedges and builds the graph\n"
        "3. **QuerySpecialist**: Queries the hypergraph to answer questions\n\n"
        "**WORKFLOW:**\n\n"
        "**For Document Processing:**\n"
        "1. User provides a document → Delegate to `DocumentIngestionSpecialist`\n"
        "2. Get document ID from ingestion result\n"
        "3. Delegate to `ExtractionSpecialist` with the document ID\n"
        "4. Report completion with statistics\n\n"
        "**For Queries:**\n"
        "1. User asks a question → Delegate to `QuerySpecialist`\n"
        "2. The specialist will search the hypergraph and synthesize an answer\n\n"
        "**IMPORTANT:**\n"
        "- Always follow the workflow: Ingestion → Extraction for new documents\n"
        "- Use `list_graph_stats` to show current status\n"
        "- Provide clear, helpful responses to users\n"
        "- Explain what each specialist is doing"
    ),
    tools=[
        list_stats_tool,
        agent_tool.AgentTool(agent=ingestion_agent),
        agent_tool.AgentTool(agent=extraction_agent),
        agent_tool.AgentTool(agent=query_agent),
    ]
)

root_agent = agent
