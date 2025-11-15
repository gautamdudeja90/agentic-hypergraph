"""Document Ingestion Specialist - ADK LlmAgent for ingesting documents."""

from google.adk.agents import LlmAgent
from google.adk.models.litellm_model import LiteLlmModel
from adk_tools.hypergraph_tools import ingest_document_tool, list_stats_tool

# Use OpenAI-compatible model (no Gemini)
ingestion_model = LiteLlmModel(
    model_name="gpt-4o-mini",
    api_key=None,  # Will be set from config
)

agent = LlmAgent(
    name="DocumentIngestionSpecialist",
    model=ingestion_model,
    description="Ingests documents and prepares them for hypergraph construction by chunking text into manageable pieces.",
    instruction=(
        "You are a document ingestion specialist. Your job is to:\n"
        "1. Accept document text from users\n"
        "2. Use the `ingest_document` tool to chunk and store the document\n"
        "3. Report back with the document ID and chunk statistics\n\n"
        "When a user provides a document, immediately ingest it and confirm success.\n"
        "Provide clear feedback about what was processed."
    ),
    tools=[ingest_document_tool, list_stats_tool]
)
