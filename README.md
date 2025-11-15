# Agentic HyperGraph - Multi-Agent RAG System

A production-ready multi-agent system for building and querying hypergraphs, combining **Google ADK** framework with **HyperGraphRAG** functionality.

## Overview

This project converts HyperGraphRAG into an agentic system using Google's Agent Development Kit (ADK), providing:

- **11 Specialized Agents**: 5 for construction, 6 for retrieval
- **Production Features**: Session persistence, resumability, artifact versioning
- **Open-Source Stack**: No Google Cloud dependencies required
- **Full HyperGraphRAG Compatibility**: Preserves all original functionality

## Architecture

### Construction Pipeline (Document → Hypergraph)
```
Document → DocumentIngestionAgent → ExtractionAgent → GraphBuilderAgent → [EmbeddingAgent + StorageCoordinator] → Hypergraph
```

### Retrieval Pipeline (Query → Response)
```
Query → QueryPlannerAgent → [EntityRetrieval + RelationshipRetrieval + TextRetrieval] → ContextAssembly → ResponseGenerator → Answer
```

## Installation

### Prerequisites
1. Python 3.10+
2. Access to Google ADK repository
3. Access to HyperGraphRAG repository

### Setup

```bash
# Clone this repository
git clone https://github.com/gautamdudeja90/agentic-hypergraph
cd agentic-hypergraph

# Install Google ADK (from local path)
pip install -e /Users/gautamdudeja/repo/adk-python

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your OpenAI API key
```

### Environment Variables

Create a `.env` file:
```bash
OPENAI_API_KEY=your_api_key_here
```

## Project Structure

```
agentic-hypergraph/
├── agents/
│   ├── construction/      # Construction agents
│   ├── retrieval/         # Retrieval agents
│   ├── services/          # Shared services (LLM, cache)
│   └── utils.py           # Agent utilities
├── storage/
│   ├── base.py            # Storage abstractions
│   └── implementations.py # Storage factory functions
├── schemas/
│   ├── common.py          # Shared types
│   ├── construction.py    # Construction schemas
│   └── retrieval.py       # Retrieval schemas
├── config/
│   └── settings.py        # Configuration
├── tests/                 # Test suite
├── examples/              # Usage examples
├── data/                  # Runtime data (sessions, artifacts, storage)
├── agents/AGENT_ARCHITECTURE.md  # Complete architecture documentation
└── README.md

## Quick Start

### 1. Basic Usage

```python
from config.settings import HyperGraphRAGConfig, default_config
from storage.implementations import create_all_storage
# Import agents (to be implemented)

# Configure
config = HyperGraphRAGConfig(
    openai_api_key="your-key",
    enable_session_persistence=True,
    enable_resumability=True,
)

# Initialize storage
kv_storage, vector_storage, graph_storage = create_all_storage(
    namespace="my_project",
    working_dir="./data",
    embedding_func=embedding_function,
)

# TODO: Initialize agents and run pipeline
```

### 2. Construction Example

```python
# Build hypergraph from documents
documents = [
    "Artificial intelligence is transforming healthcare...",
    "Machine learning models require large datasets...",
]

# Process through construction pipeline
# (Full implementation in examples/construction_example.py)
```

### 3. Retrieval Example

```python
# Query the hypergraph
query = "How is AI being used in healthcare?"

# Process through retrieval pipeline
# (Full implementation in examples/retrieval_example.py)
```

## Features

### Production-Ready Features (from Google ADK)

1. **Session Persistence**: SQLite-backed state management
2. **Resumability**: Pause/resume long-running operations
3. **Artifact Versioning**: Automatic versioning of graph snapshots
4. **Event Compaction**: Efficient long-session management
5. **Observability**: Logging plugins, OpenTelemetry support
6. **Error Recovery**: Automatic retry with reflection
7. **Streaming**: SSE and bidirectional streaming support

### HyperGraphRAG Features (Preserved)

1. **Multi-Round Gleaning**: Iterative entity extraction refinement
2. **Hyperedge Construction**: N-ary relationship modeling
3. **Retrieval Modes**: Local (entity-centric), Global (relationship-centric), Hybrid
4. **Token Budget Enforcement**: Precise context management
5. **Entity Merging**: Weight aggregation and deduplication
6. **Caching**: Embedding, query, and LLM response caching

## Configuration

See `config/settings.py` for full configuration options:

```python
class HyperGraphRAGConfig(BaseModel):
    # Construction
    chunk_token_size: int = 1200
    chunk_overlap: int = 100
    entity_extract_max_gleaning: int = 2

    # Retrieval
    top_k_entities: int = 20
    context_max_tokens: int = 4000

    # LLM
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # ADK Features
    enable_session_persistence: bool = True
    enable_resumability: bool = True
    enable_artifact_versioning: bool = True
    # ... see settings.py for all options
```

## Development Status

### Phase 1: Core Construction ✅ (Planned)
- [x] Project structure
- [x] Schemas and configuration
- [x] Storage layer integration
- [ ] DocumentIngestionAgent
- [ ] ExtractionAgent
- [ ] GraphBuilderAgent
- [ ] EmbeddingAgent
- [ ] StorageCoordinatorAgent

### Phase 2: Core Retrieval 📋 (Planned)
- [ ] QueryPlannerAgent
- [ ] EntityRetrievalAgent
- [ ] RelationshipRetrievalAgent
- [ ] TextRetrievalAgent
- [ ] ContextAssemblyAgent
- [ ] ResponseGeneratorAgent

### Phase 3: Production Polish 📋 (Planned)
- [ ] LLMServiceAgent
- [ ] CacheAgent
- [ ] Testing suite
- [ ] Examples and documentation

## Documentation

- **[AGENT_ARCHITECTURE.md](agents/AGENT_ARCHITECTURE.md)**: Complete architecture and implementation plan
- **[CLAUDE.md](CLAUDE.md)**: AI assistant guide for this codebase

## Testing

```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/unit
pytest tests/integration

# With coverage
pytest --cov=agents --cov=storage
```

## Contributing

This project follows the conventions outlined in [CLAUDE.md](CLAUDE.md):

1. Use conventional commits (`feat:`, `fix:`, `docs:`)
2. Write tests for new functionality
3. Update documentation when changing APIs
4. Preserve backward compatibility

## License

[Specify License]

## Acknowledgments

- **Google ADK**: Agent Development Kit framework
- **HyperGraphRAG**: Original hypergraph RAG implementation

## Contact

[Your contact information]

---

**Status**: Under Development
**Last Updated**: 2025-11-15
