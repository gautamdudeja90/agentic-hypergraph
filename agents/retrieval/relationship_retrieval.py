"""RelationshipRetrievalAgent - Retrieves relevant hyperedges.

This agent:
1. Performs vector search for similar hyperedges
2. Fetches connected entities
3. Traces back to source text chunks
4. Ranks relationships by relevance
"""

from typing import List

from schemas.retrieval import (
    RelationshipRetrievalInput,
    RelationshipRetrievalResult,
    RetrievedRelationship,
)
from agents.services.llm_service import EmbeddingService
from storage.base import BaseVectorStorage, BaseGraphStorage


class RelationshipRetrievalAgent:
    """Agent for relationship (hyperedge) retrieval."""

    def __init__(
        self,
        vector_storage: BaseVectorStorage,
        graph_storage: BaseGraphStorage,
        embedding_service: EmbeddingService,
    ):
        """Initialize RelationshipRetrievalAgent.

        Args:
            vector_storage: Vector storage backend
            graph_storage: Graph storage backend
            embedding_service: Embedding service
        """
        self.vector_storage = vector_storage
        self.graph_storage = graph_storage
        self.embedding_service = embedding_service

    async def _vector_search_hyperedges(
        self, keywords: List[str], top_k: int
    ) -> List[dict]:
        """Search for hyperedges using vector similarity."""
        query = " ".join(keywords)
        results = await self.vector_storage.query(query, top_k=top_k)

        # Filter for hyperedge type nodes
        hyperedge_results = [
            r for r in results if r.get("entity_type") == "hyperedge"
        ]

        return hyperedge_results[:top_k]

    async def _fetch_connected_entities(self, hyperedge_id: str) -> List[str]:
        """Fetch entities connected to this hyperedge."""
        edges = await self.graph_storage.get_node_edges(hyperedge_id)
        if not edges:
            return []

        # Extract entity IDs from edges
        entity_ids = [target_id for _, target_id in edges]
        return entity_ids

    async def process(
        self, input_data: RelationshipRetrievalInput
    ) -> RelationshipRetrievalResult:
        """Retrieve relevant relationships.

        Args:
            input_data: Relationship retrieval input

        Returns:
            Retrieved relationships with metadata
        """
        # Vector search for hyperedges
        search_results = await self._vector_search_hyperedges(
            input_data.keywords, input_data.top_k
        )

        # Process each hyperedge
        relationships = []
        for result in search_results:
            hyperedge_id = result.get("id") or result.get("__id__")
            if not hyperedge_id:
                continue

            # Fetch connected entities if requested
            connected_entities = []
            if input_data.include_entities:
                connected_entities = await self._fetch_connected_entities(
                    hyperedge_id
                )

            # Get node data
            node_data = await self.graph_storage.get_node(hyperedge_id)
            description = result.get("description", "")
            if node_data:
                description = node_data.get("description", description)

            # Create retrieved relationship
            relationship = RetrievedRelationship(
                hyperedge_id=hyperedge_id,
                description=description,
                connected_entities=connected_entities,
                relevance_score=result.get("distance", 0.0),
                source_chunk_ids=node_data.get("source_chunks", []) if node_data else [],
                weight=node_data.get("weight", 1.0) if node_data else 1.0,
            )
            relationships.append(relationship)

        return RelationshipRetrievalResult(relationships=relationships)
