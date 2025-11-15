"""EntityRetrievalAgent - Retrieves relevant entities from the hypergraph.

This agent:
1. Performs vector search for similar entities
2. Fetches entity metadata from graph storage
3. Retrieves one-hop neighbors
4. Ranks entities by relevance and degree
"""

from typing import List
import asyncio

from schemas.retrieval import (
    EntityRetrievalInput,
    EntityRetrievalResult,
    RetrievedEntity,
)
from agents.services.llm_service import EmbeddingService
from storage.base import BaseVectorStorage, BaseGraphStorage


class EntityRetrievalAgent:
    """Agent for entity retrieval."""

    def __init__(
        self,
        vector_storage: BaseVectorStorage,
        graph_storage: BaseGraphStorage,
        embedding_service: EmbeddingService,
    ):
        """Initialize EntityRetrievalAgent.

        Args:
            vector_storage: Vector storage backend
            graph_storage: Graph storage backend
            embedding_service: Embedding service
        """
        self.vector_storage = vector_storage
        self.graph_storage = graph_storage
        self.embedding_service = embedding_service

    async def _vector_search_entities(
        self, keywords: List[str], top_k: int
    ) -> List[dict]:
        """Search for entities using vector similarity."""
        # Combine keywords into query
        query = " ".join(keywords)

        # Query vector storage
        results = await self.vector_storage.query(query, top_k=top_k)

        # Filter for entity type nodes
        entity_results = [
            r for r in results if r.get("entity_type") == "entity"
        ]

        return entity_results[:top_k]

    async def _fetch_entity_metadata(self, entity_id: str) -> dict:
        """Fetch entity metadata from graph storage."""
        node_data = await self.graph_storage.get_node(entity_id)
        if not node_data:
            return {}

        # Get node degree
        degree = await self.graph_storage.node_degree(entity_id)

        # Get edges
        edges = await self.graph_storage.get_node_edges(entity_id)

        return {
            "node_data": node_data,
            "degree": degree,
            "edges": edges or [],
        }

    async def _get_neighbors(self, entity_id: str) -> List[str]:
        """Get one-hop neighbors of an entity."""
        edges = await self.graph_storage.get_node_edges(entity_id)
        if not edges:
            return []

        # Extract neighbor IDs
        neighbors = [target_id for _, target_id in edges]
        return neighbors

    async def process(
        self, input_data: EntityRetrievalInput
    ) -> EntityRetrievalResult:
        """Retrieve relevant entities.

        Args:
            input_data: Entity retrieval input

        Returns:
            Retrieved entities with metadata
        """
        # Vector search
        search_results = await self._vector_search_entities(
            input_data.keywords, input_data.top_k
        )

        # Fetch metadata for each entity
        entities = []
        for result in search_results:
            entity_id = result.get("id") or result.get("__id__")
            if not entity_id:
                continue

            # Fetch metadata
            metadata = await self._fetch_entity_metadata(entity_id)
            node_data = metadata.get("node_data", {})

            # Get neighbors if requested
            neighbors = None
            related_edges = None
            if input_data.include_neighbors:
                neighbors = await self._get_neighbors(entity_id)
                related_edges = [
                    f"{src}-{tgt}" for src, tgt in metadata.get("edges", [])
                ]

            # Create retrieved entity
            entity = RetrievedEntity(
                entity_id=entity_id,
                name=node_data.get("description", "Unknown"),
                type=node_data.get("entity_type", "unknown"),
                description=node_data.get("description", ""),
                relevance_score=result.get("distance", 0.0),
                degree=metadata.get("degree", 0),
                neighbors=neighbors,
                related_edges=related_edges,
            )
            entities.append(entity)

        return EntityRetrievalResult(entities=entities)
