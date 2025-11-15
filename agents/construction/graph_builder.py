"""GraphBuilderAgent - Merges entities and constructs hypergraph.

This agent:
1. Merges duplicate entities across chunks
2. Aggregates entity weights and descriptions
3. Constructs hyperedge nodes in graph
4. Builds edges connecting hyperedges to entities
"""

import sys
from typing import List, Dict
from collections import defaultdict
import hashlib

sys.path.insert(0, "/Users/gautamdudeja/repo/HyperGraphRAG")

from hypergraphrag.base import BaseGraphStorage
from hypergraphrag.utils import compute_mdhash_id
from schemas.common import Entity, Hyperedge, GraphNode, GraphEdge
from schemas.construction import GraphConstructionInput, GraphStructure
from agents.services.llm_service import LLMService


class GraphBuilderAgent:
    """Agent for building hypergraph from extracted entities and hyperedges."""

    def __init__(
        self,
        graph_storage: BaseGraphStorage,
        llm_service: LLMService = None,
        max_description_tokens: int = 500,
    ):
        """Initialize GraphBuilderAgent.

        Args:
            graph_storage: Graph storage backend
            llm_service: LLM service for summarization (optional)
            max_description_tokens: Max tokens for entity descriptions
        """
        self.graph_storage = graph_storage
        self.llm_service = llm_service
        self.max_description_tokens = max_description_tokens

    def _merge_entities(self, entities: List[Entity]) -> Dict[str, Entity]:
        """Merge duplicate entities by name."""
        merged = defaultdict(lambda: {
            "type": "",
            "descriptions": [],
            "weights": [],
            "source_chunks": set(),
        })

        for entity in entities:
            name = entity.name.strip()
            merged[name]["descriptions"].append(entity.description)
            merged[name]["weights"].append(entity.weight)
            merged[name]["source_chunks"].update(entity.source_chunk_ids)
            if not merged[name]["type"]:
                merged[name]["type"] = entity.type

        # Create merged entities
        result = {}
        for name, data in merged.items():
            result[name] = Entity(
                name=name,
                type=data["type"],
                description=" | ".join(data["descriptions"]),  # Concatenate descriptions
                weight=sum(data["weights"]),  # Sum weights
                source_chunk_ids=list(data["source_chunks"]),
            )

        return result

    def _create_graph_nodes(
        self, entities: Dict[str, Entity], hyperedges: List[Hyperedge]
    ) -> List[GraphNode]:
        """Create graph nodes from entities and hyperedges."""
        nodes = []

        # Entity nodes
        for name, entity in entities.items():
            node_id = compute_mdhash_id(name, prefix="ent-")
            node = GraphNode(
                node_id=node_id,
                node_type="entity",
                name=entity.name,
                description=entity.description,
                weight=entity.weight,
                metadata={
                    "entity_type": entity.type,
                    "source_chunks": entity.source_chunk_ids,
                },
            )
            nodes.append(node)

        # Hyperedge nodes
        for idx, hyperedge in enumerate(hyperedges):
            node_id = compute_mdhash_id(
                f"{hyperedge.description}_{idx}", prefix="hedge-"
            )
            node = GraphNode(
                node_id=node_id,
                node_type="hyperedge",
                name=f"Hyperedge_{idx}",
                description=hyperedge.description,
                weight=hyperedge.weight,
                metadata={
                    "relationship": hyperedge.relationship,
                    "connected_entities": hyperedge.entities,
                    "source_chunks": hyperedge.source_chunk_ids,
                },
            )
            nodes.append(node)

        return nodes

    def _create_graph_edges(
        self, entities: Dict[str, Entity], hyperedges: List[Hyperedge]
    ) -> List[GraphEdge]:
        """Create edges connecting hyperedges to entities."""
        edges = []

        for idx, hyperedge in enumerate(hyperedges):
            hyperedge_id = compute_mdhash_id(
                f"{hyperedge.description}_{idx}", prefix="hedge-"
            )

            for entity_name in hyperedge.entities:
                if entity_name in entities:
                    entity_id = compute_mdhash_id(entity_name, prefix="ent-")

                    edge = GraphEdge(
                        source_id=hyperedge_id,
                        target_id=entity_id,
                        edge_type="connects_to",
                        weight=hyperedge.weight,
                        metadata={"relationship": hyperedge.relationship},
                    )
                    edges.append(edge)

        return edges

    async def _store_graph(self, nodes: List[GraphNode], edges: List[GraphEdge]):
        """Store nodes and edges in graph storage."""
        # Store nodes
        for node in nodes:
            await self.graph_storage.upsert_node(
                node_id=node.node_id,
                node_data={
                    "entity_type": node.node_type,
                    "description": node.description,
                    "source_id": node.node_id,
                    **node.metadata,
                },
            )

        # Store edges
        for edge in edges:
            await self.graph_storage.upsert_edge(
                source_node_id=edge.source_id,
                target_node_id=edge.target_id,
                edge_data={
                    "weight": edge.weight,
                    "description": edge.metadata.get("relationship", ""),
                    **edge.metadata,
                },
            )

        # Commit
        await self.graph_storage.index_done_callback()

    async def process(self, input_data: GraphConstructionInput) -> GraphStructure:
        """Build hypergraph from entities and hyperedges.

        Args:
            input_data: Input with entities and hyperedges

        Returns:
            Graph structure with nodes and edges
        """
        # Merge duplicate entities
        merged_entities = self._merge_entities(input_data.entities)

        # Create graph nodes
        nodes = self._create_graph_nodes(merged_entities, input_data.hyperedges)

        # Create graph edges
        edges = self._create_graph_edges(merged_entities, input_data.hyperedges)

        # Store in graph storage
        await self._store_graph(nodes, edges)

        return GraphStructure(nodes=nodes, edges=edges)
