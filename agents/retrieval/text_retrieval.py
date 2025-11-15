"""TextRetrievalAgent - Retrieves source text chunks.

This agent:
1. Traces from entity/relationship IDs to source chunks
2. Ranks chunks by relevance to query
3. Enforces token budget
4. Returns formatted text units
"""

from typing import List, Set
from agents.utils import count_tokens

from schemas.retrieval import (
    TextRetrievalInput,
    TextRetrievalResult,
    TextUnit,
)
from storage.base import BaseKVStorage, BaseVectorStorage
from agents.services.llm_service import EmbeddingService


class TextRetrievalAgent:
    """Agent for source text retrieval."""

    def __init__(
        self,
        kv_storage: BaseKVStorage,
        vector_storage: BaseVectorStorage,
        embedding_service: EmbeddingService,
    ):
        """Initialize TextRetrievalAgent.

        Args:
            kv_storage: KV storage backend
            vector_storage: Vector storage backend
            embedding_service: Embedding service
        """
        self.kv_storage = kv_storage
        self.vector_storage = vector_storage
        self.embedding_service = embedding_service

    async def _get_chunk_ids_from_entities(
        self, entity_ids: List[str]
    ) -> Set[str]:
        """Get chunk IDs associated with entities."""
        chunk_ids = set()

        for entity_id in entity_ids:
            # Query vector storage for this entity
            results = await self.vector_storage.query(entity_id, top_k=1)
            if results:
                source_chunks = results[0].get("source_chunks", [])
                chunk_ids.update(source_chunks)

        return chunk_ids

    async def _get_chunk_ids_from_hyperedges(
        self, hyperedge_ids: List[str]
    ) -> Set[str]:
        """Get chunk IDs associated with hyperedges."""
        chunk_ids = set()

        for hyperedge_id in hyperedge_ids:
            results = await self.vector_storage.query(hyperedge_id, top_k=1)
            if results:
                source_chunks = results[0].get("source_chunks", [])
                chunk_ids.update(source_chunks)

        return chunk_ids

    async def _fetch_chunks(self, chunk_ids: List[str]) -> List[dict]:
        """Fetch chunk data from KV storage."""
        chunks = await self.kv_storage.get_by_ids(list(chunk_ids))
        return [c for c in chunks if c is not None]

    def _rank_chunks(self, chunks: List[dict], query_keywords: List[str]) -> List[dict]:
        """Rank chunks by keyword relevance."""
        scored_chunks = []

        for chunk in chunks:
            content = chunk.get("content", "").lower()
            score = sum(1 for kw in query_keywords if kw.lower() in content)
            scored_chunks.append((score, chunk))

        # Sort by score descending
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        return [chunk for _, chunk in scored_chunks]

    def _enforce_token_budget(
        self, chunks: List[dict], max_tokens: int
    ) -> List[dict]:
        """Limit chunks to token budget."""
        result = []
        total_tokens = 0

        for chunk in chunks:
            chunk_tokens = chunk.get("tokens", count_tokens(chunk.get("content", "")))
            if total_tokens + chunk_tokens > max_tokens:
                break
            result.append(chunk)
            total_tokens += chunk_tokens

        return result

    async def process(self, input_data: TextRetrievalInput) -> TextRetrievalResult:
        """Retrieve relevant text chunks.

        Args:
            input_data: Text retrieval input

        Returns:
            Retrieved text units
        """
        # Get chunk IDs from entities and hyperedges
        chunk_ids = set()

        if input_data.entity_ids:
            entity_chunks = await self._get_chunk_ids_from_entities(
                input_data.entity_ids
            )
            chunk_ids.update(entity_chunks)

        if input_data.hyperedge_ids:
            hyperedge_chunks = await self._get_chunk_ids_from_hyperedges(
                input_data.hyperedge_ids
            )
            chunk_ids.update(hyperedge_chunks)

        # Fetch chunks
        chunks = await self._fetch_chunks(list(chunk_ids))

        # Rank chunks (simplified - could use query keywords)
        # For now, just use as-is
        ranked_chunks = chunks[:input_data.top_k]

        # Enforce token budget
        budgeted_chunks = self._enforce_token_budget(
            ranked_chunks, input_data.max_tokens
        )

        # Create text units
        text_units = []
        total_tokens = 0

        for chunk in budgeted_chunks:
            chunk_tokens = chunk.get("tokens", count_tokens(chunk.get("content", "")))
            text_unit = TextUnit(
                chunk_id=chunk.get("chunk_id", "unknown"),
                content=chunk.get("content", ""),
                tokens=chunk_tokens,
                relevance_score=1.0,  # Could improve with actual scoring
                source_entities=input_data.entity_ids[:5],  # Sample
            )
            text_units.append(text_unit)
            total_tokens += chunk_tokens

        return TextRetrievalResult(
            text_units=text_units,
            total_tokens=total_tokens,
        )
