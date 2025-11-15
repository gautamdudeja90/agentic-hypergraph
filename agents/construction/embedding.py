"""EmbeddingAgent - Batch embedding with caching.

This agent:
1. Batch embeds entities, hyperedges, and chunks
2. Manages embedding cache for deduplication
3. Stores embeddings in vector DB
4. Handles rate limiting and concurrency control
"""

import sys
from typing import List, Dict
import hashlib

sys.path.insert(0, "/Users/gautamdudeja/repo/HyperGraphRAG")

from hypergraphrag.base import BaseVectorStorage
from schemas.common import GraphNode, TextChunk
from schemas.construction import EmbeddingInput, EmbeddingOutput, EmbeddingResult
from agents.services.llm_service import EmbeddingService


class EmbeddingAgent:
    """Agent for embedding and vector storage."""

    def __init__(
        self,
        vector_storage: BaseVectorStorage,
        embedding_service: EmbeddingService,
        enable_cache: bool = True,
    ):
        """Initialize EmbeddingAgent.

        Args:
            vector_storage: Vector storage backend
            embedding_service: Embedding service
            enable_cache: Enable embedding cache
        """
        self.vector_storage = vector_storage
        self.embedding_service = embedding_service
        self.enable_cache = enable_cache
        self.cache: Dict[str, List[float]] = {}

    def _cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    async def _embed_with_cache(
        self, texts: List[str]
    ) -> tuple[List[List[float]], int]:
        """Embed texts with caching."""
        embeddings = []
        cache_hits = 0
        to_embed = []
        to_embed_indices = []

        # Check cache
        for i, text in enumerate(texts):
            if self.enable_cache:
                cache_key = self._cache_key(text)
                if cache_key in self.cache:
                    embeddings.append(self.cache[cache_key])
                    cache_hits += 1
                    continue

            to_embed.append(text)
            to_embed_indices.append(i)

        # Embed uncached texts
        if to_embed:
            new_embeddings = await self.embedding_service.embed_batch(to_embed)

            # Update cache and results
            for i, text in enumerate(to_embed):
                embedding = new_embeddings[i].tolist()
                if self.enable_cache:
                    cache_key = self._cache_key(text)
                    self.cache[cache_key] = embedding
                embeddings.insert(to_embed_indices[i], embedding)

        return embeddings, cache_hits

    async def embed_nodes(self, nodes: List[GraphNode]) -> EmbeddingOutput:
        """Embed graph nodes and store in vector DB."""
        # Prepare data for embedding
        texts = []
        node_ids = []

        for node in nodes:
            # Create text representation for embedding
            if node.node_type == "entity":
                text = f"{node.name}: {node.description}"
            else:  # hyperedge
                text = node.description
            texts.append(text)
            node_ids.append(node.node_id)

        # Embed with cache
        embeddings, cache_hits = await self._embed_with_cache(texts)

        # Prepare data for vector storage
        vector_data = {}
        for i, node in enumerate(nodes):
            vector_data[node.node_id] = {
                "content": texts[i],
                "embedding": embeddings[i],
                "entity_type": node.node_type,
                "description": node.description,
            }

        # Upsert to vector storage
        await self.vector_storage.upsert(vector_data)

        # Create results
        results = [
            EmbeddingResult(
                item_id=node_id,
                embedding=embeddings[i],
                cached=i < cache_hits,
            )
            for i, node_id in enumerate(node_ids)
        ]

        return EmbeddingOutput(
            results=results,
            total_embedded=len(results),
            cache_hits=cache_hits,
        )

    async def embed_chunks(self, chunks: List[TextChunk]) -> EmbeddingOutput:
        """Embed text chunks and store in vector DB."""
        # Prepare data
        texts = [chunk.content for chunk in chunks]
        chunk_ids = [chunk.chunk_id for chunk in chunks]

        # Embed with cache
        embeddings, cache_hits = await self._embed_with_cache(texts)

        # Prepare data for vector storage
        vector_data = {}
        for i, chunk in enumerate(chunks):
            vector_data[chunk.chunk_id] = {
                "content": chunk.content,
                "embedding": embeddings[i],
                "tokens": chunk.tokens,
                "full_doc_id": chunk.doc_id,
            }

        # Upsert to vector storage
        await self.vector_storage.upsert(vector_data)

        # Create results
        results = [
            EmbeddingResult(
                item_id=chunk_id,
                embedding=embeddings[i],
                cached=i < cache_hits,
            )
            for i, chunk_id in enumerate(chunk_ids)
        ]

        return EmbeddingOutput(
            results=results,
            total_embedded=len(results),
            cache_hits=cache_hits,
        )

    async def process(self, input_data: EmbeddingInput) -> EmbeddingOutput:
        """Process embedding request.

        Args:
            input_data: Input with items to embed

        Returns:
            Embedding output with results
        """
        # Group by type
        texts = [item.text for item in input_data.items]
        item_ids = [item.item_id for item in input_data.items]

        # Embed with cache
        embeddings, cache_hits = await self._embed_with_cache(texts)

        # Create results
        results = [
            EmbeddingResult(
                item_id=item_ids[i],
                embedding=embeddings[i],
                cached=i < cache_hits,
            )
            for i in range(len(input_data.items))
        ]

        return EmbeddingOutput(
            results=results,
            total_embedded=len(results),
            cache_hits=cache_hits,
        )
