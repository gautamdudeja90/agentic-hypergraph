"""Storage base classes from HyperGraphRAG.

These are the core storage abstractions that our agents will use.
"""

from dataclasses import dataclass, field
from typing import TypedDict, Union, Literal, Generic, TypeVar, Any
import numpy as np


TextChunkSchema = TypedDict(
    "TextChunkSchema",
    {"tokens": int, "content": str, "full_doc_id": str, "chunk_order_index": int},
)

T = TypeVar("T")


# Embedding function type (callable that takes text and returns embedding)
EmbeddingFunc = Any  # Will be properly typed in utils


@dataclass
class StorageNameSpace:
    """Base storage namespace with callbacks."""
    namespace: str
    global_config: dict

    async def index_done_callback(self):
        """Commit the storage operations after indexing."""
        pass

    async def query_done_callback(self):
        """Commit the storage operations after querying."""
        pass


@dataclass
class BaseVectorStorage(StorageNameSpace):
    """Vector storage for semantic search."""
    embedding_func: EmbeddingFunc
    meta_fields: set = field(default_factory=set)

    async def query(self, query: str, top_k: int) -> list[dict]:
        """Query vector storage with text, returns top-k similar items."""
        raise NotImplementedError

    async def upsert(self, data: dict[str, dict]):
        """Use 'content' field from value for embedding, use key as id.
        If embedding_func is None, use 'embedding' field from value.
        """
        raise NotImplementedError


@dataclass
class BaseKVStorage(Generic[T], StorageNameSpace):
    """Key-value storage for documents and chunks."""
    embedding_func: EmbeddingFunc

    async def all_keys(self) -> list[str]:
        """Get all keys in storage."""
        raise NotImplementedError

    async def get_by_id(self, id: str) -> Union[T, None]:
        """Get value by ID."""
        raise NotImplementedError

    async def get_by_ids(
        self, ids: list[str], fields: Union[set[str], None] = None
    ) -> list[Union[T, None]]:
        """Get multiple values by IDs."""
        raise NotImplementedError

    async def filter_keys(self, data: list[str]) -> set[str]:
        """Return non-existent keys."""
        raise NotImplementedError

    async def upsert(self, data: dict[str, T]):
        """Insert or update data."""
        raise NotImplementedError

    async def drop(self):
        """Drop all data."""
        raise NotImplementedError


@dataclass
class BaseGraphStorage(StorageNameSpace):
    """Graph storage for entities and hyperedges."""
    embedding_func: EmbeddingFunc = None

    async def has_node(self, node_id: str) -> bool:
        """Check if node exists."""
        raise NotImplementedError

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        """Check if edge exists."""
        raise NotImplementedError

    async def node_degree(self, node_id: str) -> int:
        """Get node degree (number of edges)."""
        raise NotImplementedError

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        """Get edge degree."""
        raise NotImplementedError

    async def get_node(self, node_id: str) -> Union[dict, None]:
        """Get node data."""
        raise NotImplementedError

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> Union[dict, None]:
        """Get edge data."""
        raise NotImplementedError

    async def get_node_edges(
        self, source_node_id: str
    ) -> Union[list[tuple[str, str]], None]:
        """Get all edges connected to a node."""
        raise NotImplementedError

    async def upsert_node(self, node_id: str, node_data: dict[str, str]):
        """Insert or update node."""
        raise NotImplementedError

    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ):
        """Insert or update edge."""
        raise NotImplementedError

    async def delete_node(self, node_id: str):
        """Delete node."""
        raise NotImplementedError

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        """Embed nodes (not used in hypergraphrag)."""
        raise NotImplementedError("Node embedding is not used in hypergraphrag.")
