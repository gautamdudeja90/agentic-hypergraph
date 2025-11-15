"""Storage implementations - directly using HyperGraphRAG storage backends.

This module provides factory functions to create storage instances compatible
with both ADK and HyperGraphRAG.
"""

import os
import sys
from typing import Any, Callable

# Add HyperGraphRAG to path
sys.path.insert(0, "/Users/gautamdudeja/repo/HyperGraphRAG")

from hypergraphrag.storage import (
    JsonKVStorage,
    NanoVectorDBStorage,
    NetworkXStorage,
)
from hypergraphrag.base import BaseKVStorage, BaseVectorStorage, BaseGraphStorage


def create_kv_storage(
    namespace: str,
    working_dir: str,
    embedding_func: Callable = None
) -> BaseKVStorage:
    """Create JSON-based KV storage."""
    os.makedirs(working_dir, exist_ok=True)
    return JsonKVStorage(
        namespace=namespace,
        global_config={"working_dir": working_dir},
        embedding_func=embedding_func
    )


def create_vector_storage(
    namespace: str,
    working_dir: str,
    embedding_func: Callable,
    embedding_batch_num: int = 100
) -> BaseVectorStorage:
    """Create NanoVectorDB-based vector storage."""
    os.makedirs(working_dir, exist_ok=True)
    return NanoVectorDBStorage(
        namespace=namespace,
        global_config={
            "working_dir": working_dir,
            "embedding_batch_num": embedding_batch_num
        },
        embedding_func=embedding_func
    )


def create_graph_storage(
    namespace: str,
    working_dir: str,
    embedding_func: Callable = None
) -> BaseGraphStorage:
    """Create NetworkX-based graph storage."""
    os.makedirs(working_dir, exist_ok=True)
    return NetworkXStorage(
        namespace=namespace,
        global_config={"working_dir": working_dir},
        embedding_func=embedding_func
    )


def create_all_storage(
    namespace: str,
    working_dir: str,
    embedding_func: Callable,
    embedding_batch_num: int = 100
) -> tuple[BaseKVStorage, BaseVectorStorage, BaseGraphStorage]:
    """Create all three storage backends."""
    return (
        create_kv_storage(namespace, working_dir, embedding_func),
        create_vector_storage(namespace, working_dir, embedding_func, embedding_batch_num),
        create_graph_storage(namespace, working_dir, embedding_func)
    )
