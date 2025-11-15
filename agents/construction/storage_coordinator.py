"""StorageCoordinatorAgent - Orchestrates multi-storage operations.

This agent coordinates writes across KV, Vector, and Graph storage backends.
"""

import sys
from typing import Dict, Any
import uuid

sys.path.insert(0, "/Users/gautamdudeja/repo/HyperGraphRAG")

from hypergraphrag.base import BaseKVStorage, BaseVectorStorage, BaseGraphStorage
from schemas.construction import StorageRequest, StorageResponse


class StorageCoordinatorAgent:
    """Agent for coordinating storage operations."""

    def __init__(
        self,
        kv_storage: BaseKVStorage,
        vector_storage: BaseVectorStorage,
        graph_storage: BaseGraphStorage,
    ):
        """Initialize StorageCoordinatorAgent.

        Args:
            kv_storage: KV storage backend
            vector_storage: Vector storage backend
            graph_storage: Graph storage backend
        """
        self.kv_storage = kv_storage
        self.vector_storage = vector_storage
        self.graph_storage = graph_storage

    async def process(self, request: StorageRequest) -> StorageResponse:
        """Process storage request.

        Args:
            request: Storage request

        Returns:
            Storage response
        """
        operation_id = str(uuid.uuid4())
        errors = []

        try:
            if request.storage_type == "kv":
                storage = self.kv_storage
            elif request.storage_type == "vector":
                storage = self.vector_storage
            elif request.storage_type == "graph":
                storage = self.graph_storage
            else:
                raise ValueError(f"Unknown storage type: {request.storage_type}")

            if request.operation == "write":
                await storage.upsert(request.data)
                await storage.index_done_callback()
            elif request.operation == "read":
                # Implement read operations as needed
                pass
            elif request.operation == "delete":
                # Implement delete operations as needed
                pass
            else:
                raise ValueError(f"Unknown operation: {request.operation}")

            return StorageResponse(
                success=True,
                operation_id=operation_id,
                errors=[],
            )

        except Exception as e:
            errors.append(str(e))
            return StorageResponse(
                success=False,
                operation_id=operation_id,
                errors=errors,
            )

    async def commit_all(self) -> StorageResponse:
        """Commit all storage backends."""
        operation_id = str(uuid.uuid4())
        errors = []

        try:
            await self.kv_storage.index_done_callback()
            await self.vector_storage.index_done_callback()
            await self.graph_storage.index_done_callback()

            return StorageResponse(
                success=True,
                operation_id=operation_id,
                errors=[],
            )
        except Exception as e:
            errors.append(str(e))
            return StorageResponse(
                success=False,
                operation_id=operation_id,
                errors=errors,
            )
