"""DocumentIngestionAgent - Handles document chunking and deduplication.

This agent:
1. Accepts raw documents
2. Generates unique IDs via MD5 hashing
3. Checks for duplicates in KV storage
4. Chunks documents by token size
5. Stores chunks in KV storage
"""

import sys
from typing import List, Dict, Any
import hashlib

# Add paths
sys.path.insert(0, "/Users/gautamdudeja/repo/HyperGraphRAG")

from hypergraphrag.base import BaseKVStorage
from schemas.common import TextChunk
from schemas.construction import DocumentInput, ChunkedDocument
from agents.utils import chunk_text_by_tokens, count_tokens


class DocumentIngestionAgent:
    """Agent for document ingestion, chunking, and deduplication."""

    def __init__(
        self,
        kv_storage: BaseKVStorage,
        chunk_token_size: int = 1200,
        chunk_overlap: int = 100,
    ):
        """Initialize DocumentIngestionAgent.

        Args:
            kv_storage: KV storage backend
            chunk_token_size: Target tokens per chunk
            chunk_overlap: Token overlap between chunks
        """
        self.kv_storage = kv_storage
        self.chunk_token_size = chunk_token_size
        self.chunk_overlap = chunk_overlap

    def _generate_doc_id(self, content: str) -> str:
        """Generate unique document ID via MD5 hash."""
        return hashlib.md5(content.encode()).hexdigest()

    async def _check_duplicate(self, doc_id: str) -> bool:
        """Check if document already exists in storage."""
        existing = await self.kv_storage.get_by_id(doc_id)
        return existing is not None

    def _chunk_document(
        self, content: str, doc_id: str, metadata: Dict[str, Any]
    ) -> List[TextChunk]:
        """Chunk document by token size with overlap."""
        chunks_text = chunk_text_by_tokens(
            content,
            chunk_size=self.chunk_token_size,
            overlap=self.chunk_overlap,
        )

        chunks = []
        for i, chunk_text in enumerate(chunks_text):
            chunk_id = f"{doc_id}_chunk_{i}"
            tokens = count_tokens(chunk_text)

            chunk = TextChunk(
                chunk_id=chunk_id,
                content=chunk_text,
                tokens=tokens,
                metadata={
                    **metadata,
                    "chunk_order_index": i,
                    "full_doc_id": doc_id,
                },
                doc_id=doc_id,
            )
            chunks.append(chunk)

        return chunks

    async def _store_chunks(self, chunks: List[TextChunk]) -> None:
        """Store chunks in KV storage."""
        chunk_data = {
            chunk.chunk_id: {
                "content": chunk.content,
                "tokens": chunk.tokens,
                "full_doc_id": chunk.doc_id,
                "chunk_order_index": chunk.metadata.get("chunk_order_index", 0),
                **chunk.metadata,
            }
            for chunk in chunks
        }
        await self.kv_storage.upsert(chunk_data)

    async def process(self, document: DocumentInput) -> ChunkedDocument:
        """Process a document through ingestion pipeline.

        Args:
            document: Input document

        Returns:
            ChunkedDocument with all chunks

        Raises:
            ValueError: If document is duplicate
        """
        # Generate document ID
        doc_id = self._generate_doc_id(document.content)

        # Check for duplicates
        if await self._check_duplicate(doc_id):
            raise ValueError(f"Document {doc_id} already exists in storage")

        # Chunk document
        chunks = self._chunk_document(
            content=document.content,
            doc_id=doc_id,
            metadata=document.metadata or {},
        )

        # Store chunks
        await self._store_chunks(chunks)

        # Commit storage
        await self.kv_storage.index_done_callback()

        return ChunkedDocument(doc_id=doc_id, chunks=chunks)

    async def process_batch(
        self, documents: List[DocumentInput], skip_duplicates: bool = True
    ) -> List[ChunkedDocument]:
        """Process multiple documents.

        Args:
            documents: List of input documents
            skip_duplicates: Skip duplicates instead of raising error

        Returns:
            List of chunked documents
        """
        results = []
        for doc in documents:
            try:
                result = await self.process(doc)
                results.append(result)
            except ValueError as e:
                if not skip_duplicates:
                    raise
                # Skip duplicate
                print(f"Skipping duplicate: {e}")

        return results
