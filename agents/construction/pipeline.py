"""Construction Pipeline - Orchestrates the full hypergraph construction flow.

This pipeline coordinates all construction agents:
Document → Ingestion → Extraction → GraphBuilder → Embedding → Storage
"""

import sys
from typing import List

sys.path.insert(0, "/Users/gautamdudeja/repo/HyperGraphRAG")

from hypergraphrag.base import BaseKVStorage, BaseVectorStorage, BaseGraphStorage

from schemas.construction import DocumentInput, ChunkedDocument, ExtractionResult, GraphStructure
from agents.construction.document_ingestion import DocumentIngestionAgent
from agents.construction.extraction import ExtractionAgent
from agents.construction.graph_builder import GraphBuilderAgent
from agents.construction.embedding import EmbeddingAgent
from agents.construction.storage_coordinator import StorageCoordinatorAgent
from agents.services.llm_service import LLMService, EmbeddingService


class HypergraphConstructionPipeline:
    """Pipeline for constructing hypergraphs from documents."""

    def __init__(
        self,
        kv_storage: BaseKVStorage,
        vector_storage: BaseVectorStorage,
        graph_storage: BaseGraphStorage,
        llm_service: LLMService,
        embedding_service: EmbeddingService,
        chunk_token_size: int = 1200,
        chunk_overlap: int = 100,
        max_gleaning_rounds: int = 2,
    ):
        """Initialize construction pipeline.

        Args:
            kv_storage: KV storage backend
            vector_storage: Vector storage backend
            graph_storage: Graph storage backend
            llm_service: LLM service
            embedding_service: Embedding service
            chunk_token_size: Chunk size in tokens
            chunk_overlap: Overlap between chunks
            max_gleaning_rounds: Number of gleaning rounds
        """
        # Initialize agents
        self.ingestion_agent = DocumentIngestionAgent(
            kv_storage=kv_storage,
            chunk_token_size=chunk_token_size,
            chunk_overlap=chunk_overlap,
        )

        self.extraction_agent = ExtractionAgent(
            llm_service=llm_service,
            max_gleaning_rounds=max_gleaning_rounds,
        )

        self.graph_builder_agent = GraphBuilderAgent(
            graph_storage=graph_storage,
            llm_service=llm_service,
        )

        self.embedding_agent = EmbeddingAgent(
            vector_storage=vector_storage,
            embedding_service=embedding_service,
        )

        self.storage_coordinator = StorageCoordinatorAgent(
            kv_storage=kv_storage,
            vector_storage=vector_storage,
            graph_storage=graph_storage,
        )

    async def process_document(self, document: DocumentInput) -> dict:
        """Process a single document through the complete pipeline.

        Args:
            document: Input document

        Returns:
            Dictionary with pipeline results
        """
        print(f"\n=== Processing Document ===")

        # 1. Ingestion (chunking)
        print("1. Ingesting and chunking document...")
        chunked_doc = await self.ingestion_agent.process(document)
        print(f"   Created {len(chunked_doc.chunks)} chunks")

        # 2. Extraction (entities and hyperedges)
        print("2. Extracting entities and hyperedges...")
        from schemas.construction import ChunkExtractionInput
        extraction_input = ChunkExtractionInput(chunks=chunked_doc.chunks)
        extraction_result = await self.extraction_agent.process(extraction_input)
        print(f"   Extracted {len(extraction_result.entities)} entities")
        print(f"   Extracted {len(extraction_result.hyperedges)} hyperedges")

        # 3. Graph Building
        print("3. Building hypergraph...")
        from schemas.construction import GraphConstructionInput
        graph_input = GraphConstructionInput(
            entities=extraction_result.entities,
            hyperedges=extraction_result.hyperedges,
        )
        graph_structure = await self.graph_builder_agent.process(graph_input)
        print(f"   Created {len(graph_structure.nodes)} nodes")
        print(f"   Created {len(graph_structure.edges)} edges")

        # 4. Embedding
        print("4. Embedding nodes and chunks...")
        embedding_result_nodes = await self.embedding_agent.embed_nodes(
            graph_structure.nodes
        )
        embedding_result_chunks = await self.embedding_agent.embed_chunks(
            chunked_doc.chunks
        )
        print(f"   Embedded {embedding_result_nodes.total_embedded} nodes "
              f"({embedding_result_nodes.cache_hits} cache hits)")
        print(f"   Embedded {embedding_result_chunks.total_embedded} chunks "
              f"({embedding_result_chunks.cache_hits} cache hits)")

        # 5. Final commit
        print("5. Committing all storage...")
        await self.storage_coordinator.commit_all()
        print("   ✓ Pipeline complete!")

        return {
            "doc_id": chunked_doc.doc_id,
            "chunks": len(chunked_doc.chunks),
            "entities": len(extraction_result.entities),
            "hyperedges": len(extraction_result.hyperedges),
            "nodes": len(graph_structure.nodes),
            "edges": len(graph_structure.edges),
        }

    async def process_documents(self, documents: List[DocumentInput]) -> List[dict]:
        """Process multiple documents.

        Args:
            documents: List of input documents

        Returns:
            List of results for each document
        """
        results = []
        for i, doc in enumerate(documents):
            print(f"\n{'='*60}")
            print(f"Document {i+1}/{len(documents)}")
            print(f"{'='*60}")
            result = await self.process_document(doc)
            results.append(result)

        return results
