"""Retrieval Pipeline - Orchestrates the full query → response flow.

This pipeline coordinates all retrieval agents:
Query → QueryPlanner → [EntityRetrieval + RelationshipRetrieval + TextRetrieval] → ContextAssembly → ResponseGenerator
"""

from typing import List

from storage.base import BaseKVStorage, BaseVectorStorage, BaseGraphStorage
from schemas.retrieval import QueryInput, ResponseOutput
from agents.retrieval.query_planner import QueryPlannerAgent
from agents.retrieval.entity_retrieval import EntityRetrievalAgent
from agents.retrieval.relationship_retrieval import RelationshipRetrievalAgent
from agents.retrieval.text_retrieval import TextRetrievalAgent
from agents.retrieval.context_assembly import ContextAssemblyAgent
from agents.retrieval.response_generator import ResponseGeneratorAgent
from agents.services.llm_service import LLMService, EmbeddingService


class HypergraphRetrievalPipeline:
    """Pipeline for querying hypergraphs and generating responses."""

    def __init__(
        self,
        kv_storage: BaseKVStorage,
        vector_storage: BaseVectorStorage,
        graph_storage: BaseGraphStorage,
        llm_service: LLMService,
        embedding_service: EmbeddingService,
        enable_cache: bool = True,
    ):
        """Initialize retrieval pipeline.

        Args:
            kv_storage: KV storage backend
            vector_storage: Vector storage backend
            graph_storage: Graph storage backend
            llm_service: LLM service
            embedding_service: Embedding service
            enable_cache: Enable query/response caching
        """
        # Initialize agents
        self.query_planner = QueryPlannerAgent(
            llm_service=llm_service,
            enable_cache=enable_cache,
        )

        self.entity_retrieval = EntityRetrievalAgent(
            vector_storage=vector_storage,
            graph_storage=graph_storage,
            embedding_service=embedding_service,
        )

        self.relationship_retrieval = RelationshipRetrievalAgent(
            vector_storage=vector_storage,
            graph_storage=graph_storage,
            embedding_service=embedding_service,
        )

        self.text_retrieval = TextRetrievalAgent(
            kv_storage=kv_storage,
            vector_storage=vector_storage,
            embedding_service=embedding_service,
        )

        self.context_assembly = ContextAssemblyAgent()

        self.response_generator = ResponseGeneratorAgent(
            llm_service=llm_service,
            enable_cache=enable_cache,
        )

    async def query(self, query: str, mode: str = "hybrid", top_k: int = 20) -> ResponseOutput:
        """Process a query through the complete retrieval pipeline.

        Args:
            query: User query
            mode: Retrieval mode (local/global/hybrid)
            top_k: Number of top results to retrieve

        Returns:
            Response output
        """
        print(f"\n=== Processing Query ===")
        print(f"Query: {query}")
        print(f"Mode: {mode}")

        # 1. Query Planning
        print("\n1. Planning query...")
        query_input = QueryInput(query=query, mode=mode, top_k=top_k)
        query_plan = await self.query_planner.process(query_input)
        print(f"   Keywords: {', '.join(query_plan.keywords[:5])}")
        print(f"   Mode: {query_plan.mode}")
        print(f"   Tasks: {len(query_plan.retrieval_tasks)}")

        # Check if cached
        if query_plan.cached_result:
            print("   ✓ Using cached result!")
            return ResponseOutput(
                response=query_plan.cached_result,
                query_id=query_plan.query_id,
                tokens_used=0,
                cached=True,
            )

        # 2. Parallel Retrieval
        print("\n2. Retrieving from hypergraph...")

        # Entity retrieval
        from schemas.retrieval import EntityRetrievalInput
        entity_input = EntityRetrievalInput(
            keywords=query_plan.keywords,
            top_k=top_k,
            include_neighbors=True,
        )
        entity_result = await self.entity_retrieval.process(entity_input)
        print(f"   Entities: {len(entity_result.entities)}")

        # Relationship retrieval
        from schemas.retrieval import RelationshipRetrievalInput
        rel_input = RelationshipRetrievalInput(
            keywords=query_plan.keywords,
            top_k=top_k,
            include_entities=True,
        )
        rel_result = await self.relationship_retrieval.process(rel_input)
        print(f"   Relationships: {len(rel_result.relationships)}")

        # Text retrieval
        from schemas.retrieval import TextRetrievalInput
        text_input = TextRetrievalInput(
            entity_ids=[e.entity_id for e in entity_result.entities[:10]],
            hyperedge_ids=[r.hyperedge_id for r in rel_result.relationships[:10]],
            top_k=10,
            max_tokens=2000,
        )
        text_result = await self.text_retrieval.process(text_input)
        print(f"   Text units: {len(text_result.text_units)} ({text_result.total_tokens} tokens)")

        # 3. Context Assembly
        print("\n3. Assembling context...")
        from schemas.retrieval import ContextAssemblyInput
        context_input = ContextAssemblyInput(
            entities=entity_result.entities,
            relationships=rel_result.relationships,
            texts=text_result.text_units,
            mode=query_plan.mode,
            max_tokens=4000,
        )
        context = await self.context_assembly.process(context_input)
        print(f"   Total context tokens: {context.total_tokens}")

        # 4. Response Generation
        print("\n4. Generating response...")
        from schemas.retrieval import ResponseGenerationInput
        response_input = ResponseGenerationInput(
            query=query,
            context=context,
        )
        response_output = await self.response_generator.process(response_input)
        print("   ✓ Response generated!")

        return response_output

    async def query_batch(self, queries: List[str], mode: str = "hybrid") -> List[ResponseOutput]:
        """Process multiple queries.

        Args:
            queries: List of user queries
            mode: Retrieval mode

        Returns:
            List of responses
        """
        results = []
        for i, query in enumerate(queries):
            print(f"\n{'='*60}")
            print(f"Query {i+1}/{len(queries)}")
            print(f"{'='*60}")
            result = await self.query(query, mode=mode)
            results.append(result)

        return results
