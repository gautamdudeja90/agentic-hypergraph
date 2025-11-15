"""QueryPlannerAgent - Parses queries and determines retrieval strategy.

This agent:
1. Extracts keywords from user query
2. Determines retrieval mode (local/global/hybrid)
3. Creates retrieval tasks for downstream agents
4. Checks query cache
"""

from typing import List, Dict
import hashlib

from schemas.retrieval import QueryInput, QueryPlan, RetrievalTask
from agents.services.llm_service import LLMService
from agents.hypergraph_utils import PROMPTS


class QueryPlannerAgent:
    """Agent for query planning and routing."""

    def __init__(
        self,
        llm_service: LLMService,
        enable_cache: bool = True,
    ):
        """Initialize QueryPlannerAgent.

        Args:
            llm_service: LLM service
            enable_cache: Enable query result caching
        """
        self.llm_service = llm_service
        self.enable_cache = enable_cache
        self.query_cache: Dict[str, str] = {}

    def _generate_query_id(self, query: str) -> str:
        """Generate unique query ID."""
        return hashlib.md5(query.encode()).hexdigest()

    async def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query using LLM."""
        prompt = PROMPTS["keywords_extraction"].format(
            query=query,
            language=PROMPTS["DEFAULT_LANGUAGE"],
            tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        )

        messages = [{"role": "user", "content": prompt}]
        result = await self.llm_service.complete(messages)

        # Parse keywords
        keywords = [
            kw.strip()
            for kw in result.split(PROMPTS["DEFAULT_TUPLE_DELIMITER"])
            if kw.strip()
        ]

        return keywords

    def _determine_mode(self, query: str, requested_mode: str) -> str:
        """Determine retrieval mode (local/global/hybrid)."""
        # For now, use requested mode
        # Could add logic to auto-detect based on query type
        if requested_mode in ["local", "global", "hybrid"]:
            return requested_mode
        return "hybrid"

    def _create_retrieval_tasks(
        self, keywords: List[str], mode: str, top_k: int
    ) -> List[RetrievalTask]:
        """Create retrieval tasks based on mode."""
        tasks = []

        if mode in ["local", "hybrid"]:
            # Entity retrieval
            tasks.append(
                RetrievalTask(
                    agent="entity",
                    parameters={"keywords": keywords, "top_k": top_k},
                )
            )

        if mode in ["global", "hybrid"]:
            # Relationship retrieval
            tasks.append(
                RetrievalTask(
                    agent="relationship",
                    parameters={"keywords": keywords, "top_k": top_k},
                )
            )

        # Text retrieval (always included)
        tasks.append(
            RetrievalTask(
                agent="text",
                parameters={"keywords": keywords, "top_k": top_k // 2},
            )
        )

        return tasks

    async def process(self, input_data: QueryInput) -> QueryPlan:
        """Process query to create retrieval plan.

        Args:
            input_data: Query input

        Returns:
            Query plan with tasks
        """
        query_id = self._generate_query_id(input_data.query)

        # Check cache
        cached_result = None
        if self.enable_cache and query_id in self.query_cache:
            cached_result = self.query_cache[query_id]

        # Extract keywords
        keywords = await self._extract_keywords(input_data.query)

        # Determine mode
        mode = self._determine_mode(input_data.query, input_data.mode or "hybrid")

        # Create retrieval tasks
        tasks = self._create_retrieval_tasks(keywords, mode, input_data.top_k)

        return QueryPlan(
            query_id=query_id,
            keywords=keywords,
            mode=mode,
            retrieval_tasks=tasks,
            cached_result=cached_result,
        )
