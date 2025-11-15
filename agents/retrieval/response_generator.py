"""ResponseGeneratorAgent - Generates final response using LLM.

This agent:
1. Builds LLM prompt with assembled context
2. Calls LLM for response generation
3. Caches result for future queries
"""

from typing import Optional
import hashlib

from schemas.retrieval import (
    ResponseGenerationInput,
    ResponseOutput,
    AssembledContext,
)
from agents.services.llm_service import LLMService
from agents.hypergraph_utils import PROMPTS


class ResponseGeneratorAgent:
    """Agent for response generation."""

    def __init__(
        self,
        llm_service: LLMService,
        enable_cache: bool = True,
    ):
        """Initialize ResponseGeneratorAgent.

        Args:
            llm_service: LLM service
            enable_cache: Enable response caching
        """
        self.llm_service = llm_service
        self.enable_cache = enable_cache
        self.response_cache = {}

    def _build_system_prompt(
        self, context: AssembledContext, query: str
    ) -> str:
        """Build system prompt with context."""
        # Combine all context
        full_context = ""

        if context.entity_context:
            full_context += "=== Entities ===\n" + context.entity_context + "\n\n"

        if context.relationship_context:
            full_context += "=== Relationships ===\n" + context.relationship_context + "\n\n"

        if context.text_context:
            full_context += "=== Source Texts ===\n" + context.text_context + "\n\n"

        # Select prompt based on mode
        if context.mode == "local":
            prompt_template = PROMPTS["local_search"]
        elif context.mode == "global":
            prompt_template = PROMPTS["global_search"]
        else:  # hybrid
            prompt_template = PROMPTS["local_search"]  # Default to local

        # Format prompt
        prompt = prompt_template.format(
            context_data=full_context,
            query=query,
            response_type="comprehensive",  # Could be configurable
        )

        return prompt

    def _generate_query_id(self, query: str, context_hash: str) -> str:
        """Generate query ID for caching."""
        combined = f"{query}:{context_hash}"
        return hashlib.md5(combined.encode()).hexdigest()

    async def process(
        self, input_data: ResponseGenerationInput
    ) -> ResponseOutput:
        """Generate response for query.

        Args:
            input_data: Response generation input

        Returns:
            Response output
        """
        # Generate query ID for caching
        context_hash = hashlib.md5(
            (input_data.context.entity_context +
             input_data.context.relationship_context +
             input_data.context.text_context).encode()
        ).hexdigest()

        query_id = self._generate_query_id(input_data.query, context_hash)

        # Check cache
        if self.enable_cache and query_id in self.response_cache:
            cached_response = self.response_cache[query_id]
            return ResponseOutput(
                response=cached_response,
                query_id=query_id,
                tokens_used=0,  # Cached
                cached=True,
            )

        # Build prompt
        prompt = self._build_system_prompt(input_data.context, input_data.query)

        # Generate response
        messages = [{"role": "user", "content": prompt}]
        response = await self.llm_service.complete(messages)

        # Clean response (remove any artifacts)
        response = response.strip()

        # Cache result
        if self.enable_cache:
            self.response_cache[query_id] = response

        return ResponseOutput(
            response=response,
            query_id=query_id,
            tokens_used=len(response.split()),  # Rough estimate
            cached=False,
        )
