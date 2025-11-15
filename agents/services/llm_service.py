"""LLM service using LiteLLM for OpenAI-compatible endpoints.

Supports OpenAI, Anthropic, local models, and any OpenAI-compatible API.
NO Google Cloud or Gemini dependencies.
"""

import os
import sys
from typing import List, Dict, Any, Optional, Callable
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
import numpy as np

# Add HyperGraphRAG to path for utilities
sys.path.insert(0, "/Users/gautamdudeja/repo/HyperGraphRAG")

# Use OpenAI SDK directly (LiteLLM compatible)
from openai import AsyncOpenAI


class LLMService:
    """LLM service using OpenAI-compatible endpoints via LiteLLM."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,  # For custom endpoints
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_retries: int = 3,
    ):
        """Initialize LLM service.

        Args:
            api_key: API key (defaults to OPENAI_API_KEY env var)
            base_url: Custom base URL for OpenAI-compatible endpoints
            model: Model name (e.g., "gpt-4o-mini", "claude-3-sonnet", "local-model")
            temperature: Sampling temperature
            max_retries: Maximum retry attempts
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries

        # Initialize AsyncOpenAI client (works with any OpenAI-compatible endpoint)
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url,  # None = default OpenAI, or custom endpoint
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def complete(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Complete a chat conversation.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Completion text
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def batch_complete(
        self,
        messages_list: List[List[Dict[str, str]]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_concurrent: int = 16,
    ) -> List[str]:
        """Batch complete multiple conversations with concurrency control.

        Args:
            messages_list: List of message lists
            temperature: Override default temperature
            max_tokens: Maximum tokens per completion
            max_concurrent: Maximum concurrent requests

        Returns:
            List of completion texts
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def complete_with_semaphore(messages):
            async with semaphore:
                return await self.complete(messages, temperature, max_tokens)

        tasks = [complete_with_semaphore(msgs) for msgs in messages_list]
        return await asyncio.gather(*tasks)


class EmbeddingService:
    """Embedding service using OpenAI embeddings.

    Compatible with OpenAI and OpenAI-compatible embedding endpoints.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        max_retries: int = 3,
    ):
        """Initialize embedding service.

        Args:
            api_key: API key (defaults to OPENAI_API_KEY env var)
            base_url: Custom base URL for OpenAI-compatible endpoints
            model: Embedding model name
            dimensions: Embedding dimensions
            max_retries: Maximum retry attempts
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.model = model
        self.dimensions = dimensions
        self.max_retries = max_retries

        # Initialize AsyncOpenAI client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=base_url,
        )

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimensions (for HyperGraphRAG compatibility)."""
        return self.dimensions

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def __call__(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts (HyperGraphRAG compatible signature).

        Args:
            texts: List of texts to embed

        Returns:
            Numpy array of embeddings (shape: [len(texts), dimensions])
        """
        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
        )

        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 100,
        max_concurrent: int = 16,
    ) -> np.ndarray:
        """Embed texts in batches with concurrency control.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for API calls
            max_concurrent: Maximum concurrent requests

        Returns:
            Numpy array of embeddings
        """
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

        semaphore = asyncio.Semaphore(max_concurrent)

        async def embed_with_semaphore(batch):
            async with semaphore:
                return await self(batch)

        tasks = [embed_with_semaphore(batch) for batch in batches]
        batch_embeddings = await asyncio.gather(*tasks)

        return np.concatenate(batch_embeddings, axis=0)


def create_llm_service(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> LLMService:
    """Factory function to create LLM service."""
    return LLMService(api_key=api_key, base_url=base_url, model=model)


def create_embedding_service(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "text-embedding-3-small",
    dimensions: int = 1536,
) -> EmbeddingService:
    """Factory function to create embedding service."""
    return EmbeddingService(
        api_key=api_key,
        base_url=base_url,
        model=model,
        dimensions=dimensions,
    )
