"""Utility functions for agents."""

import sys
import hashlib
import tiktoken
from typing import List, Callable
import asyncio

# Add HyperGraphRAG to path
sys.path.insert(0, "/Users/gautamdudeja/repo/HyperGraphRAG")

from hypergraphrag.utils import (
    compute_mdhash_id,
    EmbeddingFunc,
)


def hash_text(text: str) -> str:
    """Generate MD5 hash of text."""
    return hashlib.md5(text.encode()).hexdigest()


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text using tiktoken."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def chunk_text_by_tokens(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 100,
    model: str = "gpt-4"
) -> List[str]:
    """Chunk text by token count with overlap."""
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)

        if end >= len(tokens):
            break
        start = end - overlap

    return chunks


async def batch_embed_with_semaphore(
    texts: List[str],
    embedding_func: Callable,
    batch_size: int = 100,
    max_concurrent: int = 16
) -> List[List[float]]:
    """Batch embed texts with concurrency control."""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def embed_batch(batch: List[str]):
        async with semaphore:
            return await embedding_func(batch)

    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    results = await asyncio.gather(*[embed_batch(batch) for batch in batches])

    # Flatten results
    embeddings = []
    for batch_result in results:
        embeddings.extend(batch_result)

    return embeddings
