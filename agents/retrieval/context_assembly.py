"""ContextAssemblyAgent - Assembles context from retrieval results.

This agent:
1. Combines entity, relationship, and text retrieval results
2. Formats as CSV or structured context
3. Enforces global token limits
4. Handles mode-specific assembly (local/global/hybrid)
"""

from typing import List
import csv
from io import StringIO

from agents.utils import count_tokens
from schemas.retrieval import (
    ContextAssemblyInput,
    AssembledContext,
    RetrievedEntity,
    RetrievedRelationship,
    TextUnit,
)


class ContextAssemblyAgent:
    """Agent for context assembly."""

    def __init__(self):
        """Initialize ContextAssemblyAgent."""
        pass

    def _format_entities_csv(self, entities: List[RetrievedEntity]) -> str:
        """Format entities as CSV."""
        if not entities:
            return ""

        output = StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(["entity_name", "entity_type", "description", "relevance_score", "degree"])

        # Rows
        for entity in entities:
            writer.writerow([
                entity.name,
                entity.type,
                entity.description[:200],  # Truncate description
                f"{entity.relevance_score:.3f}",
                entity.degree,
            ])

        return output.getvalue()

    def _format_relationships_csv(self, relationships: List[RetrievedRelationship]) -> str:
        """Format relationships as CSV."""
        if not relationships:
            return ""

        output = StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow(["relationship_id", "description", "connected_entities", "relevance_score"])

        # Rows
        for rel in relationships:
            writer.writerow([
                rel.hyperedge_id[:16],  # Shortened ID
                rel.description[:200],  # Truncate
                "|".join(rel.connected_entities[:5]),  # Limit entities shown
                f"{rel.relevance_score:.3f}",
            ])

        return output.getvalue()

    def _format_texts(self, texts: List[TextUnit]) -> str:
        """Format text units."""
        if not texts:
            return ""

        formatted = []
        for i, text in enumerate(texts):
            formatted.append(f"[Source {i+1}]\n{text.content}\n")

        return "\n".join(formatted)

    def _combine_contexts(
        self,
        entity_context: str,
        relationship_context: str,
        text_context: str,
        mode: str,
    ) -> tuple[str, str, str]:
        """Combine contexts based on mode."""
        if mode == "local":
            # Entity-centric: prioritize entities
            return entity_context, relationship_context, text_context

        elif mode == "global":
            # Relationship-centric: prioritize relationships
            return entity_context, relationship_context, text_context

        else:  # hybrid
            # Balanced: include all
            return entity_context, relationship_context, text_context

    def _enforce_token_limits(
        self,
        entity_context: str,
        relationship_context: str,
        text_context: str,
        max_tokens: int,
    ) -> tuple[str, str, str]:
        """Enforce global token limits."""
        total_tokens = (
            count_tokens(entity_context) +
            count_tokens(relationship_context) +
            count_tokens(text_context)
        )

        if total_tokens <= max_tokens:
            return entity_context, relationship_context, text_context

        # Simple truncation strategy: proportional reduction
        ratio = max_tokens / total_tokens
        target_entity_tokens = int(count_tokens(entity_context) * ratio)
        target_rel_tokens = int(count_tokens(relationship_context) * ratio)
        target_text_tokens = int(count_tokens(text_context) * ratio)

        # Truncate each section (simple word-based truncation)
        def truncate_to_tokens(text: str, target: int) -> str:
            words = text.split()
            # Rough estimate: ~1.3 words per token
            target_words = int(target * 1.3)
            return " ".join(words[:target_words])

        return (
            truncate_to_tokens(entity_context, target_entity_tokens),
            truncate_to_tokens(relationship_context, target_rel_tokens),
            truncate_to_tokens(text_context, target_text_tokens),
        )

    async def process(self, input_data: ContextAssemblyInput) -> AssembledContext:
        """Assemble context from retrieval results.

        Args:
            input_data: Context assembly input

        Returns:
            Assembled context
        """
        # Format each section
        entity_context = self._format_entities_csv(input_data.entities)
        relationship_context = self._format_relationships_csv(input_data.relationships)
        text_context = self._format_texts(input_data.texts)

        # Combine based on mode
        entity_context, relationship_context, text_context = self._combine_contexts(
            entity_context, relationship_context, text_context, input_data.mode
        )

        # Enforce token limits
        entity_context, relationship_context, text_context = self._enforce_token_limits(
            entity_context, relationship_context, text_context, input_data.max_tokens
        )

        # Calculate total tokens
        total_tokens = (
            count_tokens(entity_context) +
            count_tokens(relationship_context) +
            count_tokens(text_context)
        )

        return AssembledContext(
            entity_context=entity_context,
            relationship_context=relationship_context,
            text_context=text_context,
            total_tokens=total_tokens,
            mode=input_data.mode,
        )
