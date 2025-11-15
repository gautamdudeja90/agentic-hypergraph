"""ExtractionAgent - Entity and relationship extraction with multi-round gleaning.

This agent uses LLM to extract entities and hyperedges from text chunks,
implementing HyperGraphRAG's multi-round gleaning approach.
"""

import sys
import re
from typing import List, Dict, Tuple
from collections import defaultdict
import asyncio

sys.path.insert(0, "/Users/gautamdudeja/repo/HyperGraphRAG")

from hypergraphrag.prompt import PROMPTS
from schemas.common import Entity, Hyperedge, TextChunk
from schemas.construction import ChunkExtractionInput, ExtractionResult
from agents.services.llm_service import LLMService


class ExtractionAgent:
    """Agent for entity and hyperedge extraction with gleaning."""

    def __init__(
        self,
        llm_service: LLMService,
        max_gleaning_rounds: int = 2,
        entity_types: List[str] = None,
        language: str = "English",
    ):
        """Initialize ExtractionAgent.

        Args:
            llm_service: LLM service for extraction
            max_gleaning_rounds: Number of gleaning iterations
            entity_types: Types of entities to extract
            language: Output language
        """
        self.llm_service = llm_service
        self.max_gleaning_rounds = max_gleaning_rounds
        self.entity_types = entity_types or PROMPTS["DEFAULT_ENTITY_TYPES"]
        self.language = language

        # Prompt config
        self.tuple_delimiter = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
        self.record_delimiter = PROMPTS["DEFAULT_RECORD_DELIMITER"]
        self.completion_delimiter = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]

    def _prepare_prompts(self) -> Dict[str, str]:
        """Prepare extraction prompts."""
        # Format examples
        examples = "\n".join(PROMPTS["entity_extraction_examples"])
        context_base = dict(
            tuple_delimiter=self.tuple_delimiter,
            record_delimiter=self.record_delimiter,
            completion_delimiter=self.completion_delimiter,
            entity_types=",".join(self.entity_types),
            language=self.language,
        )
        examples = examples.format(**context_base)

        # Main extraction prompt
        entity_extract_prompt = PROMPTS["entity_extraction"]
        initial_prompt = entity_extract_prompt.format(
            **context_base,
            examples=examples,
            input_text="{input_text}",
        )

        return {
            "initial": initial_prompt,
            "continue": PROMPTS["entiti_continue_extraction"],
            "if_loop": PROMPTS["entiti_if_loop_extraction"],
        }

    async def _extract_from_chunk(
        self, chunk: TextChunk, prompts: Dict[str, str]
    ) -> Tuple[List[Entity], List[Hyperedge]]:
        """Extract entities and hyperedges from a single chunk."""
        # Initial extraction
        initial_prompt = prompts["initial"].format(input_text=chunk.content)
        messages = [{"role": "user", "content": initial_prompt}]

        result = await self.llm_service.complete(messages)
        all_results = result

        # Gleaning rounds
        for round_idx in range(self.max_gleaning_rounds):
            messages.append({"role": "assistant", "content": result})
            messages.append({"role": "user", "content": prompts["continue"]})

            glean_result = await self.llm_service.complete(messages)
            messages.append({"role": "assistant", "content": glean_result})
            all_results += glean_result

            # Check if should continue gleaning
            if round_idx < self.max_gleaning_rounds - 1:
                messages_check = messages + [{"role": "user", "content": prompts["if_loop"]}]
                should_continue = await self.llm_service.complete(messages_check)
                should_continue = should_continue.strip().strip('"').strip("'").lower()
                if should_continue != "yes":
                    break

        # Parse results
        return self._parse_extraction_results(all_results, chunk.chunk_id)

    def _parse_extraction_results(
        self, results: str, chunk_id: str
    ) -> Tuple[List[Entity], List[Hyperedge]]:
        """Parse LLM extraction results into entities and hyperedges."""
        # Split by record delimiter
        records = re.split(
            f"{re.escape(self.record_delimiter)}|{re.escape(self.completion_delimiter)}",
            results,
        )

        entities_dict = defaultdict(lambda: {"descriptions": [], "weight": 0, "type": ""})
        hyperedges_list = []

        for record in records:
            # Extract tuple
            match = re.search(r"\((.*)\)", record)
            if not match:
                continue

            content = match.group(1)
            parts = content.split(self.tuple_delimiter)

            if len(parts) < 2:
                continue

            record_type = parts[0].strip('"').strip("'").strip()

            if record_type == "entity" and len(parts) >= 5:
                # Parse entity: ("entity", name, type, description, score)
                entity_name = parts[1].strip('"').strip("'").strip()
                entity_type = parts[2].strip('"').strip("'").strip()
                entity_desc = parts[3].strip('"').strip("'").strip()
                try:
                    score = float(parts[4].strip())
                except:
                    score = 50.0

                # Aggregate entity info
                entities_dict[entity_name]["descriptions"].append(entity_desc)
                entities_dict[entity_name]["weight"] += score / 100.0
                if not entities_dict[entity_name]["type"]:
                    entities_dict[entity_name]["type"] = entity_type

            elif record_type == "hyper-relation" and len(parts) >= 3:
                # Parse hyperedge: ("hyper-relation", description, score)
                description = parts[1].strip('"').strip("'").strip()
                try:
                    score = float(parts[2].strip())
                except:
                    score = 5.0

                # Extract entities mentioned in this relation
                # Simple approach: find entity names from entities_dict in the description
                connected_entities = []
                for entity_name in entities_dict.keys():
                    if entity_name.lower() in description.lower():
                        connected_entities.append(entity_name)

                if connected_entities:
                    hyperedge = Hyperedge(
                        entities=connected_entities,
                        relationship="described_in",
                        description=description,
                        weight=score / 10.0,
                        source_chunk_ids=[chunk_id],
                    )
                    hyperedges_list.append(hyperedge)

        # Convert entities dict to list
        entities_list = [
            Entity(
                name=name,
                type=info["type"] or "unknown",
                description=" ".join(info["descriptions"]),
                weight=info["weight"],
                source_chunk_ids=[chunk_id],
            )
            for name, info in entities_dict.items()
        ]

        return entities_list, hyperedges_list

    async def process(self, input_data: ChunkExtractionInput) -> ExtractionResult:
        """Process chunks to extract entities and hyperedges.

        Args:
            input_data: Input with chunks and config

        Returns:
            Extraction result with all entities and hyperedges
        """
        prompts = self._prepare_prompts()

        # Process chunks in parallel
        tasks = [
            self._extract_from_chunk(chunk, prompts) for chunk in input_data.chunks
        ]
        results = await asyncio.gather(*tasks)

        # Aggregate results
        all_entities = []
        all_hyperedges = []
        for entities, hyperedges in results:
            all_entities.extend(entities)
            all_hyperedges.extend(hyperedges)

        return ExtractionResult(
            entities=all_entities,
            hyperedges=all_hyperedges,
            chunks_processed=len(input_data.chunks),
        )
