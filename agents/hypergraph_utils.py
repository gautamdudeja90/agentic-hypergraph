"""Utilities and prompts from HyperGraphRAG for our agentic system.

This module contains the necessary functions and prompts copied from HyperGraphRAG
so our system can work standalone.
"""

import hashlib
from typing import Any, Callable, List
import numpy as np

# ===== Utility Functions =====

def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """Generate MD5 hash ID with optional prefix."""
    hash_id = hashlib.md5(content.encode()).hexdigest()
    return f"{prefix}{hash_id}" if prefix else hash_id


# ===== Type Definitions =====

# Embedding function type
EmbeddingFunc = Callable[[List[str]], Any]


# ===== Prompts from HyperGraphRAG =====

PROMPTS = {
    "DEFAULT_LANGUAGE": "English",
    "DEFAULT_TUPLE_DELIMITER": "<|>",
    "DEFAULT_RECORD_DELIMITER": "##",
    "DEFAULT_COMPLETION_DELIMITER": "<|COMPLETE|>",
    "DEFAULT_ENTITY_TYPES": ["organization", "person", "geo", "event", "category"],
}

# Entity extraction prompt
PROMPTS["entity_extraction"] = """-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
Use {language} as output language.

-Steps-
1. Divide the text into several complete knowledge segments.  For each knowledge segment, extract the following information:
-- knowledge_segment: A sentence that describes the context of the knowledge segment.
-- completeness_score: A score from 0 to 10 indicating the completeness of the knowledge segment.
Format each knowledge segment as ("hyper-relation"{tuple_delimiter}<knowledge_segment>{tuple_delimiter}<completeness_score>)

2. Identify all entities in each knowledge segment. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: Type of the entity.
- entity_description: Comprehensive description of the entity's attributes and activities.
- key_score: A score from 0 to 100 indicating the importance of the entity in the text.
Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>{tuple_delimiter}<key_score>)

3. Return output in {language} as a single list of all the entities and relationships identified in steps 1 and 2. Use **{record_delimiter}** as the list delimiter.

4. When finished, output {completion_delimiter}

######################
-Examples-
######################
{examples}

#############################
-Real Data-
######################
Text: {input_text}
######################
Output:
"""

# Continue extraction prompt
PROMPTS["entiti_continue_extraction"] = """MANY entities and relationships were missed in the last extraction. Remember to ONLY emit entities that match any of the previously extracted types. Add them below using the same format:
"""

# Check if should continue gleaning
PROMPTS["entiti_if_loop_extraction"] = """It appears some entities and relationships may have still been missed. Answer YES | NO if there are still entities or relationships that need to be added.
"""

# Simplified examples (condensed from HyperGraphRAG)
PROMPTS["entity_extraction_examples"] = [
    """Example 1:
Text: Alex is a researcher who works with Jordan on AI projects. They collaborate with Taylor who leads the team.
Output:
("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a researcher who works on AI projects."{tuple_delimiter}90){record_delimiter}
("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan works with Alex on AI projects."{tuple_delimiter}85){record_delimiter}
("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor leads the team."{tuple_delimiter}85){record_delimiter}
("hyper-relation"{tuple_delimiter}"Alex and Jordan collaborate on AI projects with Taylor leading the team."{tuple_delimiter}8){record_delimiter}
{completion_delimiter}"""
]

# Query-related prompts
PROMPTS["keywords_extraction"] = """---Role---
You are a helpful assistant tasked with identifying both simple and complex keywords in the user's query.

---Goal---
Given the query, list all relevant keywords. Be comprehensive and include:
- Simple keywords (single words)
- Complex keywords (phrases, multi-word concepts)
- Named entities
- Technical terms

---Instructions---
- Output keywords in {language}
- Use {tuple_delimiter} to separate keywords
- Format: <keyword1>{tuple_delimiter}<keyword2>{tuple_delimiter}...

######################
-Examples-
######################
Query: How does artificial intelligence impact healthcare?
Keywords: artificial intelligence{tuple_delimiter}AI{tuple_delimiter}healthcare{tuple_delimiter}medical technology{tuple_delimiter}machine learning{tuple_delimiter}patient care

#############################
Query: {query}
Keywords:"""

# Local search prompt (entity-centric)
PROMPTS["local_search"] = """---Role---
You are a helpful assistant responding to questions about documents.

---Goal---
Generate a response of {response_type} length using the provided context from the knowledge graph.

---Context---
{context_data}

---Query---
{query}

---Instructions---
- Use the context to provide an accurate, comprehensive answer
- Cite specific entities and relationships when relevant
- If context is insufficient, state what information is missing
- Response length: {response_type}

Answer:"""

# Global search prompt (relationship-centric)
PROMPTS["global_search"] = """---Role---
You are a helpful assistant responding to questions about documents.

---Goal---
Generate a response of {response_type} length using the relationship context provided.

---Relationship Context---
{context_data}

---Query---
{query}

---Instructions---
- Focus on relationships and connections in the data
- Synthesize information from multiple relationships
- Provide a holistic view based on the hypergraph structure
- Response length: {response_type}

Answer:"""
