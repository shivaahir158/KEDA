"""
Query engine that converts natural language questions to graph operations.

Two modes:
1. Rule-based: Pattern matching on question structure (fast, no LLM needed)
2. LLM-assisted: Uses an LLM to generate structured graph queries
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


class QuestionCategory(str, Enum):
    """Categories of engineering questions."""
    STRUCTURAL = "structural"          # 1-hop: "What ports does module X have?"
    CROSS_ARTIFACT = "cross_artifact"  # 2-3 hop: "What constraints apply to module X?"
    TEMPORAL = "temporal"              # Git: "What changed recently in module X?"
    WHAT_IF = "what_if"               # Impact: "What if we change signal width?"
    UNKNOWN = "unknown"


@dataclass
class GraphQuery:
    """A structured graph query derived from a natural language question."""
    category: QuestionCategory
    seed_entities: list[str] = field(default_factory=list)
    operation: str = ""  # "neighborhood", "path", "by_type", "chain", "module_context"
    params: dict[str, Any] = field(default_factory=dict)
    explanation: str = ""


# Pattern rules for rule-based query classification and entity extraction
_PATTERNS: list[tuple[re.Pattern, QuestionCategory, str, dict]] = [
    # Structural (1-hop)
    (re.compile(r'what\s+ports?\s+does\s+(?:module\s+)?(\w+)\s+have', re.I),
     QuestionCategory.STRUCTURAL, "neighborhood",
     {"hops": 1, "type_filter": {"Port"}}),

    (re.compile(r'what\s+(?:sub)?modules?\s+does\s+(\w+)\s+instantiate', re.I),
     QuestionCategory.STRUCTURAL, "neighborhood",
     {"hops": 1, "edge_filter": {"instantiates"}, "type_filter": {"Module"}}),

    (re.compile(r'what\s+(?:sub)?modules?\s+are\s+(?:instantiated\s+(?:in|by)\s+)?(\w+)', re.I),
     QuestionCategory.STRUCTURAL, "neighborhood",
     {"hops": 1, "edge_filter": {"instantiates"}, "type_filter": {"Module"}}),

    (re.compile(r'what\s+registers?\s+(?:are\s+in|does)\s+(?:module\s+)?(\w+)', re.I),
     QuestionCategory.STRUCTURAL, "neighborhood",
     {"hops": 1, "edge_filter": {"contains_register"}, "type_filter": {"Register"}}),

    (re.compile(r'what\s+parameters?\s+does\s+(?:module\s+)?(\w+)\s+have', re.I),
     QuestionCategory.STRUCTURAL, "neighborhood",
     {"hops": 1, "edge_filter": {"has_parameter"}, "type_filter": {"Parameter"}}),

    (re.compile(r'(?:list|show|what\s+are)\s+(?:all\s+)?(?:the\s+)?modules', re.I),
     QuestionCategory.STRUCTURAL, "by_type",
     {"node_type": "Module"}),

    (re.compile(r'(?:list|show|what\s+are)\s+(?:all\s+)?(?:the\s+)?clocks', re.I),
     QuestionCategory.STRUCTURAL, "by_type",
     {"node_type": "Clock"}),

    (re.compile(r'(?:list|show|what\s+are)\s+(?:all\s+)?(?:the\s+)?constraints', re.I),
     QuestionCategory.STRUCTURAL, "by_type",
     {"node_type": "Constraint"}),

    (re.compile(r'(?:list|show|what\s+are)\s+(?:all\s+)?(?:the\s+)?assertions', re.I),
     QuestionCategory.STRUCTURAL, "by_type",
     {"node_type": "Assertion"}),

    # Cross-artifact (2-3 hop)
    (re.compile(r'what\s+constraints?\s+(?:apply|affect|are\s+(?:on|for))\s+(?:(?:ports?\s+(?:of|in)\s+)?(?:module\s+)?)?(\w+)', re.I),
     QuestionCategory.CROSS_ARTIFACT, "chain",
     {"chain": [("has_port", "Port"), ("constrained_by", "Constraint")]}),

    (re.compile(r'what\s+clocks?\s+(?:drive|are\s+used\s+(?:by|in))\s+(?:registers?\s+(?:in|of)\s+)?(?:module\s+)?(\w+)', re.I),
     QuestionCategory.CROSS_ARTIFACT, "chain",
     {"chain": [("contains_register", "Register"), ("clocked_by", "Clock")]}),

    (re.compile(r'what\s+assertions?\s+(?:verify|cover|check)\s+(?:module\s+)?(\w+)', re.I),
     QuestionCategory.CROSS_ARTIFACT, "neighborhood",
     {"hops": 1, "edge_filter": {"verified_by", "verifies"}, "type_filter": {"Assertion"}}),

    (re.compile(r'what\s+tests?\s+(?:cover|test|verify)\s+(?:module\s+)?(\w+)', re.I),
     QuestionCategory.CROSS_ARTIFACT, "neighborhood",
     {"hops": 1, "edge_filter": {"covers"}, "type_filter": {"Test"}}),

    (re.compile(r'(?:how|what)\s+(?:is|are)\s+(\w+)\s+(?:and|connected|related)\s+(?:to\s+)?(\w+)', re.I),
     QuestionCategory.CROSS_ARTIFACT, "path",
     {}),

    (re.compile(r'(?:is\s+there\s+a\s+)?path\s+(?:from|between)\s+(\w+)\s+(?:to|and)\s+(\w+)', re.I),
     QuestionCategory.CROSS_ARTIFACT, "path",
     {}),

    (re.compile(r'(?:which|what)\s+(?:requirements?|specs?)\s+(?:are\s+)?(?:affected|related|linked)\s+(?:to|by)\s+(?:module\s+)?(\w+)', re.I),
     QuestionCategory.CROSS_ARTIFACT, "module_context",
     {}),

    # Temporal (Git)
    (re.compile(r'what\s+(?:changed|was\s+modified)\s+(?:recently\s+)?(?:in\s+)?(?:module\s+)?(\w+)', re.I),
     QuestionCategory.TEMPORAL, "neighborhood",
     {"hops": 1, "edge_filter": {"modifies", "modified_by"}, "type_filter": {"Commit"}}),

    (re.compile(r'(?:who|which\s+commits?)\s+(?:modified|changed|touched)\s+(?:module\s+)?(\w+)', re.I),
     QuestionCategory.TEMPORAL, "neighborhood",
     {"hops": 1, "edge_filter": {"modifies", "modified_by"}, "type_filter": {"Commit"}}),

    (re.compile(r'(?:history|changes?|log)\s+(?:of|for)\s+(?:module\s+)?(\w+)', re.I),
     QuestionCategory.TEMPORAL, "neighborhood",
     {"hops": 1, "edge_filter": {"modifies", "modified_by"}, "type_filter": {"Commit"}}),

    # What-if / Impact
    (re.compile(r'what\s+(?:would\s+)?(?:happen|if|is\s+affected|impact)\s+.*(?:change|modify|remove|add).*(\w+)', re.I),
     QuestionCategory.WHAT_IF, "module_context",
     {}),

    (re.compile(r'(?:impact|risk|effect)\s+of\s+(?:changing|modifying)\s+(?:module\s+)?(\w+)', re.I),
     QuestionCategory.WHAT_IF, "module_context",
     {}),

    (re.compile(r'(?:why\s+is|how\s+risky\s+is)\s+(?:module\s+)?(\w+)\s+(?:risky|critical|important)', re.I),
     QuestionCategory.WHAT_IF, "module_context",
     {}),
]

# Fallback: extract entity names from questions
_ENTITY_RE = re.compile(r'\b(?:module|port|clock|register|signal|constraint)\s+(\w+)', re.I)
_QUOTED_RE = re.compile(r'["\'](\w+)["\']')


class QueryEngine:
    """Convert natural language questions to structured graph queries.

    Supports rule-based classification and LLM-assisted query generation.
    """

    def __init__(self, graph: nx.DiGraph, design_name: str = "design"):
        self.graph = graph
        self.design_name = design_name

    def parse(self, question: str) -> GraphQuery:
        """Parse a natural language question into a GraphQuery using rules."""
        for pattern, category, operation, params in _PATTERNS:
            m = pattern.search(question)
            if m:
                entities = list(m.groups())
                query = GraphQuery(
                    category=category,
                    seed_entities=entities,
                    operation=operation,
                    params=dict(params),
                    explanation=f"Matched pattern: {pattern.pattern[:60]}...",
                )

                # For path queries, need two entities
                if operation == "path" and len(entities) >= 2:
                    query.params["source"] = entities[0]
                    query.params["target"] = entities[1]

                return query

        # Fallback: extract entities and use module_context
        entities = _ENTITY_RE.findall(question)
        if not entities:
            entities = _QUOTED_RE.findall(question)
        if not entities:
            # Try to find any word that matches a node name
            words = re.findall(r'\b\w+\b', question)
            for word in words:
                if len(word) >= 3 and word.lower() not in _STOP_WORDS:
                    node_id = f"{self.design_name}::{word}"
                    if node_id in self.graph:
                        entities.append(word)

        if entities:
            return GraphQuery(
                category=QuestionCategory.UNKNOWN,
                seed_entities=entities,
                operation="module_context",
                explanation="Fallback: extracted entities from question text",
            )

        return GraphQuery(
            category=QuestionCategory.UNKNOWN,
            explanation="Could not parse question into a graph query",
        )

    def generate_llm_query_prompt(self, question: str, schema: str) -> str:
        """Generate a prompt for an LLM to produce a structured graph query."""
        return f"""You are an EDA knowledge graph query assistant. Given the following question
about a hardware design, generate a structured graph query as JSON.

{schema}

Available operations:
- "neighborhood": Get k-hop neighbors of seed nodes. Params: hops, edge_filter, type_filter
- "path": Find paths between two nodes. Params: source, target
- "by_type": Get all nodes of a type. Params: node_type, attribute_filters
- "chain": Follow typed edge chain from seed. Params: chain (list of [relation, target_type])
- "module_context": Get full context for a module (ports, clocks, constraints, etc.)

Question: {question}

Respond with JSON only:
{{
  "category": "structural|cross_artifact|temporal|what_if",
  "seed_entities": ["entity1", ...],
  "operation": "neighborhood|path|by_type|chain|module_context",
  "params": {{...}}
}}"""

    def parse_llm_response(self, response_text: str) -> GraphQuery:
        """Parse an LLM's JSON response into a GraphQuery."""
        # Extract JSON from response
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if not json_match:
            return GraphQuery(
                category=QuestionCategory.UNKNOWN,
                explanation="Failed to parse LLM response as JSON",
            )

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return GraphQuery(
                category=QuestionCategory.UNKNOWN,
                explanation="Invalid JSON in LLM response",
            )

        category = QuestionCategory.UNKNOWN
        cat_str = data.get("category", "")
        for c in QuestionCategory:
            if c.value == cat_str:
                category = c
                break

        return GraphQuery(
            category=category,
            seed_entities=data.get("seed_entities", []),
            operation=data.get("operation", "module_context"),
            params=data.get("params", {}),
            explanation="LLM-generated query",
        )


# Stop words to skip when searching for entity names
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "what", "which", "who", "whom", "whose", "where", "when", "how", "why",
    "that", "this", "these", "those", "it", "its",
    "and", "but", "or", "nor", "not", "no", "so", "if", "then",
    "for", "of", "in", "on", "at", "to", "from", "by", "with",
    "about", "between", "through", "during", "before", "after",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "only", "own", "same",
    "module", "port", "clock", "register", "signal", "constraint",
    "design", "change", "modify", "affect", "impact",
    "list", "show", "get", "find", "tell", "give", "describe",
})
