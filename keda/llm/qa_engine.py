"""
GraphRAG QA Engine for KEDA.

Implements five comparison methods for answering engineering questions:
1. LLM-only: Provide graph schema + question to LLM (no retrieval)
2. KG retrieval: Structured graph query, return subgraph as answer (no LLM)
3. KG + LLM (KEDA-QA): Graph retrieval + LLM synthesis
4. Vector RAG: Embed graph nodes, retrieve by similarity, LLM synthesis
5. KG + LLM + Vector: Combined graph + vector retrieval + LLM synthesis
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from keda.llm.graph_retriever import GraphRetriever, SubgraphResult
from keda.llm.query_engine import QueryEngine, GraphQuery, QuestionCategory

logger = logging.getLogger(__name__)


@dataclass
class QAResult:
    """Result of answering a question."""
    question: str
    answer: str
    method: str
    evidence_nodes: set[str] = field(default_factory=set)
    retrieved_context: str = ""
    latency_ms: float = 0.0
    query_used: GraphQuery | None = None
    subgraph: SubgraphResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class QAEngine:
    """Main QA engine supporting multiple methods.

    Usage:
        engine = QAEngine(graph, design_name="picorv32")
        result = engine.answer("What ports does picorv32 have?", method="keda_qa")
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        design_name: str = "design",
        llm_client: Any = None,
        llm_model: str = "gpt-4o-mini",
        llm_provider: str = "openai",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.graph = graph
        self.design_name = design_name
        self.retriever = GraphRetriever(graph, design_name)
        self.query_engine = QueryEngine(graph, design_name)
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.llm_provider = llm_provider
        self._embeddings: dict[str, np.ndarray] | None = None
        self._embedding_model_name = embedding_model
        self._embedding_model = None
        self._node_texts: dict[str, str] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        method: str = "keda_qa",
        **kwargs,
    ) -> QAResult:
        """Answer a question using the specified method.

        Methods:
            llm_only: LLM with schema context only
            kg_only: Graph retrieval, structured answer (no LLM)
            keda_qa: Graph retrieval + LLM synthesis (primary method)
            vector_rag: Vector similarity retrieval + LLM
            keda_full: Graph + vector retrieval + LLM
        """
        t0 = time.time()
        dispatch = {
            "llm_only": self._llm_only,
            "kg_only": self._kg_only,
            "keda_qa": self._keda_qa,
            "vector_rag": self._vector_rag,
            "keda_full": self._keda_full,
        }
        fn = dispatch.get(method)
        if fn is None:
            raise ValueError(f"Unknown method: {method}. Choose from {list(dispatch)}")

        result = fn(question, **kwargs)
        result.latency_ms = (time.time() - t0) * 1000
        return result

    def answer_all_methods(self, question: str) -> dict[str, QAResult]:
        """Run all methods on a question for comparison."""
        results = {}
        for method in ["llm_only", "kg_only", "keda_qa", "vector_rag", "keda_full"]:
            try:
                results[method] = self.answer(question, method=method)
            except Exception as e:
                logger.warning("Method %s failed: %s", method, e)
                results[method] = QAResult(
                    question=question, answer=f"ERROR: {e}", method=method
                )
        return results

    # ------------------------------------------------------------------
    # Method 1: LLM-only (no retrieval)
    # ------------------------------------------------------------------

    def _llm_only(self, question: str, **kwargs) -> QAResult:
        """Answer using LLM with only the graph schema as context."""
        schema = self.retriever.get_graph_schema()

        prompt = f"""You are an expert hardware design engineer. Answer the following question
about the '{self.design_name}' design based on the knowledge graph schema below.

{schema}

Question: {question}

Answer concisely and precisely. If you cannot determine the answer from the schema alone,
say so. Do not fabricate specific details like port names, signal widths, or clock frequencies
unless you can infer them from the schema."""

        answer = self._call_llm(prompt)
        return QAResult(
            question=question,
            answer=answer,
            method="llm_only",
            retrieved_context=schema,
        )

    # ------------------------------------------------------------------
    # Method 2: KG-only (no LLM)
    # ------------------------------------------------------------------

    def _kg_only(self, question: str, **kwargs) -> QAResult:
        """Answer using structured graph retrieval only (no LLM)."""
        query = self.query_engine.parse(question)
        subgraph = self._execute_query(query)

        # Generate a structured answer from the subgraph
        answer = self._format_kg_answer(question, query, subgraph)

        return QAResult(
            question=question,
            answer=answer,
            method="kg_only",
            evidence_nodes=subgraph.node_ids,
            retrieved_context=subgraph.to_context_string(),
            query_used=query,
            subgraph=subgraph,
        )

    # ------------------------------------------------------------------
    # Method 3: KEDA-QA (KG + LLM) — primary method
    # ------------------------------------------------------------------

    def _keda_qa(self, question: str, **kwargs) -> QAResult:
        """Graph retrieval + LLM synthesis (primary KEDA method)."""
        query = self.query_engine.parse(question)
        subgraph = self._execute_query(query)
        context = subgraph.to_context_string()
        schema = self.retriever.get_graph_schema()

        prompt = f"""You are an expert hardware design engineer. Answer the question using
the knowledge graph context retrieved below. Ground your answer in the evidence provided.

Design: {self.design_name}

{schema}

Retrieved Context:
{context}

Question: {question}

Instructions:
- Use ONLY the information from the retrieved context to answer
- Cite specific nodes and relationships as evidence
- If the context is insufficient, say what information is missing
- Be precise about signal names, widths, and relationships"""

        answer = self._call_llm(prompt)
        return QAResult(
            question=question,
            answer=answer,
            method="keda_qa",
            evidence_nodes=subgraph.node_ids,
            retrieved_context=context,
            query_used=query,
            subgraph=subgraph,
        )

    # ------------------------------------------------------------------
    # Method 4: Vector RAG
    # ------------------------------------------------------------------

    def _vector_rag(self, question: str, top_k: int = 20, **kwargs) -> QAResult:
        """Embed question, retrieve similar nodes, LLM synthesis."""
        self._ensure_embeddings()

        # Embed the question
        q_emb = self._embed_text(question)

        # Find top-k similar nodes
        similarities = {}
        for node_id, emb in self._embeddings.items():
            sim = float(np.dot(q_emb, emb) / (
                np.linalg.norm(q_emb) * np.linalg.norm(emb) + 1e-8
            ))
            similarities[node_id] = sim

        ranked = sorted(similarities.items(), key=lambda x: -x[1])[:top_k]
        retrieved_ids = {nid for nid, _ in ranked}

        # Build context from retrieved nodes
        context_lines = ["Retrieved nodes (by vector similarity):"]
        for node_id, sim in ranked:
            data = self.graph.nodes.get(node_id, {})
            attrs = ", ".join(
                f"{k}={v}" for k, v in data.items()
                if v is not None and v != "" and k != "type"
            )
            context_lines.append(
                f"  [{data.get('type', '?')}] {node_id} (sim={sim:.3f}): {attrs}"
            )

        # Also add edges between retrieved nodes
        context_lines.append("\nEdges between retrieved nodes:")
        for u, v, data in self.graph.edges(data=True):
            if u in retrieved_ids and v in retrieved_ids:
                context_lines.append(
                    f"  {u} --{data.get('relation', '?')}--> {v}"
                )

        context = "\n".join(context_lines)

        prompt = f"""You are an expert hardware design engineer. Answer the question using
the context retrieved via vector similarity search from the design knowledge graph.

Design: {self.design_name}

{context}

Question: {question}

Instructions:
- Use the retrieved information to answer the question
- Note that retrieved context may include noise (irrelevant nodes)
- Be precise and cite specific evidence"""

        answer = self._call_llm(prompt)
        return QAResult(
            question=question,
            answer=answer,
            method="vector_rag",
            evidence_nodes=retrieved_ids,
            retrieved_context=context,
        )

    # ------------------------------------------------------------------
    # Method 5: KEDA Full (KG + Vector + LLM)
    # ------------------------------------------------------------------

    def _keda_full(self, question: str, top_k: int = 10, **kwargs) -> QAResult:
        """Combined graph retrieval + vector retrieval + LLM synthesis."""
        # Graph retrieval
        query = self.query_engine.parse(question)
        subgraph = self._execute_query(query)
        kg_context = subgraph.to_context_string()

        # Vector retrieval (supplementary)
        self._ensure_embeddings()
        q_emb = self._embed_text(question)
        similarities = {}
        for node_id, emb in self._embeddings.items():
            if node_id not in subgraph.node_ids:  # only add new nodes
                sim = float(np.dot(q_emb, emb) / (
                    np.linalg.norm(q_emb) * np.linalg.norm(emb) + 1e-8
                ))
                similarities[node_id] = sim

        ranked = sorted(similarities.items(), key=lambda x: -x[1])[:top_k]
        vector_ids = {nid for nid, _ in ranked}

        vector_lines = ["\nAdditional context from vector retrieval:"]
        for node_id, sim in ranked:
            data = self.graph.nodes.get(node_id, {})
            attrs = ", ".join(
                f"{k}={v}" for k, v in data.items()
                if v is not None and v != "" and k != "type"
            )
            vector_lines.append(
                f"  [{data.get('type', '?')}] {node_id} (sim={sim:.3f}): {attrs}"
            )
        vector_context = "\n".join(vector_lines)

        all_evidence = subgraph.node_ids | vector_ids
        schema = self.retriever.get_graph_schema()

        prompt = f"""You are an expert hardware design engineer. Answer the question using
both the structured graph context and vector-retrieved context below.

Design: {self.design_name}

{schema}

Graph-Retrieved Context (structured):
{kg_context}

{vector_context}

Question: {question}

Instructions:
- Prefer the structured graph context as primary evidence
- Use vector-retrieved context to supplement if needed
- Cite specific nodes and relationships
- Be precise about signal names, widths, and relationships"""

        answer = self._call_llm(prompt)
        return QAResult(
            question=question,
            answer=answer,
            method="keda_full",
            evidence_nodes=all_evidence,
            retrieved_context=kg_context + "\n" + vector_context,
            query_used=query,
            subgraph=subgraph,
        )

    # ------------------------------------------------------------------
    # Query execution
    # ------------------------------------------------------------------

    def _execute_query(self, query: GraphQuery) -> SubgraphResult:
        """Execute a parsed GraphQuery against the graph retriever."""
        if not query.seed_entities and query.operation not in ("by_type",):
            return SubgraphResult(summary="No seed entities identified")

        if query.operation == "neighborhood":
            return self.retriever.neighborhood(
                seed_nodes=query.seed_entities,
                hops=query.params.get("hops", 1),
                edge_filter=query.params.get("edge_filter"),
                type_filter=query.params.get("type_filter"),
            )
        elif query.operation == "path":
            src = query.params.get("source", query.seed_entities[0] if query.seed_entities else "")
            tgt = query.params.get("target", query.seed_entities[1] if len(query.seed_entities) > 1 else "")
            return self.retriever.find_paths(src, tgt)
        elif query.operation == "by_type":
            return self.retriever.by_type(
                node_type=query.params.get("node_type", "Module"),
                attribute_filters=query.params.get("attribute_filters"),
            )
        elif query.operation == "chain":
            chain = query.params.get("chain", [])
            # Convert list of lists to list of tuples
            chain = [(c[0], c[1]) if isinstance(c, list) else c for c in chain]
            return self.retriever.cross_artifact_chain(
                seed=query.seed_entities[0] if query.seed_entities else "",
                chain=chain,
            )
        elif query.operation == "module_context":
            entity = query.seed_entities[0] if query.seed_entities else ""
            return self.retriever.module_full_context(entity)
        else:
            # Default: neighborhood
            return self.retriever.neighborhood(
                seed_nodes=query.seed_entities, hops=2
            )

    # ------------------------------------------------------------------
    # KG-only answer formatting
    # ------------------------------------------------------------------

    def _format_kg_answer(
        self, question: str, query: GraphQuery, subgraph: SubgraphResult
    ) -> str:
        """Format a structured answer from a subgraph (no LLM)."""
        if not subgraph.nodes:
            return f"No relevant information found in the knowledge graph for: {question}"

        lines = [f"Query: {query.operation} on {query.seed_entities}"]
        lines.append(f"Category: {query.category.value}")
        lines.append(f"Found {len(subgraph.nodes)} nodes, {len(subgraph.edges)} edges")
        lines.append("")

        # Group nodes by type
        by_type: dict[str, list[dict]] = {}
        for n in subgraph.nodes:
            t = n.get("type", "Unknown")
            by_type.setdefault(t, []).append(n)

        for ntype, nodes in sorted(by_type.items()):
            lines.append(f"{ntype}s ({len(nodes)}):")
            for n in nodes[:20]:
                name = n.get("name", n.get("id", "?"))
                extra = []
                if n.get("direction"):
                    extra.append(f"direction={n['direction']}")
                if n.get("width") and n.get("width") != 1:
                    extra.append(f"width={n['width']}")
                if n.get("clock_signal"):
                    extra.append(f"clock={n['clock_signal']}")
                if n.get("constraint_type"):
                    extra.append(f"type={n['constraint_type']}")
                extra_str = f" ({', '.join(extra)})" if extra else ""
                lines.append(f"  - {name}{extra_str}")
            if len(nodes) > 20:
                lines.append(f"  ... and {len(nodes) - 20} more")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM integration
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Call the LLM API. Supports OpenAI and Anthropic providers with retry."""
        if self.llm_client is None:
            return self._no_llm_fallback(prompt)

        for attempt in range(max_retries):
            try:
                if self.llm_provider == "openai":
                    response = self.llm_client.chat.completions.create(
                        model=self.llm_model,
                        max_tokens=1024,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return response.choices[0].message.content
                else:
                    # Anthropic
                    response = self.llm_client.messages.create(
                        model=self.llm_model,
                        max_tokens=1024,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return response.content[0].text
            except Exception as e:
                logger.warning("LLM call attempt %d failed: %s", attempt + 1, e)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # exponential backoff
                else:
                    return f"LLM error after {max_retries} retries: {e}"

    def _no_llm_fallback(self, prompt: str) -> str:
        """When no LLM client is available, return the retrieved context directly."""
        # Extract the context section from the prompt
        lines = prompt.split("\n")
        context_start = None
        context_end = None
        for i, line in enumerate(lines):
            if "Retrieved Context:" in line or "Graph-Retrieved Context" in line:
                context_start = i + 1
            elif context_start and line.startswith("Question:"):
                context_end = i
                break

        if context_start and context_end:
            context = "\n".join(lines[context_start:context_end]).strip()
            return f"[No LLM configured — returning retrieved context]\n\n{context}"

        return "[No LLM client configured. Set llm_client=anthropic.Anthropic() to enable LLM synthesis.]"

    # ------------------------------------------------------------------
    # Embedding support
    # ------------------------------------------------------------------

    def _ensure_embeddings(self):
        """Lazily compute embeddings for all graph nodes."""
        if self._embeddings is not None:
            return

        logger.info("Computing node embeddings (first call)...")
        self._node_texts = {}
        texts = []
        node_ids = []

        for node_id, data in self.graph.nodes(data=True):
            text = self._node_to_text(node_id, data)
            self._node_texts[node_id] = text
            texts.append(text)
            node_ids.append(node_id)

        if not texts:
            self._embeddings = {}
            return

        try:
            from sentence_transformers import SentenceTransformer
            if self._embedding_model is None:
                self._embedding_model = SentenceTransformer(self._embedding_model_name)
            embeddings = self._embedding_model.encode(texts, show_progress_bar=False)
            self._embeddings = dict(zip(node_ids, embeddings))
            logger.info("Computed %d node embeddings", len(self._embeddings))
        except ImportError:
            logger.warning("sentence-transformers not installed, using random embeddings")
            rng = np.random.RandomState(42)
            self._embeddings = {
                nid: rng.randn(384).astype(np.float32) for nid in node_ids
            }

    def _embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        if self._embedding_model is not None:
            return self._embedding_model.encode([text], show_progress_bar=False)[0]
        # Fallback: hash-based pseudo-embedding
        h = hashlib.md5(text.encode()).hexdigest()
        rng = np.random.RandomState(int(h[:8], 16))
        return rng.randn(384).astype(np.float32)

    @staticmethod
    def _node_to_text(node_id: str, data: dict) -> str:
        """Convert a graph node to a text representation for embedding."""
        parts = [
            f"type:{data.get('type', 'Unknown')}",
            f"name:{data.get('name', node_id.split('::')[-1])}",
            f"id:{node_id}",
        ]
        for key in ["direction", "width", "clock_signal", "constraint_type",
                     "assertion_type", "property_text", "summary", "message",
                     "raw_line", "period", "file_path"]:
            val = data.get(key)
            if val is not None and val != "":
                parts.append(f"{key}:{val}")
        return " ".join(parts)
