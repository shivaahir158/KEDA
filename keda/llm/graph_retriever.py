"""
Graph-based retrieval for the KEDA knowledge graph.

Converts structured queries into subgraph extractions:
- 1-hop neighborhood queries (structural)
- Multi-hop traversal (cross-artifact)
- Pattern-based subgraph extraction
- Path finding between entities
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class SubgraphResult:
    """A retrieved subgraph with context for LLM consumption."""
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    query_used: str = ""

    def to_context_string(self) -> str:
        """Format the subgraph as a text context block for LLM prompting."""
        lines = []
        if self.summary:
            lines.append(f"Subgraph Summary: {self.summary}")
            lines.append("")

        if self.nodes:
            lines.append("Nodes:")
            for n in self.nodes:
                attrs = ", ".join(
                    f"{k}={v}" for k, v in n.items()
                    if k != "id" and v is not None and v != ""
                )
                lines.append(f"  [{n.get('type', '?')}] {n['id']}: {attrs}")

        if self.edges:
            lines.append("\nEdges:")
            for e in self.edges:
                lines.append(f"  {e['source']} --{e['relation']}--> {e['target']}")

        return "\n".join(lines)

    @property
    def node_ids(self) -> set[str]:
        return {n["id"] for n in self.nodes}


class GraphRetriever:
    """Retrieve relevant subgraphs from the KEDA knowledge graph.

    Supports multiple retrieval strategies:
    - neighborhood: k-hop neighborhood around seed nodes
    - path: shortest paths between entities
    - type_filter: all nodes of a given type with optional attribute filters
    - pattern: match structural patterns (e.g., Module->Port->Constraint chains)
    """

    def __init__(self, graph: nx.DiGraph, design_name: str = "design"):
        self.graph = graph
        self.design_name = design_name
        self._node_index: dict[str, str] = {}  # lowercase name -> node_id
        self._build_index()

    def _build_index(self):
        """Build a name-to-node index for entity resolution."""
        for node_id, data in self.graph.nodes(data=True):
            name = data.get("name", "")
            if name:
                self._node_index[name.lower()] = node_id
            # Also index by the last part of the node_id
            parts = node_id.split("::")
            if len(parts) >= 2:
                self._node_index[parts[-1].lower()] = node_id

    def resolve_entity(self, name: str) -> str | None:
        """Resolve a name or partial name to a node ID.

        Prefers Module nodes over other types when multiple matches exist.
        """
        G = self.graph

        # Try with design prefix first (most common case)
        prefixed = f"{self.design_name}::{name}"
        if prefixed in G:
            return prefixed

        # Exact match in index
        lower = name.lower()
        if lower in self._node_index:
            candidate = self._node_index[lower]
            # Check if there's also a Module node with this name
            mod_id = f"{self.design_name}::{name}"
            if mod_id in G and G.nodes[mod_id].get("type") == "Module":
                return mod_id
            return candidate

        # Substring match
        matches = [
            nid for key, nid in self._node_index.items()
            if lower in key or key in lower
        ]
        if not matches:
            return None

        # Prefer Module nodes
        module_matches = [m for m in matches if G.nodes.get(m, {}).get("type") == "Module"]
        if module_matches:
            # Prefer exact suffix match among modules
            for m in module_matches:
                if m.endswith(f"::{name}"):
                    return m
            return module_matches[0]

        if len(matches) == 1:
            return matches[0]
        # Prefer exact suffix match
        for m in matches:
            if m.endswith(f"::{name}"):
                return m
        return matches[0]

    def neighborhood(
        self,
        seed_nodes: list[str],
        hops: int = 1,
        edge_filter: set[str] | None = None,
        type_filter: set[str] | None = None,
        max_nodes: int = 100,
    ) -> SubgraphResult:
        """Extract the k-hop neighborhood around seed nodes."""
        G = self.graph
        visited: set[str] = set()
        frontier = set()

        for node in seed_nodes:
            resolved = self.resolve_entity(node) if node not in G else node
            if resolved and resolved in G:
                visited.add(resolved)
                frontier.add(resolved)

        if not frontier:
            return SubgraphResult(summary=f"No matching nodes found for: {seed_nodes}")

        for _ in range(hops):
            next_frontier: set[str] = set()
            for node in frontier:
                for _, neighbor, data in G.out_edges(node, data=True):
                    rel = data.get("relation", "")
                    if edge_filter and rel not in edge_filter:
                        continue
                    ntype = G.nodes.get(neighbor, {}).get("type", "")
                    if type_filter and ntype not in type_filter:
                        continue
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_frontier.add(neighbor)
                for predecessor, _, data in G.in_edges(node, data=True):
                    rel = data.get("relation", "")
                    if edge_filter and rel not in edge_filter:
                        continue
                    ntype = G.nodes.get(predecessor, {}).get("type", "")
                    if type_filter and ntype not in type_filter:
                        continue
                    if predecessor not in visited:
                        visited.add(predecessor)
                        next_frontier.add(predecessor)
                if len(visited) >= max_nodes:
                    break
            frontier = next_frontier
            if len(visited) >= max_nodes:
                break

        return self._build_subgraph(visited, f"{hops}-hop neighborhood of {seed_nodes}")

    def find_paths(
        self,
        source: str,
        target: str,
        max_length: int = 5,
    ) -> SubgraphResult:
        """Find shortest paths between two entities."""
        G = self.graph
        src = self.resolve_entity(source) if source not in G else source
        tgt = self.resolve_entity(target) if target not in G else target

        if not src or src not in G:
            return SubgraphResult(summary=f"Source entity not found: {source}")
        if not tgt or tgt not in G:
            return SubgraphResult(summary=f"Target entity not found: {target}")

        visited: set[str] = set()
        try:
            for path in nx.all_shortest_paths(G, src, tgt):
                if len(path) - 1 > max_length:
                    break
                visited.update(path)
        except nx.NetworkXNoPath:
            # Try undirected
            try:
                UG = G.to_undirected()
                for path in nx.all_shortest_paths(UG, src, tgt):
                    if len(path) - 1 > max_length:
                        break
                    visited.update(path)
            except nx.NetworkXNoPath:
                return SubgraphResult(
                    summary=f"No path found between {source} and {target}"
                )

        return self._build_subgraph(
            visited, f"Paths between {source} and {target}"
        )

    def by_type(
        self,
        node_type: str,
        attribute_filters: dict[str, Any] | None = None,
        max_nodes: int = 50,
    ) -> SubgraphResult:
        """Retrieve all nodes of a given type, optionally filtered by attributes."""
        G = self.graph
        matched: set[str] = set()

        for node_id, data in G.nodes(data=True):
            if data.get("type", "") != node_type:
                continue
            if attribute_filters:
                match = all(
                    str(data.get(k, "")).lower() == str(v).lower()
                    for k, v in attribute_filters.items()
                )
                if not match:
                    continue
            matched.add(node_id)
            if len(matched) >= max_nodes:
                break

        filters_str = f" with {attribute_filters}" if attribute_filters else ""
        return self._build_subgraph(
            matched, f"All {node_type} nodes{filters_str}"
        )

    def cross_artifact_chain(
        self,
        seed: str,
        chain: list[tuple[str, str]],
        max_per_hop: int = 20,
    ) -> SubgraphResult:
        """Follow a typed edge chain from a seed node.

        Args:
            seed: Starting node name or ID.
            chain: List of (relation, target_type) tuples defining the traversal.
                   E.g., [("has_port", "Port"), ("constrained_by", "Constraint")]
        """
        G = self.graph
        resolved = self.resolve_entity(seed) if seed not in G else seed
        if not resolved or resolved not in G:
            return SubgraphResult(summary=f"Seed entity not found: {seed}")

        visited: set[str] = {resolved}
        current_frontier = {resolved}

        for relation, target_type in chain:
            next_frontier: set[str] = set()
            for node in current_frontier:
                for _, neighbor, data in G.out_edges(node, data=True):
                    if data.get("relation") != relation:
                        continue
                    ntype = G.nodes.get(neighbor, {}).get("type", "")
                    if target_type and ntype != target_type:
                        continue
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
                    if len(next_frontier) >= max_per_hop:
                        break
            current_frontier = next_frontier

        chain_str = " -> ".join(f"--{r}--> {t}" for r, t in chain)
        return self._build_subgraph(
            visited, f"Chain from {seed}: {chain_str}"
        )

    def module_full_context(self, module_name: str) -> SubgraphResult:
        """Get comprehensive context for a module: ports, registers, clocks,
        constraints, assertions, tests, and parent/child modules."""
        G = self.graph
        resolved = self.resolve_entity(module_name)
        if not resolved or resolved not in G:
            return SubgraphResult(summary=f"Module not found: {module_name}")

        visited: set[str] = {resolved}

        # Direct relationships (1-hop)
        for _, neighbor, data in G.out_edges(resolved, data=True):
            visited.add(neighbor)
        for predecessor, _, data in G.in_edges(resolved, data=True):
            visited.add(predecessor)

        # 2-hop for cross-artifact: Port->Constraint, Register->Clock
        second_hop = set()
        for node in list(visited):
            ntype = G.nodes.get(node, {}).get("type", "")
            if ntype in ("Port", "Register", "Clock"):
                for _, n2, data in G.out_edges(node, data=True):
                    second_hop.add(n2)
                for p2, _, data in G.in_edges(node, data=True):
                    second_hop.add(p2)
        visited.update(second_hop)

        return self._build_subgraph(visited, f"Full context for module {module_name}")

    def _build_subgraph(self, node_ids: set[str], summary: str) -> SubgraphResult:
        """Build a SubgraphResult from a set of node IDs."""
        G = self.graph
        nodes = []
        edges = []

        for nid in node_ids:
            data = dict(G.nodes.get(nid, {}))
            data["id"] = nid
            # Remove overly verbose attributes
            for key in ["property_text", "raw_line"]:
                if key in data and data[key] and len(str(data[key])) > 200:
                    data[key] = str(data[key])[:200] + "..."
            nodes.append(data)

        for u, v, data in G.edges(data=True):
            if u in node_ids and v in node_ids:
                edges.append({
                    "source": u,
                    "target": v,
                    "relation": data.get("relation", "unknown"),
                })

        return SubgraphResult(
            nodes=nodes,
            edges=edges,
            summary=summary,
        )

    def get_graph_schema(self) -> str:
        """Return a summary of the graph's schema (node types, edge types, counts)."""
        G = self.graph
        type_counts: dict[str, int] = {}
        rel_counts: dict[str, int] = {}

        for _, data in G.nodes(data=True):
            t = data.get("type", "Unknown")
            type_counts[t] = type_counts.get(t, 0) + 1

        for _, _, data in G.edges(data=True):
            r = data.get("relation", "unknown")
            rel_counts[r] = rel_counts.get(r, 0) + 1

        lines = [f"Knowledge Graph Schema for '{self.design_name}':"]
        lines.append(f"  Total nodes: {G.number_of_nodes()}")
        lines.append(f"  Total edges: {G.number_of_edges()}")
        lines.append("\n  Node types:")
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
            lines.append(f"    {t}: {c}")
        lines.append("\n  Edge types:")
        for r, c in sorted(rel_counts.items(), key=lambda x: -x[1]):
            lines.append(f"    {r}: {c}")

        return "\n".join(lines)
