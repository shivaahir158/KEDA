"""
Change-impact analysis engine for the KEDA knowledge graph.

Given a set of changed nodes (e.g., modules modified by a commit), determines
which artifacts across all categories are affected, using multiple strategies:

1. Graph traversal (BFS/weighted BFS)
2. Personalized PageRank
3. Lexical baseline (grep-like keyword matching)
4. Embedding-based retrieval stub

Also implements the risk propagation model from the research plan:
  r(v) = max over paths: product of edge weights * alpha^hops
"""

from __future__ import annotations

import heapq
import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Edge weight configuration
# ---------------------------------------------------------------------------

DEFAULT_EDGE_WEIGHTS: dict[str, float] = {
    "instantiates":       1.0,
    "depended_by":        0.8,
    "has_port":           1.0,
    "has_instance":       1.0,
    "contains_register":  1.0,
    "clocked_by":         0.8,
    "constrained_by":     0.7,
    "applies_to":         0.7,
    "references_clock":   0.6,
    "derived_from":       0.8,
    "verifies":           0.7,
    "verified_by":        0.7,
    "covers":             0.6,
    "modifies":           1.0,
    "modified_by":        1.0,
    "drives":             0.9,
    "instance_of":        0.9,
    "has_parameter":      0.8,
    "parameter_of":       0.8,
    "uses_clock":         0.7,
}

# Weights can be overridden per change type
CHANGE_TYPE_WEIGHT_OVERRIDES: dict[str, dict[str, float]] = {
    "clock": {
        "clocked_by": 1.0,
        "constrained_by": 0.9,
        "derived_from": 1.0,
    },
    "width": {
        "drives": 1.0,
        "has_port": 1.0,
        "connected_to": 1.0,
    },
    "parameter": {
        "has_parameter": 1.0,
        "parameter_of": 1.0,
        "drives": 0.7,
    },
}


class ArtifactType(str, Enum):
    """Categories of design artifacts in the impact set."""
    MODULE = "Module"
    PORT = "Port"
    REGISTER = "Register"
    CLOCK = "Clock"
    CONSTRAINT = "Constraint"
    ASSERTION = "Assertion"
    TEST = "Test"
    COMMIT = "Commit"
    PARAMETER = "Parameter"
    INSTANCE = "Instance"
    NET = "Net"


@dataclass
class ImpactedArtifact:
    """A single artifact identified as impacted by a change."""
    node_id: str
    artifact_type: str
    name: str
    risk_score: float
    hop_distance: int
    path: list[str]          # sequence of node IDs from source to this artifact
    path_relations: list[str]  # edge relations along the path
    reason: str = ""         # human-readable explanation


@dataclass
class ImpactResult:
    """Full result of a change-impact analysis."""
    changed_nodes: list[str]
    impacted: list[ImpactedArtifact] = field(default_factory=list)
    method: str = ""

    # Convenience access by artifact type
    @property
    def by_type(self) -> dict[str, list[ImpactedArtifact]]:
        groups: dict[str, list[ImpactedArtifact]] = {}
        for a in self.impacted:
            groups.setdefault(a.artifact_type, []).append(a)
        return groups

    @property
    def modules(self) -> list[ImpactedArtifact]:
        return [a for a in self.impacted if a.artifact_type == "Module"]

    @property
    def constraints(self) -> list[ImpactedArtifact]:
        return [a for a in self.impacted if a.artifact_type == "Constraint"]

    @property
    def assertions(self) -> list[ImpactedArtifact]:
        return [a for a in self.impacted if a.artifact_type == "Assertion"]

    @property
    def tests(self) -> list[ImpactedArtifact]:
        return [a for a in self.impacted if a.artifact_type == "Test"]

    @property
    def clocks(self) -> list[ImpactedArtifact]:
        return [a for a in self.impacted if a.artifact_type == "Clock"]

    @property
    def registers(self) -> list[ImpactedArtifact]:
        return [a for a in self.impacted if a.artifact_type == "Register"]

    def top_k(self, k: int = 10) -> list[ImpactedArtifact]:
        return sorted(self.impacted, key=lambda a: -a.risk_score)[:k]

    def at_hop(self, hop: int) -> list[ImpactedArtifact]:
        return [a for a in self.impacted if a.hop_distance == hop]

    def artifact_ids(self, artifact_type: str | None = None) -> set[str]:
        if artifact_type:
            return {a.node_id for a in self.impacted if a.artifact_type == artifact_type}
        return {a.node_id for a in self.impacted}


# ---------------------------------------------------------------------------
# Traversal directions — which edges to follow from each node type
# ---------------------------------------------------------------------------

# For impact analysis, we want to propagate "outward" from changed modules:
#   Module -> (depended_by) -> parent modules
#   Module -> (has_port) -> ports -> (constrained_by) -> constraints
#   Module -> (contains_register) -> registers -> (clocked_by) -> clocks
#   Module -> (verified_by) -> assertions
#   Module -> (instantiates) -> child modules
#   Clock -> (applies_to from constraints) -> constraints
#   etc.
#
# We use BOTH directions on the undirected semantic relationship.
# The directed graph already has edges in both directions for key relations.

# Relations to follow for forward impact propagation
PROPAGATION_RELATIONS = frozenset({
    "instantiates", "depended_by",
    "has_port", "has_instance", "contains_register",
    "clocked_by", "constrained_by", "applies_to",
    "references_clock", "derived_from",
    "verifies", "verified_by", "covers",
    "modifies", "modified_by",
    "drives", "instance_of",
    "has_parameter", "parameter_of",
    "uses_clock",
})


class ChangeImpactAnalyzer:
    """Analyze the impact of design changes across the knowledge graph.

    Implements multiple strategies:
    - weighted_bfs: Weighted BFS with exponential decay (primary method)
    - bfs: Unweighted BFS to fixed depth
    - pagerank: Personalized PageRank from changed nodes
    - structural: Module-hierarchy-only traversal (baseline)
    - lexical: Keyword-based matching (baseline)
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        edge_weights: dict[str, float] | None = None,
    ):
        self.graph = graph
        self.edge_weights = edge_weights or DEFAULT_EDGE_WEIGHTS

    # ------------------------------------------------------------------
    # Primary method: weighted BFS with exponential decay
    # ------------------------------------------------------------------

    def weighted_bfs(
        self,
        changed_nodes: list[str],
        alpha: float = 0.7,
        max_depth: int = 5,
        min_risk: float = 0.01,
        change_type: str | None = None,
        exclude_types: set[str] | None = None,
    ) -> ImpactResult:
        """Weighted BFS with exponential decay per hop.

        Risk score for node v:
          r(v) = max over all paths P from any changed node s to v:
                 product(edge_weight(e) for e in P) * alpha^|P|

        Args:
            changed_nodes: Node IDs of directly changed artifacts.
            alpha: Decay factor per hop (0 < alpha <= 1).
            max_depth: Maximum traversal depth.
            min_risk: Minimum risk score to include in results.
            change_type: Optional change category for weight overrides.
            exclude_types: Node types to skip during traversal.
        """
        weights = dict(self.edge_weights)
        if change_type and change_type in CHANGE_TYPE_WEIGHT_OVERRIDES:
            weights.update(CHANGE_TYPE_WEIGHT_OVERRIDES[change_type])

        exclude = exclude_types or set()
        G = self.graph

        # Best risk score, hop distance, and path to each node
        best_risk: dict[str, float] = {}
        best_hop: dict[str, int] = {}
        best_path: dict[str, list[str]] = {}
        best_relations: dict[str, list[str]] = {}

        # Initialize changed nodes
        for node in changed_nodes:
            if node not in G:
                logger.warning("Changed node '%s' not in graph, skipping", node)
                continue
            best_risk[node] = 1.0
            best_hop[node] = 0
            best_path[node] = [node]
            best_relations[node] = []

        # Priority queue: (-risk, node_id, depth, path, relations)
        pq: list[tuple[float, str, int, list[str], list[str]]] = []
        for node in changed_nodes:
            if node in G:
                heapq.heappush(pq, (-1.0, node, 0, [node], []))

        visited_at_risk: dict[str, float] = {}

        while pq:
            neg_risk, node, depth, path, relations = heapq.heappop(pq)
            current_risk = -neg_risk

            # Skip if we've already processed this node at a higher risk
            if node in visited_at_risk and visited_at_risk[node] >= current_risk:
                continue
            visited_at_risk[node] = current_risk

            if depth >= max_depth:
                continue

            # Expand neighbors
            for _, neighbor, edge_data in G.out_edges(node, data=True):
                relation = edge_data.get("relation", "")
                if relation not in PROPAGATION_RELATIONS:
                    continue

                neighbor_type = G.nodes.get(neighbor, {}).get("type", "")
                if neighbor_type in exclude:
                    continue

                # Skip if neighbor is already in the current path (avoid cycles)
                if neighbor in path:
                    continue

                w = weights.get(relation, 0.5)
                new_risk = current_risk * w * alpha
                if new_risk < min_risk:
                    continue

                if new_risk > best_risk.get(neighbor, 0):
                    best_risk[neighbor] = new_risk
                    best_hop[neighbor] = depth + 1
                    new_path = path + [neighbor]
                    new_relations = relations + [relation]
                    best_path[neighbor] = new_path
                    best_relations[neighbor] = new_relations

                    heapq.heappush(pq, (-new_risk, neighbor, depth + 1,
                                        new_path, new_relations))

        # Build result (exclude the changed nodes themselves)
        result = ImpactResult(
            changed_nodes=list(changed_nodes),
            method="weighted_bfs",
        )
        changed_set = set(changed_nodes)

        for node_id, risk in best_risk.items():
            if node_id in changed_set:
                continue
            node_data = G.nodes.get(node_id, {})
            artifact_type = node_data.get("type", "Unknown")
            name = node_data.get("name", node_id.split("::")[-1])

            result.impacted.append(ImpactedArtifact(
                node_id=node_id,
                artifact_type=artifact_type,
                name=name or node_id,
                risk_score=risk,
                hop_distance=best_hop[node_id],
                path=best_path[node_id],
                path_relations=best_relations[node_id],
                reason=self._build_reason(best_path[node_id], best_relations[node_id]),
            ))

        result.impacted.sort(key=lambda a: -a.risk_score)
        return result

    # ------------------------------------------------------------------
    # Unweighted BFS
    # ------------------------------------------------------------------

    def bfs(
        self,
        changed_nodes: list[str],
        max_depth: int = 3,
        exclude_types: set[str] | None = None,
    ) -> ImpactResult:
        """Simple unweighted BFS from changed nodes.

        All discovered nodes get risk_score = 1.0 / (hop + 1).
        """
        exclude = exclude_types or set()
        G = self.graph

        visited: dict[str, int] = {}  # node -> hop distance
        parent_path: dict[str, list[str]] = {}
        parent_rels: dict[str, list[str]] = {}

        frontier = set()
        for node in changed_nodes:
            if node in G:
                visited[node] = 0
                parent_path[node] = [node]
                parent_rels[node] = []
                frontier.add(node)

        for depth in range(1, max_depth + 1):
            next_frontier = set()
            for node in frontier:
                for _, neighbor, edge_data in G.out_edges(node, data=True):
                    relation = edge_data.get("relation", "")
                    if relation not in PROPAGATION_RELATIONS:
                        continue
                    if neighbor in visited:
                        continue
                    neighbor_type = G.nodes.get(neighbor, {}).get("type", "")
                    if neighbor_type in exclude:
                        continue

                    visited[neighbor] = depth
                    parent_path[neighbor] = parent_path[node] + [neighbor]
                    parent_rels[neighbor] = parent_rels[node] + [relation]
                    next_frontier.add(neighbor)
            frontier = next_frontier

        result = ImpactResult(changed_nodes=list(changed_nodes), method="bfs")
        changed_set = set(changed_nodes)
        for node_id, hop in visited.items():
            if node_id in changed_set:
                continue
            node_data = G.nodes.get(node_id, {})
            result.impacted.append(ImpactedArtifact(
                node_id=node_id,
                artifact_type=node_data.get("type", "Unknown"),
                name=node_data.get("name", node_id.split("::")[-1]) or node_id,
                risk_score=1.0 / (hop + 1),
                hop_distance=hop,
                path=parent_path[node_id],
                path_relations=parent_rels[node_id],
                reason=self._build_reason(parent_path[node_id], parent_rels[node_id]),
            ))
        result.impacted.sort(key=lambda a: -a.risk_score)
        return result

    # ------------------------------------------------------------------
    # Personalized PageRank
    # ------------------------------------------------------------------

    def pagerank(
        self,
        changed_nodes: list[str],
        alpha: float = 0.85,
        min_score: float = 1e-4,
    ) -> ImpactResult:
        """Personalized PageRank from changed nodes.

        Uses the NetworkX implementation with personalization vector
        concentrated on the changed nodes.
        """
        G = self.graph
        valid = [n for n in changed_nodes if n in G]
        if not valid:
            return ImpactResult(changed_nodes=list(changed_nodes), method="pagerank")

        personalization = {n: 0.0 for n in G.nodes()}
        for n in valid:
            personalization[n] = 1.0 / len(valid)

        try:
            pr = nx.pagerank(
                G, alpha=alpha,
                personalization=personalization,
                max_iter=200,
                tol=1e-6,
            )
        except nx.PowerIterationFailedConvergence:
            logger.warning("PageRank did not converge, using partial results")
            pr = nx.pagerank(G, alpha=alpha, personalization=personalization,
                             max_iter=500, tol=1e-4)

        # Normalize so max = 1.0
        max_pr = max(pr.values()) if pr else 1.0
        if max_pr == 0:
            max_pr = 1.0

        result = ImpactResult(changed_nodes=list(changed_nodes), method="pagerank")
        changed_set = set(changed_nodes)
        for node_id, score in pr.items():
            normalized = score / max_pr
            if node_id in changed_set:
                continue
            if normalized < min_score:
                continue
            node_data = G.nodes.get(node_id, {})

            # Approximate hop distance using shortest path
            min_dist = float("inf")
            for s in valid:
                try:
                    dist = nx.shortest_path_length(G, s, node_id)
                    min_dist = min(min_dist, dist)
                except nx.NetworkXNoPath:
                    pass
            hop = int(min_dist) if min_dist != float("inf") else -1

            # Get shortest path for explanation
            path = []
            rels = []
            if hop > 0:
                for s in valid:
                    try:
                        sp = nx.shortest_path(G, s, node_id)
                        if len(sp) - 1 == hop:
                            path = sp
                            rels = [
                                G[sp[i]][sp[i+1]].get("relation", "?")
                                for i in range(len(sp) - 1)
                            ]
                            break
                    except nx.NetworkXNoPath:
                        pass

            result.impacted.append(ImpactedArtifact(
                node_id=node_id,
                artifact_type=node_data.get("type", "Unknown"),
                name=node_data.get("name", node_id.split("::")[-1]) or node_id,
                risk_score=normalized,
                hop_distance=hop,
                path=path,
                path_relations=rels,
            ))

        result.impacted.sort(key=lambda a: -a.risk_score)
        return result

    # ------------------------------------------------------------------
    # Structural baseline (module hierarchy only)
    # ------------------------------------------------------------------

    def structural(
        self,
        changed_nodes: list[str],
        max_depth: int = 5,
    ) -> ImpactResult:
        """Module-hierarchy-only traversal (baseline).

        Only follows instantiates/depended_by edges between Module nodes.
        """
        structural_relations = {"instantiates", "depended_by"}
        G = self.graph

        visited: dict[str, int] = {}
        parent_path: dict[str, list[str]] = {}
        parent_rels: dict[str, list[str]] = {}

        frontier = set()
        for node in changed_nodes:
            if node in G and G.nodes[node].get("type") == "Module":
                visited[node] = 0
                parent_path[node] = [node]
                parent_rels[node] = []
                frontier.add(node)

        for depth in range(1, max_depth + 1):
            next_frontier = set()
            for node in frontier:
                for _, neighbor, edge_data in G.out_edges(node, data=True):
                    relation = edge_data.get("relation", "")
                    if relation not in structural_relations:
                        continue
                    if G.nodes.get(neighbor, {}).get("type") != "Module":
                        continue
                    if neighbor in visited:
                        continue
                    visited[neighbor] = depth
                    parent_path[neighbor] = parent_path[node] + [neighbor]
                    parent_rels[neighbor] = parent_rels[node] + [relation]
                    next_frontier.add(neighbor)
            frontier = next_frontier

        result = ImpactResult(changed_nodes=list(changed_nodes), method="structural")
        changed_set = set(changed_nodes)
        for node_id, hop in visited.items():
            if node_id in changed_set:
                continue
            node_data = G.nodes.get(node_id, {})
            result.impacted.append(ImpactedArtifact(
                node_id=node_id,
                artifact_type="Module",
                name=node_data.get("name", node_id.split("::")[-1]) or node_id,
                risk_score=1.0 / (hop + 1),
                hop_distance=hop,
                path=parent_path[node_id],
                path_relations=parent_rels[node_id],
                reason=self._build_reason(parent_path[node_id], parent_rels[node_id]),
            ))
        result.impacted.sort(key=lambda a: -a.risk_score)
        return result

    # ------------------------------------------------------------------
    # Lexical baseline
    # ------------------------------------------------------------------

    def lexical(
        self,
        changed_nodes: list[str],
    ) -> ImpactResult:
        """Lexical/keyword-based impact analysis (baseline).

        Collects names from changed nodes and finds other nodes whose names
        or attributes contain those keywords.
        """
        G = self.graph
        changed_set = set(changed_nodes)

        # Collect keywords from changed nodes
        keywords = set()
        for node in changed_nodes:
            data = G.nodes.get(node, {})
            name = data.get("name", "")
            if name:
                keywords.add(name.lower())
                # Also add sub-tokens (e.g., "uart_top" -> {"uart", "top"})
                keywords.update(name.lower().split("_"))

        # Remove very short / common tokens
        keywords = {k for k in keywords if len(k) >= 3}

        if not keywords:
            return ImpactResult(changed_nodes=list(changed_nodes), method="lexical")

        result = ImpactResult(changed_nodes=list(changed_nodes), method="lexical")
        for node_id, data in G.nodes(data=True):
            if node_id in changed_set:
                continue

            # Build a searchable text from node attributes
            searchable = " ".join(str(v) for v in [
                data.get("name", ""),
                data.get("raw_line", ""),
                data.get("property_text", ""),
                data.get("summary", ""),
                data.get("message", ""),
                data.get("constraint_type", ""),
            ]).lower()

            # Count keyword matches
            matches = sum(1 for kw in keywords if kw in searchable)
            if matches > 0:
                score = min(1.0, matches / len(keywords))
                result.impacted.append(ImpactedArtifact(
                    node_id=node_id,
                    artifact_type=data.get("type", "Unknown"),
                    name=data.get("name", node_id.split("::")[-1]) or node_id,
                    risk_score=score,
                    hop_distance=-1,
                    path=[],
                    path_relations=[],
                    reason=f"Lexical match ({matches} keywords)",
                ))

        result.impacted.sort(key=lambda a: -a.risk_score)
        return result

    # ------------------------------------------------------------------
    # Convenience: run from commit
    # ------------------------------------------------------------------

    def from_commit(
        self,
        commit_node_id: str,
        method: str = "weighted_bfs",
        **kwargs,
    ) -> ImpactResult:
        """Analyze impact starting from a commit node.

        Resolves the commit's modified modules and runs the specified method.
        """
        G = self.graph
        if commit_node_id not in G:
            raise ValueError(f"Commit node '{commit_node_id}' not in graph")

        # Find modules modified by this commit
        modified_modules = [
            v for _, v, d in G.out_edges(commit_node_id, data=True)
            if d.get("relation") == "modifies"
        ]

        if not modified_modules:
            logger.warning("Commit %s has no modifies edges", commit_node_id)
            return ImpactResult(changed_nodes=[commit_node_id], method=method)

        runner = getattr(self, method)
        return runner(changed_nodes=modified_modules, **kwargs)

    def from_modules(
        self,
        module_names: list[str],
        design_name: str = "design",
        method: str = "weighted_bfs",
        **kwargs,
    ) -> ImpactResult:
        """Analyze impact starting from module names (convenience)."""
        node_ids = [f"{design_name}::{m}" for m in module_names]
        valid = [n for n in node_ids if n in self.graph]
        if not valid:
            raise ValueError(f"No matching modules found for: {module_names}")
        runner = getattr(self, method)
        return runner(changed_nodes=valid, **kwargs)

    # ------------------------------------------------------------------
    # Compare methods
    # ------------------------------------------------------------------

    def compare_methods(
        self,
        changed_nodes: list[str],
        ground_truth: set[str] | None = None,
        methods: list[str] | None = None,
        **kwargs,
    ) -> dict[str, ImpactResult | dict[str, float]]:
        """Run multiple methods and optionally compute metrics against ground truth.

        Returns a dict with keys: method names -> ImpactResult,
        plus "metrics" -> {method -> {metric -> value}} if ground_truth provided.
        """
        methods = methods or ["weighted_bfs", "bfs", "pagerank", "structural", "lexical"]
        results: dict[str, Any] = {}

        for method_name in methods:
            runner = getattr(self, method_name, None)
            if runner is None:
                logger.warning("Unknown method: %s", method_name)
                continue
            try:
                results[method_name] = runner(changed_nodes=changed_nodes, **kwargs)
            except Exception as e:
                logger.error("Method %s failed: %s", method_name, e)

        if ground_truth is not None:
            metrics: dict[str, dict[str, float]] = {}
            for method_name, impact_result in results.items():
                if isinstance(impact_result, ImpactResult):
                    metrics[method_name] = compute_metrics(
                        impact_result, ground_truth
                    )
            results["metrics"] = metrics

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_reason(self, path: list[str], relations: list[str]) -> str:
        """Build a human-readable explanation of the impact path."""
        if len(path) <= 1 or not relations:
            return "directly changed"

        G = self.graph
        parts = []
        for i in range(len(relations)):
            src_name = G.nodes.get(path[i], {}).get("name", path[i].split("::")[-1])
            dst_name = G.nodes.get(path[i+1], {}).get("name", path[i+1].split("::")[-1])
            rel = relations[i]
            parts.append(f"{src_name} --{rel}--> {dst_name}")

        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(
    result: ImpactResult,
    ground_truth: set[str],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute precision, recall, F1, Recall@K, MRR, NDCG against ground truth.

    Args:
        result: The impact analysis result.
        ground_truth: Set of node IDs that are truly impacted.
        k_values: K values for Recall@K (default [5, 10, 20]).
    """
    k_values = k_values or [5, 10, 20]
    predicted = result.artifact_ids()
    gt = ground_truth

    tp = predicted & gt
    fp = predicted - gt
    fn = gt - predicted

    precision = len(tp) / len(predicted) if predicted else 0.0
    recall = len(tp) / len(gt) if gt else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": len(tp),
        "false_positives": len(fp),
        "false_negatives": len(fn),
        "predicted_count": len(predicted),
        "ground_truth_count": len(gt),
    }

    # Ranked metrics (using risk_score ordering)
    ranked = [a.node_id for a in result.impacted]  # already sorted by risk_score desc

    # Recall@K
    for k in k_values:
        top_k = set(ranked[:k])
        recall_at_k = len(top_k & gt) / len(gt) if gt else 0.0
        metrics[f"recall@{k}"] = recall_at_k

    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for i, node_id in enumerate(ranked, 1):
        if node_id in gt:
            mrr = 1.0 / i
            break
    metrics["mrr"] = mrr

    # NDCG@K
    for k in k_values:
        metrics[f"ndcg@{k}"] = _ndcg_at_k(ranked, gt, k)

    return metrics


def _ndcg_at_k(ranked: list[str], ground_truth: set[str], k: int) -> float:
    """Compute NDCG@K."""
    dcg = 0.0
    for i, node_id in enumerate(ranked[:k], 1):
        rel = 1.0 if node_id in ground_truth else 0.0
        dcg += rel / math.log2(i + 1)

    # Ideal DCG: all relevant items at the top
    n_relevant = min(len(ground_truth), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_relevant + 1))

    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Impact result comparison across artifact types
# ---------------------------------------------------------------------------

def compute_per_type_metrics(
    result: ImpactResult,
    ground_truth_by_type: dict[str, set[str]],
) -> dict[str, dict[str, float]]:
    """Compute metrics separately for each artifact type.

    Args:
        ground_truth_by_type: Maps artifact type name to set of node IDs.
    """
    per_type: dict[str, dict[str, float]] = {}
    for atype, gt_nodes in ground_truth_by_type.items():
        predicted_of_type = result.artifact_ids(atype)
        tp = predicted_of_type & gt_nodes
        fp = predicted_of_type - gt_nodes
        fn = gt_nodes - predicted_of_type
        p = len(tp) / len(predicted_of_type) if predicted_of_type else 0.0
        r = len(tp) / len(gt_nodes) if gt_nodes else 0.0
        f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        per_type[atype] = {"precision": p, "recall": r, "f1": f,
                           "tp": len(tp), "fp": len(fp), "fn": len(fn)}
    return per_type


def compute_per_hop_recall(
    result: ImpactResult,
    ground_truth: set[str],
    max_hop: int = 5,
) -> dict[int, float]:
    """Compute recall broken down by hop distance.

    For each hop h, reports what fraction of ground-truth items at distance h
    were found. Items whose ground-truth distance is unknown use their
    predicted hop distance.
    """
    per_hop: dict[int, dict[str, int]] = {}
    for h in range(1, max_hop + 1):
        per_hop[h] = {"found": 0, "total": 0}

    for artifact in result.impacted:
        h = artifact.hop_distance
        if h < 1 or h > max_hop:
            continue
        per_hop[h]["total"] += 1
        if artifact.node_id in ground_truth:
            per_hop[h]["found"] += 1

    return {
        h: (counts["found"] / counts["total"] if counts["total"] > 0 else 0.0)
        for h, counts in per_hop.items()
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_impact_summary(result: ImpactResult, top_k: int = 20):
    """Print a summary of impact analysis results."""
    print(f"\nChange Impact Analysis ({result.method})")
    print(f"  Changed nodes: {len(result.changed_nodes)}")
    print(f"  Total impacted: {len(result.impacted)}")

    by_type = result.by_type
    print("  By artifact type:")
    for atype, items in sorted(by_type.items()):
        print(f"    {atype}: {len(items)}")

    by_hop: dict[int, int] = {}
    for a in result.impacted:
        by_hop[a.hop_distance] = by_hop.get(a.hop_distance, 0) + 1
    print("  By hop distance:")
    for h in sorted(by_hop):
        print(f"    hop {h}: {by_hop[h]}")

    print(f"\n  Top {min(top_k, len(result.impacted))} impacted artifacts:")
    for a in result.top_k(top_k):
        print(f"    [{a.artifact_type:12s}] {a.name:30s} "
              f"risk={a.risk_score:.4f}  hop={a.hop_distance}  "
              f"{'| ' + a.reason if a.reason else ''}")


def print_metrics(metrics: dict[str, float]):
    """Print computed metrics."""
    print(f"  Precision:  {metrics.get('precision', 0):.4f}")
    print(f"  Recall:     {metrics.get('recall', 0):.4f}")
    print(f"  F1:         {metrics.get('f1', 0):.4f}")
    print(f"  MRR:        {metrics.get('mrr', 0):.4f}")
    for k in [5, 10, 20]:
        key = f"recall@{k}"
        if key in metrics:
            print(f"  Recall@{k}:  {metrics[key]:.4f}")
        nkey = f"ndcg@{k}"
        if nkey in metrics:
            print(f"  NDCG@{k}:   {metrics[nkey]:.4f}")
