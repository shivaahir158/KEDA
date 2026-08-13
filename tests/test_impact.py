"""Tests for the change-impact analysis engine."""

import sys
from pathlib import Path

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from keda.graph.builder import KGBuilder, DesignConfig
from keda.analysis.impact import (
    ChangeImpactAnalyzer,
    ImpactResult,
    compute_metrics,
    compute_per_type_metrics,
    compute_per_hop_recall,
    print_impact_summary,
    print_metrics,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def full_graph():
    config = DesignConfig(
        name="uart",
        rtl_files=[
            FIXTURE_DIR / "uart_top.v",
            FIXTURE_DIR / "uart_tx.v",
            FIXTURE_DIR / "uart_rx.v",
            FIXTURE_DIR / "baud_gen.v",
        ],
        top_module="uart_top",
        sdc_files=[FIXTURE_DIR / "uart.sdc"],
        sva_files=[FIXTURE_DIR / "uart_assertions.sv"],
        repo_path=FIXTURE_DIR / "git_repo",
    )
    return KGBuilder().build(config).graph


@pytest.fixture(scope="module")
def analyzer(full_graph):
    return ChangeImpactAnalyzer(full_graph)


# ---- Helper to find node IDs by name/type ----

def find_node(G, name=None, ntype=None):
    for n, d in G.nodes(data=True):
        if name and d.get("name") != name:
            continue
        if ntype and d.get("type") != ntype:
            continue
        return n
    return None


def find_nodes(G, ntype):
    return {n for n, d in G.nodes(data=True) if d.get("type") == ntype}


# ==========================================================================
# Weighted BFS tests
# ==========================================================================

class TestWeightedBFS:
    def test_basic_from_module(self, analyzer, full_graph):
        baud_gen = find_node(full_graph, "baud_gen", "Module")
        result = analyzer.weighted_bfs([baud_gen])

        assert isinstance(result, ImpactResult)
        assert result.method == "weighted_bfs"
        assert len(result.impacted) > 0

    def test_changed_node_not_in_result(self, analyzer, full_graph):
        baud_gen = find_node(full_graph, "baud_gen", "Module")
        result = analyzer.weighted_bfs([baud_gen])

        impacted_ids = result.artifact_ids()
        assert baud_gen not in impacted_ids

    def test_finds_parent_module(self, analyzer, full_graph):
        """baud_gen is instantiated by uart_top -> uart_top should be impacted."""
        baud_gen = find_node(full_graph, "baud_gen", "Module")
        result = analyzer.weighted_bfs([baud_gen])

        uart_top = find_node(full_graph, "uart_top", "Module")
        impacted_ids = result.artifact_ids()
        assert uart_top in impacted_ids

    def test_finds_registers(self, analyzer, full_graph):
        baud_gen = find_node(full_graph, "baud_gen", "Module")
        result = analyzer.weighted_bfs([baud_gen])

        assert len(result.registers) > 0

    def test_finds_clocks(self, analyzer, full_graph):
        baud_gen = find_node(full_graph, "baud_gen", "Module")
        result = analyzer.weighted_bfs([baud_gen])

        assert len(result.clocks) > 0

    def test_finds_constraints(self, analyzer, full_graph):
        baud_gen = find_node(full_graph, "baud_gen", "Module")
        result = analyzer.weighted_bfs([baud_gen])

        assert len(result.constraints) > 0

    def test_risk_scores_decrease_with_distance(self, analyzer, full_graph):
        baud_gen = find_node(full_graph, "baud_gen", "Module")
        result = analyzer.weighted_bfs([baud_gen])

        # Group by hop and check average risk decreases
        hop_risks: dict[int, list[float]] = {}
        for a in result.impacted:
            hop_risks.setdefault(a.hop_distance, []).append(a.risk_score)

        avg_risks = {h: sum(rs) / len(rs) for h, rs in hop_risks.items()}
        hops = sorted(avg_risks.keys())
        for i in range(len(hops) - 1):
            assert avg_risks[hops[i]] >= avg_risks[hops[i + 1]], (
                f"Average risk at hop {hops[i]} ({avg_risks[hops[i]]:.4f}) "
                f"< hop {hops[i+1]} ({avg_risks[hops[i+1]]:.4f})"
            )

    def test_max_depth_limits_results(self, analyzer, full_graph):
        baud_gen = find_node(full_graph, "baud_gen", "Module")

        result_d1 = analyzer.weighted_bfs([baud_gen], max_depth=1)
        result_d5 = analyzer.weighted_bfs([baud_gen], max_depth=5)

        assert len(result_d1.impacted) <= len(result_d5.impacted)
        # All hop distances in d1 should be <= 1
        for a in result_d1.impacted:
            assert a.hop_distance <= 1

    def test_alpha_affects_scores(self, analyzer, full_graph):
        baud_gen = find_node(full_graph, "baud_gen", "Module")

        result_high = analyzer.weighted_bfs([baud_gen], alpha=0.9)
        result_low = analyzer.weighted_bfs([baud_gen], alpha=0.3)

        # Higher alpha should produce higher risk scores at depth > 1
        high_hop2 = [a for a in result_high.impacted if a.hop_distance >= 2]
        low_hop2 = [a for a in result_low.impacted if a.hop_distance >= 2]

        if high_hop2 and low_hop2:
            assert max(a.risk_score for a in high_hop2) > max(a.risk_score for a in low_hop2)

    def test_min_risk_filters(self, analyzer, full_graph):
        baud_gen = find_node(full_graph, "baud_gen", "Module")

        result_low = analyzer.weighted_bfs([baud_gen], min_risk=0.001)
        result_high = analyzer.weighted_bfs([baud_gen], min_risk=0.5)

        assert len(result_high.impacted) <= len(result_low.impacted)
        for a in result_high.impacted:
            assert a.risk_score >= 0.5

    def test_paths_are_valid(self, analyzer, full_graph):
        baud_gen = find_node(full_graph, "baud_gen", "Module")
        result = analyzer.weighted_bfs([baud_gen])

        G = full_graph
        for a in result.impacted:
            # Path should start at a changed node and end at this artifact
            assert len(a.path) >= 2
            assert a.path[0] == baud_gen
            assert a.path[-1] == a.node_id
            # Each step should be a real edge
            assert len(a.path_relations) == len(a.path) - 1
            for i in range(len(a.path) - 1):
                assert G.has_edge(a.path[i], a.path[i + 1]), (
                    f"Missing edge: {a.path[i]} -> {a.path[i+1]}"
                )

    def test_reasons_generated(self, analyzer, full_graph):
        baud_gen = find_node(full_graph, "baud_gen", "Module")
        result = analyzer.weighted_bfs([baud_gen])

        for a in result.impacted:
            assert a.reason  # should be non-empty

    def test_multiple_changed_nodes(self, analyzer, full_graph):
        baud_gen = find_node(full_graph, "baud_gen", "Module")
        uart_tx = find_node(full_graph, "uart_tx", "Module")
        result = analyzer.weighted_bfs([baud_gen, uart_tx])

        assert len(result.changed_nodes) == 2
        # Neither changed node should be in result
        assert baud_gen not in result.artifact_ids()
        assert uart_tx not in result.artifact_ids()

    def test_change_type_overrides(self, analyzer, full_graph):
        baud_gen = find_node(full_graph, "baud_gen", "Module")

        result_default = analyzer.weighted_bfs([baud_gen])
        result_clock = analyzer.weighted_bfs([baud_gen], change_type="clock")

        # Clock-type change should boost clock-related scores
        # Both should find clocks, but scores may differ
        assert len(result_clock.clocks) >= len(result_default.clocks)

    def test_exclude_types(self, analyzer, full_graph):
        baud_gen = find_node(full_graph, "baud_gen", "Module")
        result = analyzer.weighted_bfs([baud_gen], exclude_types={"Constraint", "Commit"})

        assert len(result.constraints) == 0

    def test_nonexistent_node_warning(self, analyzer, full_graph):
        result = analyzer.weighted_bfs(["nonexistent::node"])
        assert len(result.impacted) == 0


# ==========================================================================
# Unweighted BFS tests
# ==========================================================================

class TestBFS:
    def test_basic(self, analyzer, full_graph):
        baud_gen = find_node(full_graph, "baud_gen", "Module")
        result = analyzer.bfs([baud_gen], max_depth=3)

        assert result.method == "bfs"
        assert len(result.impacted) > 0

    def test_depth_limit(self, analyzer, full_graph):
        baud_gen = find_node(full_graph, "baud_gen", "Module")
        result = analyzer.bfs([baud_gen], max_depth=1)

        for a in result.impacted:
            assert a.hop_distance <= 1

    def test_finds_multi_hop(self, analyzer, full_graph):
        baud_gen = find_node(full_graph, "baud_gen", "Module")
        result = analyzer.bfs([baud_gen], max_depth=4)

        # Should find artifacts at hop > 1
        max_hop = max((a.hop_distance for a in result.impacted), default=0)
        assert max_hop >= 2


# ==========================================================================
# PageRank tests
# ==========================================================================

class TestPageRank:
    def test_basic(self, analyzer, full_graph):
        baud_gen = find_node(full_graph, "baud_gen", "Module")
        result = analyzer.pagerank([baud_gen])

        assert result.method == "pagerank"
        assert len(result.impacted) > 0

    def test_scores_normalized(self, analyzer, full_graph):
        baud_gen = find_node(full_graph, "baud_gen", "Module")
        result = analyzer.pagerank([baud_gen])

        for a in result.impacted:
            assert 0.0 <= a.risk_score <= 1.0


# ==========================================================================
# Structural baseline tests
# ==========================================================================

class TestStructural:
    def test_only_modules(self, analyzer, full_graph):
        baud_gen = find_node(full_graph, "baud_gen", "Module")
        result = analyzer.structural([baud_gen])

        assert result.method == "structural"
        for a in result.impacted:
            assert a.artifact_type == "Module"

    def test_finds_parent(self, analyzer, full_graph):
        baud_gen = find_node(full_graph, "baud_gen", "Module")
        result = analyzer.structural([baud_gen])

        names = {a.name for a in result.impacted}
        assert "uart_top" in names

    def test_misses_constraints(self, analyzer, full_graph):
        """Structural baseline should NOT find constraints."""
        baud_gen = find_node(full_graph, "baud_gen", "Module")
        result = analyzer.structural([baud_gen])

        assert len(result.constraints) == 0


# ==========================================================================
# Lexical baseline tests
# ==========================================================================

class TestLexical:
    def test_basic(self, analyzer, full_graph):
        baud_gen = find_node(full_graph, "baud_gen", "Module")
        result = analyzer.lexical([baud_gen])

        assert result.method == "lexical"
        assert len(result.impacted) > 0

    def test_finds_keyword_matches(self, analyzer, full_graph):
        baud_gen = find_node(full_graph, "baud_gen", "Module")
        result = analyzer.lexical([baud_gen])

        # Should find nodes with "baud" in their name/attributes
        names = {a.name for a in result.impacted if a.name}
        has_baud = any("baud" in n.lower() for n in names if n)
        assert has_baud


# ==========================================================================
# Convenience methods
# ==========================================================================

class TestConvenienceMethods:
    def test_from_commit(self, analyzer, full_graph):
        # Find a commit that actually has modifies edges to known modules
        G = full_graph
        commit = None
        for n, d in G.nodes(data=True):
            if d.get("type") != "Commit":
                continue
            has_mod = any(
                G.nodes.get(v, {}).get("type") == "Module"
                for _, v, ed in G.out_edges(n, data=True)
                if ed.get("relation") == "modifies"
            )
            if has_mod:
                commit = n
                break
        assert commit is not None

        result = analyzer.from_commit(commit)
        assert len(result.impacted) > 0

    def test_from_modules(self, analyzer, full_graph):
        result = analyzer.from_modules(["baud_gen"], design_name="uart")
        assert len(result.impacted) > 0

    def test_from_modules_invalid(self, analyzer, full_graph):
        with pytest.raises(ValueError):
            analyzer.from_modules(["nonexistent_module"], design_name="uart")


# ==========================================================================
# Metrics tests
# ==========================================================================

class TestMetrics:
    def test_perfect_recall(self):
        result = ImpactResult(changed_nodes=["a"], method="test")
        from keda.analysis.impact import ImpactedArtifact
        result.impacted = [
            ImpactedArtifact("b", "Module", "b", 0.9, 1, [], []),
            ImpactedArtifact("c", "Module", "c", 0.8, 1, [], []),
        ]
        gt = {"b", "c"}
        metrics = compute_metrics(result, gt)
        assert metrics["recall"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["f1"] == 1.0

    def test_partial_recall(self):
        result = ImpactResult(changed_nodes=["a"], method="test")
        from keda.analysis.impact import ImpactedArtifact
        result.impacted = [
            ImpactedArtifact("b", "Module", "b", 0.9, 1, [], []),
        ]
        gt = {"b", "c"}
        metrics = compute_metrics(result, gt)
        assert metrics["recall"] == 0.5
        assert metrics["precision"] == 1.0
        assert metrics["false_negatives"] == 1

    def test_with_false_positives(self):
        result = ImpactResult(changed_nodes=["a"], method="test")
        from keda.analysis.impact import ImpactedArtifact
        result.impacted = [
            ImpactedArtifact("b", "Module", "b", 0.9, 1, [], []),
            ImpactedArtifact("d", "Module", "d", 0.5, 2, [], []),
        ]
        gt = {"b", "c"}
        metrics = compute_metrics(result, gt)
        assert metrics["precision"] == 0.5
        assert metrics["recall"] == 0.5
        assert metrics["false_positives"] == 1

    def test_empty_prediction(self):
        result = ImpactResult(changed_nodes=["a"], method="test")
        gt = {"b", "c"}
        metrics = compute_metrics(result, gt)
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0

    def test_empty_ground_truth(self):
        result = ImpactResult(changed_nodes=["a"], method="test")
        from keda.analysis.impact import ImpactedArtifact
        result.impacted = [
            ImpactedArtifact("b", "Module", "b", 0.9, 1, [], []),
        ]
        gt: set[str] = set()
        metrics = compute_metrics(result, gt)
        assert metrics["recall"] == 0.0

    def test_recall_at_k(self):
        from keda.analysis.impact import ImpactedArtifact
        result = ImpactResult(changed_nodes=["a"], method="test")
        result.impacted = [
            ImpactedArtifact(f"n{i}", "Module", f"n{i}", 1.0 - i * 0.1, 1, [], [])
            for i in range(15)
        ]
        gt = {"n0", "n5", "n12"}
        metrics = compute_metrics(result, gt, k_values=[5, 10, 15])

        assert metrics["recall@5"] == pytest.approx(1 / 3)   # only n0 in top 5
        assert metrics["recall@10"] == pytest.approx(2 / 3)   # n0 and n5
        assert metrics["recall@15"] == pytest.approx(3 / 3)   # all 3

    def test_mrr(self):
        from keda.analysis.impact import ImpactedArtifact
        result = ImpactResult(changed_nodes=["a"], method="test")
        result.impacted = [
            ImpactedArtifact("n0", "Module", "n0", 0.9, 1, [], []),
            ImpactedArtifact("n1", "Module", "n1", 0.8, 1, [], []),
            ImpactedArtifact("n2", "Module", "n2", 0.7, 1, [], []),
        ]
        gt = {"n1", "n2"}
        metrics = compute_metrics(result, gt)
        # First relevant item is n1 at rank 2
        assert metrics["mrr"] == pytest.approx(0.5)

    def test_ndcg(self):
        from keda.analysis.impact import ImpactedArtifact
        result = ImpactResult(changed_nodes=["a"], method="test")
        result.impacted = [
            ImpactedArtifact("n0", "Module", "n0", 0.9, 1, [], []),
            ImpactedArtifact("n1", "Module", "n1", 0.8, 1, [], []),
        ]
        gt = {"n0", "n1"}
        metrics = compute_metrics(result, gt, k_values=[2])
        # Perfect ranking -> NDCG = 1.0
        assert metrics["ndcg@2"] == pytest.approx(1.0)


class TestPerTypeMetrics:
    def test_basic(self):
        from keda.analysis.impact import ImpactedArtifact
        result = ImpactResult(changed_nodes=["a"], method="test")
        result.impacted = [
            ImpactedArtifact("m1", "Module", "m1", 0.9, 1, [], []),
            ImpactedArtifact("c1", "Constraint", "c1", 0.7, 2, [], []),
        ]
        gt_by_type = {
            "Module": {"m1", "m2"},
            "Constraint": {"c1"},
        }
        per_type = compute_per_type_metrics(result, gt_by_type)
        assert per_type["Module"]["recall"] == 0.5
        assert per_type["Constraint"]["recall"] == 1.0


# ==========================================================================
# Method comparison
# ==========================================================================

class TestCompare:
    def test_compare_methods(self, analyzer, full_graph):
        baud_gen = find_node(full_graph, "baud_gen", "Module")
        results = analyzer.compare_methods(
            [baud_gen],
            methods=["weighted_bfs", "bfs", "structural", "lexical"],
        )

        assert "weighted_bfs" in results
        assert "bfs" in results
        assert "structural" in results
        assert "lexical" in results

    def test_compare_with_ground_truth(self, analyzer, full_graph):
        baud_gen = find_node(full_graph, "baud_gen", "Module")
        uart_top = find_node(full_graph, "uart_top", "Module")
        gt = {uart_top}

        results = analyzer.compare_methods(
            [baud_gen],
            ground_truth=gt,
            methods=["weighted_bfs", "structural"],
        )

        assert "metrics" in results
        assert "weighted_bfs" in results["metrics"]
        assert results["metrics"]["weighted_bfs"]["recall"] == 1.0


# ==========================================================================
# ImpactResult convenience
# ==========================================================================

class TestImpactResult:
    def test_top_k(self, analyzer, full_graph):
        baud_gen = find_node(full_graph, "baud_gen", "Module")
        result = analyzer.weighted_bfs([baud_gen])
        top5 = result.top_k(5)
        assert len(top5) <= 5
        # Should be sorted descending by risk
        for i in range(len(top5) - 1):
            assert top5[i].risk_score >= top5[i + 1].risk_score

    def test_at_hop(self, analyzer, full_graph):
        baud_gen = find_node(full_graph, "baud_gen", "Module")
        result = analyzer.weighted_bfs([baud_gen], max_depth=5)
        hop1 = result.at_hop(1)
        for a in hop1:
            assert a.hop_distance == 1

    def test_by_type(self, analyzer, full_graph):
        baud_gen = find_node(full_graph, "baud_gen", "Module")
        result = analyzer.weighted_bfs([baud_gen])
        by_type = result.by_type
        assert isinstance(by_type, dict)


# ==========================================================================
# Cross-artifact discovery (key research contribution)
# ==========================================================================

class TestCrossArtifactDiscovery:
    """These tests verify the core research claim: KG traversal finds
    cross-artifact impacts that structural-only methods miss."""

    def test_structural_misses_constraints(self, analyzer, full_graph):
        baud_gen = find_node(full_graph, "baud_gen", "Module")

        structural = analyzer.structural([baud_gen])
        kg = analyzer.weighted_bfs([baud_gen])

        assert len(structural.constraints) == 0
        assert len(kg.constraints) > 0

    def test_structural_misses_clocks(self, analyzer, full_graph):
        baud_gen = find_node(full_graph, "baud_gen", "Module")

        structural = analyzer.structural([baud_gen])
        kg = analyzer.weighted_bfs([baud_gen])

        assert len(structural.clocks) == 0
        assert len(kg.clocks) > 0

    def test_kg_finds_more_total(self, analyzer, full_graph):
        baud_gen = find_node(full_graph, "baud_gen", "Module")

        structural = analyzer.structural([baud_gen])
        kg = analyzer.weighted_bfs([baud_gen])

        assert len(kg.impacted) > len(structural.impacted)

    def test_multi_hop_path_to_constraint(self, analyzer, full_graph):
        """Verify a multi-hop path: Module -> Port -> constrained_by -> Constraint."""
        baud_gen = find_node(full_graph, "baud_gen", "Module")
        result = analyzer.weighted_bfs([baud_gen])

        constraint_hops = [a.hop_distance for a in result.constraints]
        assert constraint_hops  # at least one constraint found
        assert max(constraint_hops) >= 2  # constraints are >= 2 hops away


# ==========================================================================
# Print functions
# ==========================================================================

class TestPrinting:
    def test_print_impact_summary(self, analyzer, full_graph, capsys):
        baud_gen = find_node(full_graph, "baud_gen", "Module")
        result = analyzer.weighted_bfs([baud_gen])
        print_impact_summary(result, top_k=5)
        out = capsys.readouterr().out
        assert "Change Impact Analysis" in out
        assert "Total impacted:" in out

    def test_print_metrics(self, capsys):
        metrics = {"precision": 0.8, "recall": 0.6, "f1": 0.685,
                    "mrr": 0.5, "recall@5": 0.4, "ndcg@5": 0.55}
        print_metrics(metrics)
        out = capsys.readouterr().out
        assert "Precision:" in out
        assert "Recall:" in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
