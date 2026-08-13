"""Integration tests for the unified KG builder."""

import sys
from pathlib import Path

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from keda.graph.builder import KGBuilder, DesignConfig, print_build_summary, export_graph

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def build_result():
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
    builder = KGBuilder()
    return builder.build(config)


class TestFullBuild:
    def test_graph_not_empty(self, build_result):
        G = build_result.graph
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0

    def test_all_node_types_present(self, build_result):
        G = build_result.graph
        types = {d.get("type") for _, d in G.nodes(data=True)}
        assert "Module" in types
        assert "Port" in types
        assert "Register" in types
        assert "Clock" in types
        assert "Constraint" in types
        assert "Assertion" in types
        assert "Commit" in types
        assert "Parameter" in types

    def test_cross_artifact_edges_exist(self, build_result):
        G = build_result.graph
        relations = {d.get("relation") for _, _, d in G.edges(data=True)}

        # RTL relations
        assert "instantiates" in relations
        assert "has_port" in relations
        assert "contains_register" in relations
        assert "clocked_by" in relations

        # SDC relations
        assert "constrained_by" in relations or "applies_to" in relations

        # SVA — Assertion nodes should exist even if verifies edges don't
        # (assertions in our fixture are in separate modules not in the RTL)
        assertion_nodes = [
            n for n, d in G.nodes(data=True) if d.get("type") == "Assertion"
        ]
        assert len(assertion_nodes) > 0

        # Git relations
        assert "modifies" in relations

    def test_module_to_constraint_path(self, build_result):
        """Verify a multi-hop path exists: Module -> Port -> constrained_by -> Constraint."""
        G = build_result.graph

        # Find a port with a constraint
        constrained_ports = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("relation") == "constrained_by"
        ]
        assert len(constrained_ports) > 0

        # Verify one of these ports belongs to a module
        port_id = constrained_ports[0][0]
        port_data = G.nodes.get(port_id, {})
        assert port_data.get("type") == "Port"
        # The port should be reachable from a module via has_port
        parent_modules = [
            u for u, v, d in G.edges(data=True)
            if v == port_id and d.get("relation") == "has_port"
        ]
        assert len(parent_modules) > 0

    def test_commit_to_module_edges(self, build_result):
        G = build_result.graph
        modifies_edges = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("relation") == "modifies"
        ]
        assert len(modifies_edges) >= 3

        # Verify the source is a Commit and target is a Module
        for u, v in modifies_edges:
            assert G.nodes[u].get("type") == "Commit"
            assert G.nodes[v].get("type") == "Module"

    def test_module_annotated_with_change_freq(self, build_result):
        G = build_result.graph
        for _, d in G.nodes(data=True):
            if d.get("type") == "Module" and d.get("name") == "baud_gen":
                assert d.get("change_frequency", 0) >= 1

    def test_multi_hop_traversal(self, build_result):
        """Test a realistic multi-hop query:
        Commit -> modifies -> Module -> contains_register -> Register -> clocked_by -> Clock
        """
        G = build_result.graph

        # Find a commit that modifies baud_gen
        baud_commits = []
        for u, v, d in G.edges(data=True):
            if (d.get("relation") == "modifies" and
                    G.nodes.get(v, {}).get("name") == "baud_gen"):
                baud_commits.append(u)

        assert len(baud_commits) >= 1

        commit = baud_commits[0]
        # Traverse: commit -> module -> register -> clock
        modules = [v for _, v, d in G.out_edges(commit, data=True)
                   if d.get("relation") == "modifies"]
        assert len(modules) >= 1

        registers = []
        for mod in modules:
            for _, reg, d in G.out_edges(mod, data=True):
                if d.get("relation") == "contains_register":
                    registers.append(reg)
        assert len(registers) >= 1

        clocks = set()
        for reg in registers:
            for _, clk, d in G.out_edges(reg, data=True):
                if d.get("relation") == "clocked_by":
                    clocks.add(clk)
        assert len(clocks) >= 1

    def test_graph_size_reasonable(self, build_result):
        G = build_result.graph
        # For UART design: expect ~60-120 nodes, ~100-250 edges
        assert 40 < G.number_of_nodes() < 200
        assert 60 < G.number_of_edges() < 400


class TestExport:
    def test_export_json(self, build_result, tmp_path):
        out = tmp_path / "test_graph.json"
        export_graph(build_result.graph, out, format="json")
        assert out.exists()
        assert out.stat().st_size > 100

    def test_export_graphml(self, build_result, tmp_path):
        out = tmp_path / "test_graph.graphml"
        export_graph(build_result.graph, out, format="graphml")
        assert out.exists()
        assert out.stat().st_size > 100


class TestPrintSummary:
    def test_summary(self, build_result, capsys):
        print_build_summary(build_result)
        captured = capsys.readouterr()
        assert "KEDA Knowledge Graph" in captured.out
        assert "RTL Extraction" in captured.out
        assert "SDC Constraints" in captured.out
        assert "Git History" in captured.out
        assert "Total nodes:" in captured.out
