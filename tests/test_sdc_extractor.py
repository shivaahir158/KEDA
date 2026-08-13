"""Tests for the SDC constraint extractor."""

import sys
from pathlib import Path

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from keda.extractors.sdc_extractor import SDCExtractor, print_sdc_summary

FIXTURE_DIR = Path(__file__).parent / "fixtures"
SDC_FILE = FIXTURE_DIR / "uart.sdc"


@pytest.fixture(scope="module")
def sdc_result():
    extractor = SDCExtractor()
    return extractor.extract([SDC_FILE], design_name="uart")


class TestSDCExtraction:
    def test_file_parsed(self, sdc_result):
        assert len(sdc_result.file_paths) == 1

    def test_constraint_count(self, sdc_result):
        # create_clock, create_generated_clock, 5 input_delay, 4 output_delay,
        # 2 false_path, 1 multicycle, 1 max_delay, 1 uncertainty, 1 load, 1 driving_cell
        assert len(sdc_result.constraints) >= 15

    def test_clock_definitions(self, sdc_result):
        assert "clk_core" in sdc_result.clocks
        assert sdc_result.clocks["clk_core"]["period"] == 20.0

    def test_clock_frequency(self, sdc_result):
        clk = sdc_result.clocks["clk_core"]
        assert clk["frequency_mhz"] == pytest.approx(50.0)

    def test_generated_clock(self, sdc_result):
        assert "clk_baud" in sdc_result.clocks
        clk_baud = sdc_result.clocks["clk_baud"]
        assert clk_baud.get("generated") is True
        assert clk_baud.get("divide_by") == 434

    def test_input_delays(self, sdc_result):
        input_delays = [
            c for c in sdc_result.constraints
            if c.constraint_type == "set_input_delay"
        ]
        assert len(input_delays) >= 4  # rx_in (max, min), tx_data, tx_valid, baud_div

    def test_output_delays(self, sdc_result):
        output_delays = [
            c for c in sdc_result.constraints
            if c.constraint_type == "set_output_delay"
        ]
        assert len(output_delays) >= 4

    def test_false_paths(self, sdc_result):
        false_paths = [
            c for c in sdc_result.constraints
            if c.constraint_type == "set_false_path"
        ]
        assert len(false_paths) == 2

    def test_false_path_targets(self, sdc_result):
        fp = [c for c in sdc_result.constraints if c.constraint_type == "set_false_path"]
        # First false path: from rst_n
        rst_fp = [c for c in fp if "rst_n" in c.targets]
        assert len(rst_fp) == 1

    def test_multicycle_path(self, sdc_result):
        mcp = [c for c in sdc_result.constraints if c.constraint_type == "set_multicycle_path"]
        assert len(mcp) == 1
        assert mcp[0].attributes.get("multiplier") == 2

    def test_max_delay(self, sdc_result):
        md = [c for c in sdc_result.constraints if c.constraint_type == "set_max_delay"]
        assert len(md) == 1
        assert md[0].attributes.get("delay") == 10.0

    def test_clock_uncertainty(self, sdc_result):
        cu = [c for c in sdc_result.constraints if c.constraint_type == "set_clock_uncertainty"]
        assert len(cu) == 1
        assert cu[0].attributes.get("uncertainty") == 0.5

    def test_constraint_has_source(self, sdc_result):
        for c in sdc_result.constraints:
            assert c.file_path
            assert c.line_number > 0

    def test_input_delay_clock_ref(self, sdc_result):
        input_delays = [
            c for c in sdc_result.constraints
            if c.constraint_type == "set_input_delay"
        ]
        for c in input_delays:
            assert c.source_clock == "clk_core"


class TestSDCGraphIntegration:
    def test_add_to_empty_graph(self, sdc_result):
        G = nx.DiGraph()
        extractor = SDCExtractor()
        extractor.add_to_graph(sdc_result, G, design_name="uart")

        constraint_nodes = [
            n for n, d in G.nodes(data=True) if d.get("type") == "Constraint"
        ]
        assert len(constraint_nodes) >= 15

        clock_nodes = [
            n for n, d in G.nodes(data=True) if d.get("type") == "Clock"
        ]
        assert len(clock_nodes) >= 2

    def test_add_to_rtl_graph(self, sdc_result):
        """Test linking SDC constraints to an existing RTL graph."""
        from keda.extractors.yosys_extractor import YosysExtractor

        yosys_ext = YosysExtractor()
        rtl_result, G = yosys_ext.extract_and_build(
            [FIXTURE_DIR / "uart_top.v", FIXTURE_DIR / "uart_tx.v",
             FIXTURE_DIR / "uart_rx.v", FIXTURE_DIR / "baud_gen.v"],
            top_module="uart_top",
            design_name="uart",
        )

        sdc_ext = SDCExtractor()
        sdc_ext.add_to_graph(sdc_result, G, design_name="uart")

        # Check that constraints are linked to ports
        applies_to_edges = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("relation") == "applies_to"
        ]
        assert len(applies_to_edges) > 0

        # Check constrained_by edges (reverse)
        constrained_by = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("relation") == "constrained_by"
        ]
        assert len(constrained_by) > 0

        # The clk port should be constrained
        clk_constrained = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("relation") == "constrained_by"
            and G.nodes[u].get("name") == "clk"
        ]
        assert len(clk_constrained) >= 1

    def test_generated_clock_derived_from(self, sdc_result):
        G = nx.DiGraph()
        # Add the source clock node first
        G.add_node("uart::clk::clk_core", type="Clock", name="clk_core")

        extractor = SDCExtractor()
        extractor.add_to_graph(sdc_result, G, design_name="uart")

        derived_edges = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("relation") == "derived_from"
        ]
        assert len(derived_edges) == 1


class TestPrintSummary:
    def test_summary(self, sdc_result, capsys):
        print_sdc_summary(sdc_result)
        captured = capsys.readouterr()
        assert "Total constraints:" in captured.out
        assert "Clocks defined:" in captured.out
