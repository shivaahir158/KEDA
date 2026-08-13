"""Tests for the SVA assertion and testbench extractor."""

import sys
from pathlib import Path

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from keda.extractors.sva_extractor import SVAExtractor, print_sva_summary

FIXTURE_DIR = Path(__file__).parent / "fixtures"
ASSERTION_FILE = FIXTURE_DIR / "uart_assertions.sv"


@pytest.fixture(scope="module")
def sva_result():
    extractor = SVAExtractor()
    return extractor.extract(
        [ASSERTION_FILE],
        design_name="uart",
        known_modules={"uart_top", "uart_tx", "uart_rx", "baud_gen"},
    )


class TestSVAExtraction:
    def test_files_scanned(self, sva_result):
        assert len(sva_result.files_scanned) == 1

    def test_assertions_found(self, sva_result):
        # uart_assertions module: A1(unnamed), a_tx_idle, a_rx_valid_pulse,
        #   a_baud_tick_pulse, c_loopback, a_no_tx_during_busy, a_reset_tx_idle
        # baud_gen_with_asserts: a_counter_no_overflow, a_tick_one_hot
        assert len(sva_result.assertions) >= 9

    def test_assertion_types(self, sva_result):
        types = {a.assertion_type for a in sva_result.assertions}
        assert "assert" in types
        assert "cover" in types
        assert "assume" in types

    def test_named_assertions(self, sva_result):
        named = [a for a in sva_result.assertions if a.name is not None]
        names = {a.name for a in named}
        assert "a_tx_idle" in names
        assert "a_rx_valid_pulse" in names
        assert "a_baud_tick_pulse" in names
        assert "c_loopback" in names
        assert "a_counter_no_overflow" in names

    def test_containing_module(self, sva_result):
        for a in sva_result.assertions:
            if a.name == "a_tx_idle":
                assert a.containing_module == "uart_assertions"
            if a.name == "a_counter_no_overflow":
                assert a.containing_module == "baud_gen_with_asserts"

    def test_assertion_property_type(self, sva_result):
        for a in sva_result.assertions:
            assert a.property_type == "concurrent"

    def test_clock_signal_extraction(self, sva_result):
        for a in sva_result.assertions:
            if a.name == "a_tx_idle":
                assert a.clock_signal == "clk"

    def test_disable_condition(self, sva_result):
        for a in sva_result.assertions:
            if a.name == "a_tx_idle":
                assert a.disable_condition is not None
                assert "rst_n" in a.disable_condition

    def test_error_messages(self, sva_result):
        msgs = [a.error_message for a in sva_result.assertions if a.error_message]
        assert len(msgs) >= 1
        assert any("TX" in m or "tx" in m.lower() for m in msgs)

    def test_cover_assertion(self, sva_result):
        covers = [a for a in sva_result.assertions if a.assertion_type == "cover"]
        assert len(covers) >= 1
        assert covers[0].name == "c_loopback"

    def test_assume_assertion(self, sva_result):
        assumes = [a for a in sva_result.assertions if a.assertion_type == "assume"]
        assert len(assumes) >= 1

    def test_source_info(self, sva_result):
        for a in sva_result.assertions:
            assert a.file_path
            assert a.line_number > 0

    def test_property_text_captured(self, sva_result):
        for a in sva_result.assertions:
            assert a.property_text
            assert len(a.property_text) > 10


class TestSVAGraphIntegration:
    def test_add_to_graph_with_modules(self, sva_result):
        """Test linking assertions to existing module nodes."""
        G = nx.DiGraph()
        # Add module nodes that the assertions reference
        G.add_node("uart::uart_assertions", type="Module", name="uart_assertions")
        G.add_node("uart::baud_gen_with_asserts", type="Module", name="baud_gen_with_asserts")
        G.add_node("uart::clk::clk", type="Clock", name="clk")

        extractor = SVAExtractor()
        extractor.add_to_graph(sva_result, G, design_name="uart")

        # Assertions should be linked to modules
        verifies_edges = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("relation") == "verifies"
        ]
        assert len(verifies_edges) >= 2

        # Reverse edges
        verified_by = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("relation") == "verified_by"
        ]
        assert len(verified_by) >= 2

        # Clock links
        uses_clock = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("relation") == "uses_clock"
        ]
        assert len(uses_clock) >= 1


class TestTestbenchDetection:
    def test_testbench_fixture(self):
        """Create a minimal testbench fixture and test detection."""
        import tempfile
        tb_content = """\
module tb_uart_top;
    reg clk, rst_n;
    wire tx_out, rx_valid;

    uart_top #(.CLK_FREQ(50000000)) dut (
        .clk(clk),
        .rst_n(rst_n),
        .tx_out(tx_out),
        .rx_valid(rx_valid)
    );

    initial begin
        clk = 0;
        rst_n = 0;
        #100 rst_n = 1;
        #10000 $finish;
    end

    always #10 clk = ~clk;
endmodule
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_tb.sv", delete=False
        ) as f:
            f.write(tb_content)
            tb_path = f.name

        extractor = SVAExtractor()
        result = extractor.extract(
            [tb_path],
            design_name="uart",
            known_modules={"uart_top", "uart_tx", "uart_rx", "baud_gen"},
        )

        assert len(result.tests) == 1
        assert result.tests[0].name == "tb_uart_top"
        assert "uart_top" in result.tests[0].modules_instantiated

        Path(tb_path).unlink()


class TestPrintSummary:
    def test_summary(self, sva_result, capsys):
        print_sva_summary(sva_result)
        captured = capsys.readouterr()
        assert "Assertions found:" in captured.out
