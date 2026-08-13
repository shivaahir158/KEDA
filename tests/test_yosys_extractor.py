"""Tests for the Yosys RTL extractor."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from keda.extractors.yosys_extractor import (
    YosysExtractor,
    print_extraction_summary,
    print_graph_summary,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures"
UART_FILES = [
    FIXTURE_DIR / "uart_top.v",
    FIXTURE_DIR / "uart_tx.v",
    FIXTURE_DIR / "uart_rx.v",
    FIXTURE_DIR / "baud_gen.v",
]


@pytest.fixture(scope="module")
def extractor():
    return YosysExtractor()


@pytest.fixture(scope="module")
def extraction_result(extractor):
    return extractor.extract(UART_FILES, top_module="uart_top", design_name="uart")


@pytest.fixture(scope="module")
def graph(extractor, extraction_result):
    return extractor.build_graph(extraction_result)


class TestYosysExtraction:
    def test_modules_extracted(self, extraction_result):
        modules = extraction_result.modules
        module_names = {m["name"] for m in modules.values()}
        assert "uart_top" in module_names
        assert "uart_tx" in module_names
        assert "uart_rx" in module_names
        assert "baud_gen" in module_names

    def test_top_module_flagged(self, extraction_result):
        top_modules = [
            m for m in extraction_result.modules.values() if m.get("is_top")
        ]
        assert len(top_modules) == 1
        assert top_modules[0]["name"] == "uart_top"

    def test_ports_extracted(self, extraction_result):
        ports = extraction_result.ports
        # uart_top should have: clk, rst_n, tx_data, tx_valid, tx_out,
        #   tx_busy, rx_in, rx_data, rx_valid, baud_div
        top_ports = {
            p["name"] for p in ports.values()
            if "uart_top" in p.get("module", "")
        }
        assert "clk" in top_ports
        assert "rst_n" in top_ports
        assert "tx_data" in top_ports
        assert "rx_data" in top_ports
        assert "baud_div" in top_ports

    def test_port_directions(self, extraction_result):
        for port in extraction_result.ports.values():
            if port["name"] == "clk" and "uart_top" in port["module"]:
                assert port["direction"] == "input"
            if port["name"] == "tx_out" and "uart_top" in port["module"]:
                assert port["direction"] == "output"

    def test_port_widths(self, extraction_result):
        for port in extraction_result.ports.values():
            if port["name"] == "tx_data" and "uart_top" in port["module"]:
                assert port["width"] == 8
            if port["name"] == "baud_div" and "uart_top" in port["module"]:
                assert port["width"] == 16
            if port["name"] == "clk" and "uart_top" in port["module"]:
                assert port["width"] == 1

    def test_instances_extracted(self, extraction_result):
        instances = extraction_result.instances
        inst_names = {i["name"] for i in instances.values()}
        assert "u_baud_gen" in inst_names
        assert "u_uart_tx" in inst_names
        assert "u_uart_rx" in inst_names

    def test_instance_hierarchy(self, extraction_result):
        for inst in extraction_result.instances.values():
            if inst["name"] == "u_baud_gen":
                assert "uart_top" in inst["parent_module"]
                assert "baud_gen" in inst["child_module"]

    def test_registers_extracted(self, extraction_result):
        regs = extraction_result.registers
        assert len(regs) > 0
        # baud_gen should have counter and baud_tick registers
        baud_regs = [
            r for r in regs.values()
            if "baud_gen" in r.get("module", "")
        ]
        assert len(baud_regs) >= 2  # counter (16-bit) and baud_tick (1-bit)

    def test_register_clock_assignment(self, extraction_result):
        for reg in extraction_result.registers.values():
            # All registers in this design should be clocked by 'clk'
            assert reg.get("clock_signal") == "clk", (
                f"Register {reg['name']} in {reg['module']} has "
                f"clock_signal={reg.get('clock_signal')}"
            )

    def test_clocks_inferred(self, extraction_result):
        clocks = extraction_result.clocks
        assert len(clocks) >= 1
        clk_names = {c["name"] for c in clocks.values()}
        assert "clk" in clk_names

    def test_parameters_extracted(self, extraction_result):
        params = extraction_result.parameters
        param_names = {p["name"] for p in params.values()}
        assert "CLK_FREQ" in param_names
        assert "BAUD_RATE" in param_names
        assert "DATA_BITS" in param_names

    def test_parameter_values(self, extraction_result):
        for param in extraction_result.parameters.values():
            if param["name"] == "DATA_BITS":
                assert param["default_value"] == 8
            if param["name"] == "BAUD_RATE":
                assert param["default_value"] == 115200


class TestGraphConstruction:
    def test_graph_has_nodes(self, graph):
        assert graph.number_of_nodes() > 0

    def test_graph_has_edges(self, graph):
        assert graph.number_of_edges() > 0

    def test_module_nodes_present(self, graph):
        module_nodes = [
            n for n, d in graph.nodes(data=True) if d.get("type") == "Module"
        ]
        assert len(module_nodes) == 4

    def test_instantiation_edges(self, graph):
        inst_edges = [
            (u, v) for u, v, d in graph.edges(data=True)
            if d.get("relation") == "instantiates"
        ]
        assert len(inst_edges) == 3  # uart_top instantiates 3 submodules

    def test_has_port_edges(self, graph):
        port_edges = [
            (u, v) for u, v, d in graph.edges(data=True)
            if d.get("relation") == "has_port"
        ]
        assert len(port_edges) > 0

    def test_clocked_by_edges(self, graph):
        clk_edges = [
            (u, v) for u, v, d in graph.edges(data=True)
            if d.get("relation") == "clocked_by"
        ]
        assert len(clk_edges) > 0
        # Every register should have a clocked_by edge
        reg_nodes = [
            n for n, d in graph.nodes(data=True) if d.get("type") == "Register"
        ]
        regs_with_clock = {u for u, v, d in graph.edges(data=True)
                           if d.get("relation") == "clocked_by"}
        assert regs_with_clock == set(reg_nodes)

    def test_depended_by_edges(self, graph):
        dep_edges = [
            (u, v, d) for u, v, d in graph.edges(data=True)
            if d.get("relation") == "depended_by"
        ]
        # baud_gen, uart_tx, uart_rx all depended_by uart_top
        assert len(dep_edges) == 3

    def test_traversal_from_module(self, graph):
        """Test that we can traverse from a module to its registers and clocks."""
        # Find baud_gen module
        baud_mod = None
        for n, d in graph.nodes(data=True):
            if d.get("type") == "Module" and d.get("name") == "baud_gen":
                baud_mod = n
                break
        assert baud_mod is not None

        # Traverse to registers
        regs = [
            v for _, v, d in graph.out_edges(baud_mod, data=True)
            if d.get("relation") == "contains_register"
        ]
        assert len(regs) >= 2

        # Traverse from registers to clocks
        clocks = set()
        for reg in regs:
            for _, clk, d in graph.out_edges(reg, data=True):
                if d.get("relation") == "clocked_by":
                    clocks.add(clk)
        assert len(clocks) >= 1


class TestPrintSummary:
    def test_extraction_summary(self, extraction_result, capsys):
        print_extraction_summary(extraction_result)
        captured = capsys.readouterr()
        assert "Modules:" in captured.out
        assert "Registers:" in captured.out

    def test_graph_summary(self, graph, capsys):
        print_graph_summary(graph)
        captured = capsys.readouterr()
        assert "Total nodes:" in captured.out
        assert "Total edges:" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
