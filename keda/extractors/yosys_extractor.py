"""
Yosys-based RTL extractor for the KEDA knowledge graph.

Extracts from Yosys JSON output:
- Modules (definitions with source locations)
- Instances (module instantiations with hierarchy)
- Ports (with direction, width)
- Registers (DFF cells with clock/reset info)
- Nets (named signals with width)
- Parameters (module parameter defaults)
- Structural relationships (instantiates, has_port, contains_register, etc.)
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)

# Yosys cell types recognized as flip-flops / registers
DFF_CELL_TYPES = frozenset({
    "$dff", "$dffe", "$adff", "$adffe", "$sdff", "$sdffe", "$sdffce",
    "$dffsr", "$dffsre", "$dlatch", "$adlatch",
    # Technology-mapped variants may use these prefixes
    "$_DFF_", "$_DFFE_", "$_SDFF_", "$_DLATCH_",
})


def _is_dff_cell(cell_type: str) -> bool:
    """Check whether a Yosys cell type represents a register/flip-flop."""
    if cell_type in DFF_CELL_TYPES:
        return True
    # Match pattern like $_DFF_PP0_ or $adff etc.
    return bool(re.match(r"^\$_?(a?d?s?d?ff|dlatch|adlatch)", cell_type, re.IGNORECASE))


def _bin_str_to_int(s: str) -> int:
    """Convert a Yosys binary-string parameter value to an integer."""
    try:
        return int(s, 2)
    except (ValueError, TypeError):
        return 0


def _bit_width(bits: list) -> int:
    """Compute the width of a port/net from its bit vector."""
    return len(bits)


def _parse_src(src: str) -> tuple[str | None, int | None, int | None]:
    """Parse a Yosys 'src' attribute like 'uart_top.v:1.1-56.10'.

    Returns (file, start_line, end_line).
    """
    if not src:
        return None, None, None
    m = re.match(r"^(.+?):(\d+)\.\d+(?:-(\d+)\.\d+)?$", src)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3)) if m.group(3) else int(m.group(2))
    # Simpler form: file:line
    m = re.match(r"^(.+?):(\d+)$", src)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(2))
    return src, None, None


def _resolve_hdlname(attributes: dict) -> str | None:
    """Extract the original HDL name from Yosys attributes."""
    hdlname = attributes.get("hdlname", "")
    if hdlname:
        # Strip leading backslash that Yosys adds
        return hdlname.lstrip("\\").strip()
    return None


def _resolve_module_name(yosys_name: str, attributes: dict) -> str:
    """Map a Yosys mangled module name back to the original HDL name."""
    hdlname = _resolve_hdlname(attributes)
    if hdlname:
        return hdlname
    # Strip $paramod prefix and extract base name
    name = yosys_name
    if name.startswith("$paramod"):
        # e.g. $paramod$hash\module_name or $paramod\module\PARAM=val
        parts = name.split("\\")
        if len(parts) >= 2:
            # Take the module name part (after first backslash)
            name = parts[1].split("\\")[0]
        else:
            name = name.split("$")[-1]
    return name.lstrip("\\").strip()


@dataclass
class YosysExtractionResult:
    """Container for all data extracted from a Yosys JSON elaboration."""
    modules: dict[str, dict[str, Any]] = field(default_factory=dict)
    instances: dict[str, dict[str, Any]] = field(default_factory=dict)
    ports: dict[str, dict[str, Any]] = field(default_factory=dict)
    registers: dict[str, dict[str, Any]] = field(default_factory=dict)
    nets: dict[str, dict[str, Any]] = field(default_factory=dict)
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    clocks: dict[str, dict[str, Any]] = field(default_factory=dict)
    raw_json: dict | None = None


class YosysExtractor:
    """Extract RTL structural information using Yosys synthesis and JSON export.

    Usage:
        extractor = YosysExtractor()
        result = extractor.extract(
            verilog_files=["uart_top.v", "uart_rx.v", ...],
            top_module="uart_top"
        )
        graph = extractor.build_graph(result)
    """

    def __init__(self, yosys_binary: str = "yosys"):
        self.yosys_binary = yosys_binary
        self._verify_yosys()

    def _verify_yosys(self):
        """Check that Yosys is available."""
        try:
            result = subprocess.run(
                [self.yosys_binary, "--version"],
                capture_output=True, text=True, timeout=10,
            )
            logger.info("Yosys version: %s", result.stdout.strip())
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            raise RuntimeError(f"Yosys not found at '{self.yosys_binary}': {e}") from e

    def run_yosys(
        self,
        verilog_files: list[str | Path],
        top_module: str | None = None,
        work_dir: str | Path | None = None,
        extra_commands: str = "",
        include_dirs: list[str | Path] | None = None,
    ) -> dict:
        """Run Yosys on the given Verilog files and return the JSON design dict.

        Pipeline: read_verilog → hierarchy → proc → opt → write_json
        """
        verilog_files = [str(Path(f).resolve()) for f in verilog_files]
        for f in verilog_files:
            if not Path(f).exists():
                raise FileNotFoundError(f"Verilog file not found: {f}")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            json_path = tmp.name

        file_args = " ".join(f'"{f}"' for f in verilog_files)
        inc_args = ""
        if include_dirs:
            inc_args = " ".join(f'-I{Path(d).resolve()}' for d in include_dirs)
            inc_args = " " + inc_args
        script_lines = [
            f"read_verilog -sv{inc_args} {file_args}",
        ]
        if top_module:
            script_lines.append(f"hierarchy -top {top_module}")
        elif top_module is None:
            script_lines.append("hierarchy -auto-top")
        # top_module == "" explicitly skips hierarchy (keeps all modules)

        script_lines += [
            "proc",
            "opt",
            extra_commands,
            f'write_json "{json_path}"',
        ]
        script = "; ".join(line for line in script_lines if line)

        logger.info("Running Yosys: %s", script)
        result = subprocess.run(
            [self.yosys_binary, "-p", script],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=work_dir,
        )

        if result.returncode != 0:
            logger.error("Yosys stderr:\n%s", result.stderr[-2000:])
            raise RuntimeError(
                f"Yosys failed (exit code {result.returncode}). "
                f"Last stderr: {result.stderr[-500:]}"
            )

        with open(json_path) as f:
            design = json.load(f)

        Path(json_path).unlink(missing_ok=True)
        return design

    def extract(
        self,
        verilog_files: list[str | Path],
        top_module: str | None = None,
        work_dir: str | Path | None = None,
        design_name: str = "design",
        include_dirs: list[str | Path] | None = None,
    ) -> YosysExtractionResult:
        """Run Yosys and extract all structural information.

        Returns a YosysExtractionResult with modules, instances, ports,
        registers, nets, parameters, and inferred clocks.
        """
        design = self.run_yosys(verilog_files, top_module, work_dir,
                                include_dirs=include_dirs)
        return self._parse_design(design, design_name)

    def extract_from_json(
        self,
        json_path: str | Path,
        design_name: str = "design",
    ) -> YosysExtractionResult:
        """Extract from a pre-existing Yosys JSON file."""
        with open(json_path) as f:
            design = json.load(f)
        return self._parse_design(design, design_name)

    def _parse_design(self, design: dict, design_name: str) -> YosysExtractionResult:
        """Parse a Yosys JSON design dict into structured extraction results."""
        result = YosysExtractionResult(raw_json=design)

        # Build a mapping from Yosys mangled names to original HDL names
        yosys_to_hdl: dict[str, str] = {}
        for yosys_name, mod_data in design.get("modules", {}).items():
            hdl_name = _resolve_module_name(yosys_name, mod_data.get("attributes", {}))
            yosys_to_hdl[yosys_name] = hdl_name

        # Track which bit IDs map to which clock signals (per module)
        # This is used for register → clock linking
        module_clock_bits: dict[str, dict[int, str]] = {}

        for yosys_name, mod_data in design.get("modules", {}).items():
            hdl_name = yosys_to_hdl[yosys_name]
            attrs = mod_data.get("attributes", {})
            src_file, src_start, src_end = _parse_src(attrs.get("src", ""))
            is_top = attrs.get("top") is not None

            module_id = f"{design_name}::{hdl_name}"
            result.modules[module_id] = {
                "name": hdl_name,
                "yosys_name": yosys_name,
                "type": "Module",
                "src_file": src_file,
                "src_start_line": src_start,
                "src_end_line": src_end,
                "is_top": is_top,
                "design": design_name,
            }

            # Extract parameters
            for param_name, param_value in mod_data.get("parameter_default_values", {}).items():
                param_id = f"{module_id}::{param_name}"
                result.parameters[param_id] = {
                    "name": param_name,
                    "type": "Parameter",
                    "module": module_id,
                    "default_value_bin": param_value,
                    "default_value": _bin_str_to_int(param_value),
                }

            # Build bit → net name mapping for this module
            bit_to_net: dict[int, str] = {}
            for net_name, net_data in mod_data.get("netnames", {}).items():
                for bit_id in net_data.get("bits", []):
                    if isinstance(bit_id, int):
                        bit_to_net[bit_id] = net_name

            # Extract ports
            port_clock_bits: dict[int, str] = {}
            for port_name, port_data in mod_data.get("ports", {}).items():
                bits = port_data.get("bits", [])
                port_id = f"{module_id}::{port_name}"
                direction = port_data.get("direction", "unknown")
                width = _bit_width(bits)

                net_src = mod_data.get("netnames", {}).get(port_name, {}).get("attributes", {}).get("src", "")
                p_file, p_line, _ = _parse_src(net_src)

                result.ports[port_id] = {
                    "name": port_name,
                    "type": "Port",
                    "module": module_id,
                    "direction": direction,
                    "width": width,
                    "bits": bits,
                    "src_file": p_file,
                    "src_line": p_line,
                }

                # Heuristic: identify clock ports by naming convention
                if _is_clock_name(port_name) and direction == "input" and width == 1:
                    for bit_id in bits:
                        if isinstance(bit_id, int):
                            port_clock_bits[bit_id] = port_name

            module_clock_bits[yosys_name] = port_clock_bits

            # Extract nets (internal signals)
            for net_name, net_data in mod_data.get("netnames", {}).items():
                bits = net_data.get("bits", [])
                net_src = net_data.get("attributes", {}).get("src", "")
                n_file, n_line, _ = _parse_src(net_src)

                net_id = f"{module_id}::{net_name}"
                result.nets[net_id] = {
                    "name": net_name,
                    "type": "Net",
                    "module": module_id,
                    "width": _bit_width(bits),
                    "bits": bits,
                    "src_file": n_file,
                    "src_line": n_line,
                }

            # Extract cells: instances and registers
            for cell_name, cell_data in mod_data.get("cells", {}).items():
                cell_type = cell_data.get("type", "")
                cell_attrs = cell_data.get("attributes", {})
                cell_src = cell_attrs.get("src", "")
                c_file, c_start, c_end = _parse_src(cell_src)

                if _is_dff_cell(cell_type):
                    # This is a register
                    connections = cell_data.get("connections", {})
                    params = cell_data.get("parameters", {})

                    clk_bits = connections.get("CLK", connections.get("C", []))
                    width_param = params.get("WIDTH", "1")
                    width = _bin_str_to_int(width_param) if isinstance(width_param, str) else int(width_param)

                    # Resolve clock signal name
                    clk_name = None
                    for bit_id in clk_bits:
                        if isinstance(bit_id, int) and bit_id in bit_to_net:
                            clk_name = bit_to_net[bit_id]
                            break
                        if isinstance(bit_id, int) and bit_id in port_clock_bits:
                            clk_name = port_clock_bits[bit_id]
                            break

                    # Resolve Q (output) signal name for the register name
                    q_bits = connections.get("Q", [])
                    reg_signal = None
                    for bit_id in q_bits:
                        if isinstance(bit_id, int) and bit_id in bit_to_net:
                            reg_signal = bit_to_net[bit_id]
                            break

                    # Determine reset type
                    has_async_reset = "ARST" in connections
                    has_sync_reset = "SRST" in connections
                    reset_polarity = None
                    if has_async_reset:
                        pol = params.get("ARST_POLARITY", "1")
                        reset_polarity = "active_low" if pol in ("0", 0) else "active_high"
                    elif has_sync_reset:
                        pol = params.get("SRST_POLARITY", "1")
                        reset_polarity = "active_low" if pol in ("0", 0) else "active_high"

                    clk_polarity = params.get("CLK_POLARITY", "1")
                    clk_edge = "posedge" if clk_polarity in ("1", 1, "00000000000000000000000000000001") else "negedge"

                    reg_id = f"{module_id}::reg::{reg_signal or cell_name}"
                    result.registers[reg_id] = {
                        "name": reg_signal or cell_name,
                        "type": "Register",
                        "module": module_id,
                        "cell_type": cell_type,
                        "width": width,
                        "clock_signal": clk_name,
                        "clock_edge": clk_edge,
                        "has_async_reset": has_async_reset,
                        "has_sync_reset": has_sync_reset,
                        "reset_polarity": reset_polarity,
                        "src_file": c_file,
                        "src_start_line": c_start,
                    }

                    # Track clock
                    if clk_name:
                        clk_id = f"{design_name}::clk::{clk_name}"
                        if clk_id not in result.clocks:
                            result.clocks[clk_id] = {
                                "name": clk_name,
                                "type": "Clock",
                                "design": design_name,
                                "driven_registers": [],
                                "modules": set(),
                            }
                        result.clocks[clk_id]["driven_registers"].append(reg_id)
                        result.clocks[clk_id]["modules"].add(module_id)

                elif cell_type in yosys_to_hdl:
                    # This is a module instantiation
                    inst_module_name = yosys_to_hdl[cell_type]
                    inst_module_id = f"{design_name}::{inst_module_name}"

                    inst_id = f"{module_id}::inst::{cell_name}"
                    connections = cell_data.get("connections", {})
                    port_dirs = cell_data.get("port_directions", {})

                    # Resolve port connections to net names
                    port_connections = {}
                    for port, bits in connections.items():
                        connected_nets = set()
                        for bit_id in bits:
                            if isinstance(bit_id, int) and bit_id in bit_to_net:
                                connected_nets.add(bit_to_net[bit_id])
                        port_connections[port] = list(connected_nets)

                    result.instances[inst_id] = {
                        "name": cell_name,
                        "type": "Instance",
                        "parent_module": module_id,
                        "child_module": inst_module_id,
                        "child_module_name": inst_module_name,
                        "port_connections": port_connections,
                        "port_directions": port_dirs,
                        "src_file": c_file,
                        "src_start_line": c_start,
                        "src_end_line": c_end,
                    }

        # Convert clock module sets to lists for serialization
        for clk_data in result.clocks.values():
            clk_data["modules"] = list(clk_data["modules"])

        return result

    def build_graph(self, result: YosysExtractionResult) -> nx.DiGraph:
        """Build a NetworkX directed graph from extraction results.

        Node types: Module, Instance, Port, Register, Net, Clock, Parameter
        Edge types: has_port, has_net, contains_register, instantiates,
                    instance_of, clocked_by, parameter_of, connected_to
        """
        G = nx.DiGraph()

        # Add Module nodes
        for mod_id, mod_data in result.modules.items():
            G.add_node(mod_id, **mod_data)

        # Add Parameter nodes and edges
        for param_id, param_data in result.parameters.items():
            G.add_node(param_id, **param_data)
            G.add_edge(param_data["module"], param_id, relation="has_parameter")
            G.add_edge(param_id, param_data["module"], relation="parameter_of")

        # Add Port nodes and edges
        for port_id, port_data in result.ports.items():
            G.add_node(port_id, **port_data)
            G.add_edge(port_data["module"], port_id, relation="has_port")

        # Add Clock nodes
        for clk_id, clk_data in result.clocks.items():
            clk_node_data = {k: v for k, v in clk_data.items() if k != "driven_registers"}
            G.add_node(clk_id, **clk_node_data)

        # Add Register nodes and edges
        for reg_id, reg_data in result.registers.items():
            G.add_node(reg_id, **reg_data)
            G.add_edge(reg_data["module"], reg_id, relation="contains_register")

            # Register → clocked_by → Clock
            if reg_data.get("clock_signal"):
                clk_id = f"{result.modules[reg_data['module']]['design']}::clk::{reg_data['clock_signal']}"
                if clk_id in result.clocks:
                    G.add_edge(reg_id, clk_id,
                               relation="clocked_by",
                               edge=reg_data.get("clock_edge", "posedge"))

        # Add Instance nodes and instantiation edges
        for inst_id, inst_data in result.instances.items():
            G.add_node(inst_id, **inst_data)
            parent = inst_data["parent_module"]
            child = inst_data["child_module"]

            # Parent module → instantiates → child module
            G.add_edge(parent, child,
                        relation="instantiates",
                        instance_name=inst_data["name"],
                        instance_id=inst_id)

            # Instance → instance_of → child module
            G.add_edge(inst_id, child, relation="instance_of")

            # Parent module → has_instance → instance
            G.add_edge(parent, inst_id, relation="has_instance")

            # Add port connection edges
            for port_name, connected_nets in inst_data.get("port_connections", {}).items():
                # Instance port → connected_to → parent module net
                child_port_id = f"{child}::{port_name}"
                for net_name in connected_nets:
                    parent_net_id = f"{parent}::{net_name}"
                    if child_port_id in result.ports and parent_net_id in result.nets:
                        direction = inst_data.get("port_directions", {}).get(port_name, "")
                        if direction == "output":
                            G.add_edge(child_port_id, parent_net_id,
                                       relation="drives",
                                       via_instance=inst_id)
                        elif direction == "input":
                            G.add_edge(parent_net_id, child_port_id,
                                       relation="drives",
                                       via_instance=inst_id)

        # Compute module-level depends_on edges from instantiation
        self._add_dependency_edges(G, result)

        # Add Net nodes (optional, can create large graphs)
        # Only add nets that participate in cross-module connections
        cross_module_nets = set()
        for inst_data in result.instances.values():
            for connected_nets in inst_data.get("port_connections", {}).values():
                for net_name in connected_nets:
                    net_id = f"{inst_data['parent_module']}::{net_name}"
                    cross_module_nets.add(net_id)

        for net_id, net_data in result.nets.items():
            if net_id in cross_module_nets:
                G.add_node(net_id, **net_data)

        return G

    def _add_dependency_edges(self, G: nx.DiGraph, result: YosysExtractionResult):
        """Add module-level depends_on edges.

        A child module depends_on its parent in the sense that the parent
        uses/instantiates the child. But for impact analysis, if a child
        changes, the parent is affected. We model both directions:

        - parent → instantiates → child (structural)
        - child → depended_by → parent (for reverse impact traversal)
        """
        for inst_data in result.instances.values():
            parent = inst_data["parent_module"]
            child = inst_data["child_module"]
            # If child changes, parent is affected
            if not G.has_edge(child, parent) or G[child][parent].get("relation") != "depended_by":
                G.add_edge(child, parent,
                           relation="depended_by",
                           dependency_type="structural")

    def extract_and_build(
        self,
        verilog_files: list[str | Path],
        top_module: str | None = None,
        work_dir: str | Path | None = None,
        design_name: str = "design",
    ) -> tuple[YosysExtractionResult, nx.DiGraph]:
        """Convenience: extract and build graph in one call."""
        result = self.extract(verilog_files, top_module, work_dir, design_name)
        graph = self.build_graph(result)
        return result, graph


def _is_clock_name(name: str) -> bool:
    """Heuristic: does this signal name look like a clock?"""
    name_lower = name.lower()
    clock_patterns = [
        "clk", "clock", "ck", "clkin", "clk_",
        "_clk", "sysclk", "pclk", "hclk", "fclk",
        "mclk", "sclk", "gclk", "refclk",
    ]
    return any(pat in name_lower for pat in clock_patterns)


def print_extraction_summary(result: YosysExtractionResult):
    """Print a summary of extracted data."""
    print(f"  Modules:    {len(result.modules)}")
    print(f"  Instances:  {len(result.instances)}")
    print(f"  Ports:      {len(result.ports)}")
    print(f"  Registers:  {len(result.registers)}")
    print(f"  Nets:       {len(result.nets)}")
    print(f"  Parameters: {len(result.parameters)}")
    print(f"  Clocks:     {len(result.clocks)}")


def print_graph_summary(G: nx.DiGraph):
    """Print a summary of the knowledge graph."""
    print(f"  Total nodes: {G.number_of_nodes()}")
    print(f"  Total edges: {G.number_of_edges()}")

    # Count by type
    type_counts: dict[str, int] = {}
    for _, data in G.nodes(data=True):
        t = data.get("type", "Unknown")
        type_counts[t] = type_counts.get(t, 0) + 1
    print("  Node types:")
    for t, c in sorted(type_counts.items()):
        print(f"    {t}: {c}")

    # Count by relation
    rel_counts: dict[str, int] = {}
    for _, _, data in G.edges(data=True):
        r = data.get("relation", "unknown")
        rel_counts[r] = rel_counts.get(r, 0) + 1
    print("  Edge relations:")
    for r, c in sorted(rel_counts.items()):
        print(f"    {r}: {c}")
