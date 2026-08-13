"""
SDC (Synopsys Design Constraints) parser for the KEDA knowledge graph.

Extracts:
- Clock definitions (create_clock, create_generated_clock)
- Input/output delays (set_input_delay, set_output_delay)
- Timing exceptions (set_false_path, set_multicycle_path, set_max_delay, set_min_delay)
- Clock uncertainty
- Load and driving cell

Produces Constraint nodes and edges linking them to Ports, Clocks, and Modules.
"""

from __future__ import annotations

import logging
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class SDCConstraint:
    """A single parsed SDC constraint."""
    constraint_id: str
    constraint_type: str
    raw_line: str
    line_number: int
    file_path: str
    attributes: dict[str, Any] = field(default_factory=dict)
    targets: list[str] = field(default_factory=list)
    source_clock: str | None = None
    dest_clock: str | None = None


@dataclass
class SDCExtractionResult:
    """All constraints parsed from one or more SDC files."""
    constraints: list[SDCConstraint] = field(default_factory=list)
    clocks: dict[str, dict[str, Any]] = field(default_factory=dict)
    file_paths: list[str] = field(default_factory=list)


def _extract_get_targets(token: str) -> list[str]:
    """Extract target names from SDC get_* commands.

    Handles forms like:
      [get_ports clk]
      [get_ports {tx_data[*]}]
      [get_ports {a b c}]
      [get_pins u_baud_gen/counter_reg[*]]
      [get_clocks clk_core]
    """
    targets = []
    # Match [get_xxx ...] patterns
    for m in re.finditer(r'\[get_(?:ports|pins|clocks|cells|nets|registers)\s+([^\]]+)\]', token):
        inner = m.group(1).strip()
        # Remove braces for bus notation: {tx_data[*]} -> tx_data[*]
        inner = inner.strip("{}")
        # Split on whitespace for multi-target: {a b c} -> [a, b, c]
        for part in inner.split():
            # Clean up wildcards for matching: tx_data[*] -> tx_data
            clean = re.sub(r'\[\*?\]$', '', part)
            if clean:
                targets.append(clean)
    return targets


def _extract_get_type(text: str) -> str | None:
    """Extract the type of get_* command (ports, pins, clocks, etc.)."""
    m = re.search(r'\[get_(ports|pins|clocks|cells|nets|registers)', text)
    return m.group(1) if m else None


class SDCExtractor:
    """Parse SDC constraint files and extract constraint information.

    Usage:
        extractor = SDCExtractor()
        result = extractor.extract(["design.sdc"])
        extractor.add_to_graph(result, graph, design_name="uart")
    """

    # Recognized SDC commands
    CONSTRAINT_COMMANDS = {
        "create_clock", "create_generated_clock",
        "set_input_delay", "set_output_delay",
        "set_false_path", "set_multicycle_path",
        "set_max_delay", "set_min_delay",
        "set_clock_uncertainty", "set_clock_latency",
        "set_load", "set_driving_cell",
        "set_max_fanout", "set_dont_touch",
        "set_case_analysis",
    }

    def extract(self, sdc_files: list[str | Path], design_name: str = "design") -> SDCExtractionResult:
        """Parse one or more SDC files and extract all constraints."""
        result = SDCExtractionResult()
        constraint_counter = 0

        for sdc_file in sdc_files:
            sdc_path = Path(sdc_file)
            if not sdc_path.exists():
                logger.warning("SDC file not found: %s", sdc_path)
                continue

            result.file_paths.append(str(sdc_path))
            text = sdc_path.read_text()

            # Join continuation lines (backslash-newline)
            text = re.sub(r'\\\n\s*', ' ', text)

            for line_num, line in enumerate(text.splitlines(), 1):
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith("#"):
                    continue

                # Get the command name
                try:
                    tokens = shlex.split(line, comments=True)
                except ValueError:
                    tokens = line.split()

                if not tokens:
                    continue

                command = tokens[0]
                if command not in self.CONSTRAINT_COMMANDS:
                    continue

                constraint_counter += 1
                cid = f"{design_name}::sdc::{constraint_counter:04d}"

                constraint = SDCConstraint(
                    constraint_id=cid,
                    constraint_type=command,
                    raw_line=line,
                    line_number=line_num,
                    file_path=str(sdc_path),
                )

                # Parse based on command type
                parser = getattr(self, f"_parse_{command}", None)
                if parser:
                    parser(constraint, line, tokens)
                else:
                    # Generic: extract all targets
                    constraint.targets = _extract_get_targets(line)

                result.constraints.append(constraint)

                # Track clock definitions
                if command == "create_clock":
                    clk_name = constraint.attributes.get("name", "")
                    if clk_name:
                        result.clocks[clk_name] = {
                            "name": clk_name,
                            "type": "Clock",
                            "period": constraint.attributes.get("period"),
                            "frequency_mhz": (
                                1000.0 / constraint.attributes["period"]
                                if constraint.attributes.get("period")
                                else None
                            ),
                            "port": constraint.targets[0] if constraint.targets else None,
                            "source": "sdc",
                            "constraint_id": cid,
                        }
                elif command == "create_generated_clock":
                    clk_name = constraint.attributes.get("name", "")
                    if clk_name:
                        result.clocks[clk_name] = {
                            "name": clk_name,
                            "type": "Clock",
                            "generated": True,
                            "source_clock": constraint.source_clock,
                            "divide_by": constraint.attributes.get("divide_by"),
                            "multiply_by": constraint.attributes.get("multiply_by"),
                            "target": constraint.targets[0] if constraint.targets else None,
                            "source": "sdc",
                            "constraint_id": cid,
                        }

        return result

    # --- Per-command parsers ---

    def _parse_create_clock(self, c: SDCConstraint, line: str, tokens: list[str]):
        args = self._parse_args(tokens[1:], {
            "-period": float,
            "-name": str,
            "-waveform": str,
            "-add": bool,
        })
        c.attributes = args
        c.targets = _extract_get_targets(line)
        if "name" not in c.attributes and c.targets:
            c.attributes["name"] = c.targets[0]

    def _parse_create_generated_clock(self, c: SDCConstraint, line: str, tokens: list[str]):
        args = self._parse_args(tokens[1:], {
            "-name": str,
            "-source": str,
            "-divide_by": int,
            "-multiply_by": int,
            "-duty_cycle": float,
            "-add": bool,
            "-master_clock": str,
        })
        c.attributes = args
        # Extract source clock from -source argument
        source_text = args.get("source", "")
        if source_text:
            source_targets = _extract_get_targets(source_text)
            if source_targets:
                c.source_clock = source_targets[0]
        # The last [get_*] in the line is the target
        all_targets = _extract_get_targets(line)
        # Source targets are already captured, remaining are the constraint target
        c.targets = all_targets

    def _parse_set_input_delay(self, c: SDCConstraint, line: str, tokens: list[str]):
        args = self._parse_args(tokens[1:], {
            "-clock": str,
            "-max": float,
            "-min": float,
            "-rise": bool,
            "-fall": bool,
            "-add_delay": bool,
        })
        c.attributes = args
        c.targets = _extract_get_targets(line)
        # Extract clock reference
        clock_text = args.get("clock", "")
        if clock_text:
            clk_targets = _extract_get_targets(clock_text)
            if clk_targets:
                c.source_clock = clk_targets[0]
            elif not clock_text.startswith("["):
                c.source_clock = clock_text
        # The delay value is the first positional numeric arg
        c.attributes["delay"] = self._find_numeric_arg(tokens[1:])

    def _parse_set_output_delay(self, c: SDCConstraint, line: str, tokens: list[str]):
        # Same structure as input delay
        self._parse_set_input_delay(c, line, tokens)

    def _parse_set_false_path(self, c: SDCConstraint, line: str, tokens: list[str]):
        args = self._parse_args(tokens[1:], {
            "-from": str,
            "-to": str,
            "-through": str,
            "-setup": bool,
            "-hold": bool,
        })
        c.attributes = args
        c.targets = _extract_get_targets(line)
        # Separate from/to
        if "from" in args:
            from_targets = _extract_get_targets(args["from"])
            c.attributes["from_targets"] = from_targets
        if "to" in args:
            to_targets = _extract_get_targets(args["to"])
            c.attributes["to_targets"] = to_targets

    def _parse_set_multicycle_path(self, c: SDCConstraint, line: str, tokens: list[str]):
        args = self._parse_args(tokens[1:], {
            "-from": str,
            "-to": str,
            "-through": str,
            "-setup": bool,
            "-hold": bool,
        })
        c.attributes = args
        c.targets = _extract_get_targets(line)
        # The multiplier is the first positional arg
        c.attributes["multiplier"] = self._find_numeric_arg(tokens[1:])
        if "from" in args:
            c.attributes["from_targets"] = _extract_get_targets(args["from"])
        if "to" in args:
            c.attributes["to_targets"] = _extract_get_targets(args["to"])

    def _parse_set_max_delay(self, c: SDCConstraint, line: str, tokens: list[str]):
        args = self._parse_args(tokens[1:], {
            "-from": str,
            "-to": str,
            "-through": str,
        })
        c.attributes = args
        c.targets = _extract_get_targets(line)
        c.attributes["delay"] = self._find_numeric_arg(tokens[1:])
        if "from" in args:
            c.attributes["from_targets"] = _extract_get_targets(args["from"])
        if "to" in args:
            c.attributes["to_targets"] = _extract_get_targets(args["to"])

    def _parse_set_min_delay(self, c: SDCConstraint, line: str, tokens: list[str]):
        self._parse_set_max_delay(c, line, tokens)

    def _parse_set_clock_uncertainty(self, c: SDCConstraint, line: str, tokens: list[str]):
        args = self._parse_args(tokens[1:], {
            "-setup": bool,
            "-hold": bool,
            "-from": str,
            "-to": str,
        })
        c.attributes = args
        c.targets = _extract_get_targets(line)
        c.attributes["uncertainty"] = self._find_numeric_arg(tokens[1:])

    def _parse_set_load(self, c: SDCConstraint, line: str, tokens: list[str]):
        c.targets = _extract_get_targets(line)
        c.attributes["load"] = self._find_numeric_arg(tokens[1:])

    def _parse_set_driving_cell(self, c: SDCConstraint, line: str, tokens: list[str]):
        args = self._parse_args(tokens[1:], {
            "-lib_cell": str,
            "-pin": str,
        })
        c.attributes = args
        c.targets = _extract_get_targets(line)

    # --- Helpers ---

    @staticmethod
    def _parse_args(tokens: list[str], schema: dict[str, type]) -> dict[str, Any]:
        """Parse SDC-style -flag value arguments."""
        result = {}
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.startswith("-"):
                key = token[1:]  # strip leading dash
                expected_type = schema.get(token, str)
                if expected_type is bool:
                    result[key] = True
                elif i + 1 < len(tokens):
                    i += 1
                    val = tokens[i]
                    # Value might extend to include [get_*] brackets
                    # Reassemble bracketed expressions
                    if "[" in val and "]" not in val:
                        while i + 1 < len(tokens) and "]" not in val:
                            i += 1
                            val += " " + tokens[i]
                    try:
                        result[key] = expected_type(val)
                    except (ValueError, TypeError):
                        result[key] = val
            i += 1
        return result

    @staticmethod
    def _find_numeric_arg(tokens: list[str]) -> float | int | None:
        """Find the first positional numeric argument."""
        for token in tokens:
            if token.startswith("-") or token.startswith("["):
                continue
            try:
                if "." in token:
                    return float(token)
                return int(token)
            except ValueError:
                continue
        return None

    def add_to_graph(
        self,
        result: SDCExtractionResult,
        graph: nx.DiGraph,
        design_name: str = "design",
    ):
        """Add SDC constraints and clocks to an existing knowledge graph.

        Links constraints to ports/clocks/modules found in the graph.
        """
        # Build port-to-clock mapping from primary clock definitions
        # e.g., create_clock ... -name clk_core [get_ports clk] -> clk -> clk_core
        port_to_clock: dict[str, str] = {}
        for clk_name, clk_data in result.clocks.items():
            if not clk_data.get("generated") and clk_data.get("port"):
                port_to_clock[clk_data["port"]] = clk_name

        # Add clock nodes from SDC
        for clk_name, clk_data in result.clocks.items():
            clk_id = f"{design_name}::clk::{clk_name}"
            if graph.has_node(clk_id):
                # Merge SDC info into existing clock node
                nx.set_node_attributes(graph, {clk_id: clk_data})
            else:
                graph.add_node(clk_id, **clk_data)

            # Link generated clock to source
            if clk_data.get("generated") and clk_data.get("source_clock"):
                src = clk_data["source_clock"]
                # Resolve port name to clock name if needed
                if src in port_to_clock:
                    src = port_to_clock[src]
                src_clk_id = f"{design_name}::clk::{src}"
                if graph.has_node(src_clk_id):
                    graph.add_edge(clk_id, src_clk_id,
                                   relation="derived_from",
                                   divide_by=clk_data.get("divide_by"),
                                   multiply_by=clk_data.get("multiply_by"))

        # Add constraint nodes and link to targets
        for constraint in result.constraints:
            cid = constraint.constraint_id
            graph.add_node(cid, **{
                "type": "Constraint",
                "constraint_type": constraint.constraint_type,
                "raw_line": constraint.raw_line,
                "line_number": constraint.line_number,
                "file_path": constraint.file_path,
                **{k: v for k, v in constraint.attributes.items()
                   if not isinstance(v, (list, dict))},
            })

            # Link constraint to clock
            if constraint.source_clock:
                clk_id = f"{design_name}::clk::{constraint.source_clock}"
                if graph.has_node(clk_id):
                    graph.add_edge(cid, clk_id, relation="references_clock")

            # Link constraint to targets (ports, pins, modules)
            for target in constraint.targets:
                linked = self._link_target(graph, cid, target, design_name)
                if not linked:
                    logger.debug("Could not link constraint %s target '%s' to graph", cid, target)

            # Link from/to targets for path constraints
            for key in ("from_targets", "to_targets"):
                for target in constraint.attributes.get(key, []):
                    self._link_target(graph, cid, target, design_name)

    def _link_target(
        self,
        graph: nx.DiGraph,
        constraint_id: str,
        target: str,
        design_name: str,
    ) -> bool:
        """Try to link a constraint to a target node in the graph.

        Tries matching as: port, pin (instance/port), clock, module.
        Returns True if a link was created.
        """
        linked = False

        # Try as port: search for any module's port matching this name
        for node_id, data in graph.nodes(data=True):
            if data.get("type") != "Port":
                continue
            if data.get("name") == target and node_id.startswith(f"{design_name}::"):
                graph.add_edge(constraint_id, node_id, relation="applies_to")
                graph.add_edge(node_id, constraint_id, relation="constrained_by")
                linked = True

        # Try as pin: target may be "instance/pin" (e.g., u_baud_gen/baud_tick)
        if "/" in target:
            parts = target.split("/")
            inst_name = parts[0]
            pin_name = re.sub(r'\[.*\]', '', parts[-1])
            # Find the instance
            for node_id, data in graph.nodes(data=True):
                if data.get("type") == "Instance" and data.get("name") == inst_name:
                    graph.add_edge(constraint_id, node_id, relation="applies_to")
                    linked = True
                    # Also link to the child module's port
                    child_mod = data.get("child_module", "")
                    port_id = f"{child_mod}::{pin_name}"
                    if graph.has_node(port_id):
                        graph.add_edge(constraint_id, port_id, relation="applies_to")

        # Try as clock
        clk_id = f"{design_name}::clk::{target}"
        if graph.has_node(clk_id):
            graph.add_edge(constraint_id, clk_id, relation="applies_to")
            linked = True

        # Try as module
        mod_id = f"{design_name}::{target}"
        if graph.has_node(mod_id) and graph.nodes[mod_id].get("type") == "Module":
            graph.add_edge(constraint_id, mod_id, relation="applies_to")
            linked = True

        return linked


def print_sdc_summary(result: SDCExtractionResult):
    """Print a summary of parsed SDC constraints."""
    print(f"  SDC files parsed: {len(result.file_paths)}")
    print(f"  Total constraints: {len(result.constraints)}")
    print(f"  Clocks defined: {len(result.clocks)}")

    type_counts: dict[str, int] = {}
    for c in result.constraints:
        type_counts[c.constraint_type] = type_counts.get(c.constraint_type, 0) + 1
    print("  By type:")
    for t, count in sorted(type_counts.items()):
        print(f"    {t}: {count}")
