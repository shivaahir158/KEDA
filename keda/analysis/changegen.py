"""
Controlled design-change generation and ground-truth computation for KEDA-Bench.

Generates 7 categories of deterministic, reproducible RTL / SDC mutations:
  1. Parameter modifications
  2. Interface-width changes
  3. Clock changes
  4. Reset changes
  5. Module-dependency changes
  6. Constraint changes
  7. Hierarchy modifications

Each change is represented as a DesignChange with:
  - A textual patch (old_text -> new_text per file)
  - Metadata (change type, affected module, description)
  - An automatically computed ground-truth impact set

Ground truth is derived by graph traversal from the changed node(s),
then validated (optionally) by re-running Yosys on the mutated design.
"""

from __future__ import annotations

import copy
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class ChangeType(str, Enum):
    PARAMETER = "parameter"
    WIDTH = "width"
    CLOCK = "clock"
    RESET = "reset"
    DEPENDENCY = "dependency"
    CONSTRAINT = "constraint"
    HIERARCHY = "hierarchy"


@dataclass
class FilePatch:
    """A single text substitution in one file."""
    file_path: str
    old_text: str
    new_text: str
    line_number: int | None = None

    def apply(self, content: str) -> str:
        """Apply this patch to file content. Returns modified content."""
        if self.old_text not in content:
            raise ValueError(
                f"Patch target not found in {self.file_path}: "
                f"{self.old_text!r:.80}"
            )
        return content.replace(self.old_text, self.new_text, 1)

    def revert(self, content: str) -> str:
        """Reverse this patch."""
        if self.new_text not in content:
            raise ValueError(
                f"Revert target not found in {self.file_path}: "
                f"{self.new_text!r:.80}"
            )
        return content.replace(self.new_text, self.old_text, 1)


@dataclass
class DesignChange:
    """A single controlled design change with metadata and ground truth."""
    change_id: str
    change_type: ChangeType
    description: str
    target_module: str
    patches: list[FilePatch] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Ground truth: node IDs of every artifact truly impacted
    ground_truth: set[str] = field(default_factory=set)
    # Ground truth broken out by artifact type
    ground_truth_by_type: dict[str, set[str]] = field(default_factory=dict)
    # Ground truth broken out by hop distance from change origin
    ground_truth_by_hop: dict[int, set[str]] = field(default_factory=dict)

    # Whether Yosys synthesis still succeeds after this change
    synthesis_valid: bool | None = None

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "change_id": self.change_id,
            "change_type": self.change_type.value,
            "description": self.description,
            "target_module": self.target_module,
            "patches": [
                {"file_path": p.file_path, "old_text": p.old_text,
                 "new_text": p.new_text, "line_number": p.line_number}
                for p in self.patches
            ],
            "metadata": self.metadata,
            "ground_truth": sorted(self.ground_truth),
            "ground_truth_by_type": {
                k: sorted(v) for k, v in self.ground_truth_by_type.items()
            },
            "ground_truth_by_hop": {
                str(k): sorted(v) for k, v in self.ground_truth_by_hop.items()
            },
            "synthesis_valid": self.synthesis_valid,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DesignChange":
        """Deserialize from dict."""
        dc = cls(
            change_id=d["change_id"],
            change_type=ChangeType(d["change_type"]),
            description=d["description"],
            target_module=d["target_module"],
            patches=[
                FilePatch(p["file_path"], p["old_text"], p["new_text"],
                          p.get("line_number"))
                for p in d["patches"]
            ],
            metadata=d.get("metadata", {}),
            ground_truth=set(d.get("ground_truth", [])),
            ground_truth_by_type={
                k: set(v) for k, v in d.get("ground_truth_by_type", {}).items()
            },
            ground_truth_by_hop={
                int(k): set(v) for k, v in d.get("ground_truth_by_hop", {}).items()
            },
            synthesis_valid=d.get("synthesis_valid"),
        )
        return dc


@dataclass
class ChangeSet:
    """A collection of changes for one design."""
    design_name: str
    changes: list[DesignChange] = field(default_factory=list)

    def by_type(self) -> dict[ChangeType, list[DesignChange]]:
        groups: dict[ChangeType, list[DesignChange]] = {}
        for c in self.changes:
            groups.setdefault(c.change_type, []).append(c)
        return groups

    def save(self, path: str | Path):
        """Save to JSON."""
        with open(path, "w") as f:
            json.dump({
                "design_name": self.design_name,
                "changes": [c.to_dict() for c in self.changes],
            }, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ChangeSet":
        """Load from JSON."""
        with open(path) as f:
            d = json.load(f)
        cs = cls(design_name=d["design_name"])
        cs.changes = [DesignChange.from_dict(c) for c in d["changes"]]
        return cs


# ---------------------------------------------------------------------------
# Ground truth computation
# ---------------------------------------------------------------------------

class GroundTruthComputer:
    """Compute ground-truth impact sets from the knowledge graph.

    For each DesignChange, determines which graph nodes are truly affected
    by traversing from the changed module outward, following semantically
    appropriate edges for each change type.
    """

    # Edges to traverse for cross-artifact ground truth expansion
    EXPANSION_EDGES = {
        # From Module outward
        "depended_by", "instantiates",
        "has_port", "has_instance", "contains_register",
        "has_parameter",
        # From Port onward
        "constrained_by", "drives",
        # From Register onward
        "clocked_by",
        # From Clock onward
        "applies_to", "derived_from",
        # From Constraint onward (reverse links)
        "references_clock",
        # Verification
        "verified_by", "verifies", "covers",
        # Git (don't include — commits are inputs, not outputs)
    }

    # Change-type-specific edge sets for more precise ground truth
    CHANGE_TYPE_EDGES: dict[ChangeType, frozenset[str]] = {
        ChangeType.PARAMETER: frozenset({
            "depended_by", "has_parameter", "has_port",
            "constrained_by", "verified_by", "covers",
        }),
        ChangeType.WIDTH: frozenset({
            "depended_by", "has_port", "drives", "constrained_by",
            "verified_by", "covers", "instantiates", "has_instance",
        }),
        ChangeType.CLOCK: frozenset({
            "depended_by", "contains_register", "clocked_by",
            "constrained_by", "applies_to", "derived_from",
            "verified_by", "covers", "has_port",
        }),
        ChangeType.RESET: frozenset({
            "depended_by", "contains_register", "has_port",
            "verified_by", "covers",
        }),
        ChangeType.DEPENDENCY: frozenset({
            "depended_by", "instantiates", "has_port", "has_instance",
            "constrained_by", "verified_by", "covers",
            "contains_register", "clocked_by",
        }),
        ChangeType.CONSTRAINT: frozenset({
            "applies_to", "references_clock", "constrained_by",
        }),
        ChangeType.HIERARCHY: frozenset({
            "depended_by", "instantiates", "has_port", "has_instance",
            "constrained_by", "verified_by", "covers",
            "contains_register", "clocked_by",
        }),
    }

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    def compute(
        self,
        change: DesignChange,
        design_name: str,
        max_depth: int = 6,
        use_type_specific_edges: bool = True,
    ) -> DesignChange:
        """Compute ground truth for a single change. Modifies the change in place."""
        G = self.graph
        module_id = f"{design_name}::{change.target_module}"

        if module_id not in G:
            logger.warning("Module %s not in graph", module_id)
            return change

        # Select edge set
        if use_type_specific_edges and change.change_type in self.CHANGE_TYPE_EDGES:
            allowed_edges = self.CHANGE_TYPE_EDGES[change.change_type]
        else:
            allowed_edges = self.EXPANSION_EDGES

        # BFS from the changed module
        visited: dict[str, int] = {module_id: 0}
        frontier = {module_id}

        for depth in range(1, max_depth + 1):
            next_frontier = set()
            for node in frontier:
                for _, neighbor, edge_data in G.out_edges(node, data=True):
                    relation = edge_data.get("relation", "")
                    if relation not in allowed_edges:
                        continue
                    if neighbor in visited:
                        continue
                    visited[neighbor] = depth
                    next_frontier.add(neighbor)
            frontier = next_frontier
            if not frontier:
                break

        # Remove the changed module itself from ground truth
        visited.pop(module_id, None)

        # For constraint changes, start from the constraint node instead
        if change.change_type == ChangeType.CONSTRAINT:
            constraint_targets = self._find_constraint_targets(change, design_name)
            for ct in constraint_targets:
                if ct in G and ct not in visited:
                    visited[ct] = 0
                    # Expand from constraint
                    c_frontier = {ct}
                    for depth in range(1, max_depth + 1):
                        c_next = set()
                        for node in c_frontier:
                            for _, nb, ed in G.out_edges(node, data=True):
                                rel = ed.get("relation", "")
                                if rel in allowed_edges and nb not in visited:
                                    visited[nb] = depth
                                    c_next.add(nb)
                        c_frontier = c_next
                        if not c_frontier:
                            break

        # Build ground truth sets
        change.ground_truth = set(visited.keys())

        change.ground_truth_by_type = {}
        for node_id, hop in visited.items():
            ntype = G.nodes.get(node_id, {}).get("type", "Unknown")
            change.ground_truth_by_type.setdefault(ntype, set()).add(node_id)

        change.ground_truth_by_hop = {}
        for node_id, hop in visited.items():
            change.ground_truth_by_hop.setdefault(hop, set()).add(node_id)

        return change

    def _find_constraint_targets(
        self, change: DesignChange, design_name: str
    ) -> list[str]:
        """For constraint changes, find the constraint nodes being modified."""
        targets = []
        # Look for constraint nodes matching metadata
        old_line = change.metadata.get("old_line", "")
        for node_id, data in self.graph.nodes(data=True):
            if data.get("type") != "Constraint":
                continue
            if old_line and old_line in data.get("raw_line", ""):
                targets.append(node_id)
        return targets

    def compute_all(
        self,
        changeset: ChangeSet,
        max_depth: int = 6,
    ) -> ChangeSet:
        """Compute ground truth for every change in a ChangeSet."""
        for change in changeset.changes:
            self.compute(change, changeset.design_name, max_depth)
        return changeset


# ---------------------------------------------------------------------------
# Change generators
# ---------------------------------------------------------------------------

class ChangeGenerator:
    """Generate controlled design changes from a knowledge graph and RTL source.

    Scans the graph for mutable elements (parameters, ports, clocks, etc.)
    and produces DesignChange objects with patches and metadata.

    Usage:
        gen = ChangeGenerator(graph, rtl_sources, design_name="uart")
        changeset = gen.generate_all()
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        rtl_sources: dict[str, str],
        design_name: str = "design",
        sdc_sources: dict[str, str] | None = None,
    ):
        """
        Args:
            graph: The built knowledge graph.
            rtl_sources: Mapping file_path -> file_content for all RTL files.
            design_name: Design name matching graph node ID prefix.
            sdc_sources: Mapping file_path -> file_content for SDC files.
        """
        self.graph = graph
        self.rtl_sources = rtl_sources
        self.design_name = design_name
        self.sdc_sources = sdc_sources or {}
        self._counter = 0

    def _next_id(self, ctype: ChangeType) -> str:
        self._counter += 1
        return f"{self.design_name}::change::{ctype.value}::{self._counter:04d}"

    # ------------------------------------------------------------------
    # Generate all change types
    # ------------------------------------------------------------------

    def generate_all(
        self,
        max_per_type: int | None = None,
    ) -> ChangeSet:
        """Generate changes of every type. Returns a ChangeSet."""
        cs = ChangeSet(design_name=self.design_name)

        generators = [
            self.generate_parameter_changes,
            self.generate_width_changes,
            self.generate_clock_changes,
            self.generate_reset_changes,
            self.generate_dependency_changes,
            self.generate_constraint_changes,
            self.generate_hierarchy_changes,
        ]

        for gen_fn in generators:
            try:
                changes = gen_fn()
                if max_per_type is not None:
                    changes = changes[:max_per_type]
                cs.changes.extend(changes)
            except Exception as e:
                logger.error("Generator %s failed: %s", gen_fn.__name__, e)

        return cs

    # ------------------------------------------------------------------
    # 1. Parameter modifications
    # ------------------------------------------------------------------

    def generate_parameter_changes(self) -> list[DesignChange]:
        """Generate parameter value modifications.

        For each parameter with a numeric default, produce variants:
        - double the value
        - halve the value
        - set to 1 (boundary)
        """
        changes = []
        G = self.graph

        for node_id, data in G.nodes(data=True):
            if data.get("type") != "Parameter":
                continue

            param_name = data.get("name", "")
            default_val = data.get("default_value", 0)
            module_id = data.get("module", "")
            module_data = G.nodes.get(module_id, {})
            module_name = module_data.get("name", "")
            src_file = module_data.get("src_file", "")

            if not src_file or default_val is None or default_val == 0:
                continue

            content = self._get_source(src_file)
            if content is None:
                continue

            # Find the parameter declaration line
            # Matches: parameter PARAM_NAME = VALUE
            patterns = [
                (rf'(parameter\s+{re.escape(param_name)}\s*=\s*)(\d[\d_]*)',
                 "decimal"),
                (rf'(parameter\s+{re.escape(param_name)}\s*=\s*)(\d+\'[hdbo][\da-fA-F_]+)',
                 "sized"),
            ]

            for pat, fmt in patterns:
                m = re.search(pat, content)
                if not m:
                    continue
                prefix = m.group(1)
                old_val_str = m.group(2)

                # Generate variants
                variants = self._parameter_variants(default_val, old_val_str)
                for new_val, new_val_str, variant_desc in variants:
                    old_text = prefix + old_val_str
                    new_text = prefix + new_val_str

                    changes.append(DesignChange(
                        change_id=self._next_id(ChangeType.PARAMETER),
                        change_type=ChangeType.PARAMETER,
                        description=(
                            f"Change parameter {param_name} in {module_name} "
                            f"from {old_val_str} to {new_val_str} ({variant_desc})"
                        ),
                        target_module=module_name,
                        patches=[FilePatch(src_file, old_text, new_text)],
                        metadata={
                            "parameter_name": param_name,
                            "old_value": default_val,
                            "new_value": new_val,
                            "variant": variant_desc,
                        },
                    ))
                break  # only first pattern match

        return changes

    def _parameter_variants(
        self, value: int, value_str: str
    ) -> list[tuple[int, str, str]]:
        """Generate numeric variants of a parameter value."""
        variants = []
        if value >= 2:
            doubled = value * 2
            variants.append((doubled, self._format_like(doubled, value_str), "double"))
        if value >= 4:
            halved = value // 2
            variants.append((halved, self._format_like(halved, value_str), "halve"))
        if value > 1:
            variants.append((1, self._format_like(1, value_str), "boundary_min"))
        return variants

    @staticmethod
    def _format_like(value: int, template: str) -> str:
        """Format a number to look like the template (with underscores, etc.)."""
        # If template uses underscores (e.g., 50_000_000), format similarly
        if "_" in template:
            s = str(value)
            # Insert underscores every 3 digits from right
            parts = []
            while len(s) > 3:
                parts.append(s[-3:])
                s = s[:-3]
            parts.append(s)
            return "_".join(reversed(parts))
        return str(value)

    # ------------------------------------------------------------------
    # 2. Interface-width changes
    # ------------------------------------------------------------------

    def generate_width_changes(self) -> list[DesignChange]:
        """Generate port width modifications.

        For each multi-bit port, produce:
        - double the width
        - halve the width (if >= 4)
        """
        changes = []
        G = self.graph

        # Collect ports with width > 1
        seen_ports: set[tuple[str, str]] = set()  # (module_name, port_name)

        for node_id, data in G.nodes(data=True):
            if data.get("type") != "Port":
                continue
            width = data.get("width", 1)
            if width <= 1:
                continue

            module_id = data.get("module", "")
            module_data = G.nodes.get(module_id, {})
            module_name = module_data.get("name", "")
            port_name = data.get("name", "")
            direction = data.get("direction", "")
            src_file = module_data.get("src_file", "")

            key = (module_name, port_name)
            if key in seen_ports or not src_file:
                continue
            seen_ports.add(key)

            content = self._get_source(src_file)
            if content is None:
                continue

            # Match port declaration:  input wire [WIDTH-1:0] port_name
            # or parameterized: [DATA_BITS-1:0] port_name
            pat = rf'(\b(?:input|output|inout)\s+(?:wire|reg|logic)?\s*)\[([^\]]+)\]\s+{re.escape(port_name)}'
            m = re.search(pat, content)
            if not m:
                continue

            prefix = m.group(1)
            range_expr = m.group(2)

            # Try to parse as [N-1:0] for concrete width
            range_m = re.match(r'(\d+)\s*:\s*0', range_expr)
            if range_m:
                msb = int(range_m.group(1))
                actual_width = msb + 1

                # Double
                new_msb = actual_width * 2 - 1
                old_text = f"{prefix}[{range_expr}] {port_name}"
                new_text = f"{prefix}[{new_msb}:0] {port_name}"
                changes.append(DesignChange(
                    change_id=self._next_id(ChangeType.WIDTH),
                    change_type=ChangeType.WIDTH,
                    description=(
                        f"Widen port {port_name} in {module_name} "
                        f"from {actual_width} to {actual_width * 2} bits"
                    ),
                    target_module=module_name,
                    patches=[FilePatch(src_file, old_text, new_text)],
                    metadata={
                        "port_name": port_name,
                        "direction": direction,
                        "old_width": actual_width,
                        "new_width": actual_width * 2,
                        "variant": "double",
                    },
                ))

                # Halve (if width >= 4)
                if actual_width >= 4:
                    half_msb = actual_width // 2 - 1
                    new_text_h = f"{prefix}[{half_msb}:0] {port_name}"
                    changes.append(DesignChange(
                        change_id=self._next_id(ChangeType.WIDTH),
                        change_type=ChangeType.WIDTH,
                        description=(
                            f"Narrow port {port_name} in {module_name} "
                            f"from {actual_width} to {actual_width // 2} bits"
                        ),
                        target_module=module_name,
                        patches=[FilePatch(src_file, old_text, new_text_h)],
                        metadata={
                            "port_name": port_name,
                            "direction": direction,
                            "old_width": actual_width,
                            "new_width": actual_width // 2,
                            "variant": "halve",
                        },
                    ))

        return changes

    # ------------------------------------------------------------------
    # 3. Clock changes
    # ------------------------------------------------------------------

    def generate_clock_changes(self) -> list[DesignChange]:
        """Generate clock signal changes.

        - Swap posedge/negedge in always blocks
        - Rename clock signal in a module
        """
        changes = []
        G = self.graph

        for node_id, data in G.nodes(data=True):
            if data.get("type") != "Register":
                continue

            clk_edge = data.get("clock_edge", "")
            clk_signal = data.get("clock_signal", "")
            module_id = data.get("module", "")
            module_data = G.nodes.get(module_id, {})
            module_name = module_data.get("name", "")
            src_file = module_data.get("src_file", "")

            if not src_file or not clk_signal:
                continue

            content = self._get_source(src_file)
            if content is None:
                continue

            # Edge swap: posedge clk -> negedge clk (or vice versa)
            if clk_edge in ("posedge", "negedge"):
                new_edge = "negedge" if clk_edge == "posedge" else "posedge"
                old_text = f"{clk_edge} {clk_signal}"
                new_text = f"{new_edge} {clk_signal}"

                if old_text in content:
                    # Deduplicate: only one change per (module, old_text)
                    already = any(
                        c.target_module == module_name
                        and c.metadata.get("variant") == "edge_swap"
                        and c.metadata.get("clock_signal") == clk_signal
                        for c in changes
                    )
                    if not already:
                        changes.append(DesignChange(
                            change_id=self._next_id(ChangeType.CLOCK),
                            change_type=ChangeType.CLOCK,
                            description=(
                                f"Swap clock edge in {module_name}: "
                                f"{clk_edge} {clk_signal} -> {new_edge} {clk_signal}"
                            ),
                            target_module=module_name,
                            patches=[FilePatch(src_file, old_text, new_text)],
                            metadata={
                                "clock_signal": clk_signal,
                                "old_edge": clk_edge,
                                "new_edge": new_edge,
                                "variant": "edge_swap",
                            },
                        ))

        # Clock rename: change clock port name in a module
        for node_id, data in G.nodes(data=True):
            if data.get("type") != "Port":
                continue
            if data.get("direction") != "input" or data.get("width", 0) != 1:
                continue

            port_name = data.get("name", "")
            if not self._is_clock_name(port_name):
                continue

            module_id = data.get("module", "")
            module_data = G.nodes.get(module_id, {})
            module_name = module_data.get("name", "")
            src_file = module_data.get("src_file", "")

            if not src_file:
                continue

            content = self._get_source(src_file)
            if content is None:
                continue

            new_name = "clk_alt"
            if new_name == port_name:
                new_name = "clk_renamed"

            # Only do this for non-top modules to avoid massive cascade
            if module_data.get("is_top"):
                continue

            # Substitute all occurrences of the clock name in this file
            if content.count(port_name) >= 2:  # at least port decl + usage
                changes.append(DesignChange(
                    change_id=self._next_id(ChangeType.CLOCK),
                    change_type=ChangeType.CLOCK,
                    description=(
                        f"Rename clock port in {module_name}: "
                        f"{port_name} -> {new_name}"
                    ),
                    target_module=module_name,
                    patches=[FilePatch(src_file, port_name, new_name)],
                    metadata={
                        "old_clock_name": port_name,
                        "new_clock_name": new_name,
                        "variant": "clock_rename",
                        "replace_all": True,
                    },
                ))

        return changes

    # ------------------------------------------------------------------
    # 4. Reset changes
    # ------------------------------------------------------------------

    def generate_reset_changes(self) -> list[DesignChange]:
        """Generate reset-related changes.

        - Swap reset polarity (active-low -> active-high)
        - Change async reset to sync reset
        """
        changes = []
        G = self.graph

        seen_modules: set[str] = set()

        for node_id, data in G.nodes(data=True):
            if data.get("type") != "Register":
                continue

            module_id = data.get("module", "")
            module_data = G.nodes.get(module_id, {})
            module_name = module_data.get("name", "")
            src_file = module_data.get("src_file", "")
            has_async = data.get("has_async_reset", False)
            polarity = data.get("reset_polarity", "")

            if not src_file or module_name in seen_modules:
                continue

            content = self._get_source(src_file)
            if content is None:
                continue

            # Polarity swap: negedge rst_n -> posedge rst_n, !rst_n -> rst_n
            if has_async and polarity == "active_low":
                if "negedge rst_n" in content and "!rst_n" in content:
                    seen_modules.add(module_name)

                    patches = [
                        FilePatch(src_file, "negedge rst_n", "posedge rst_n"),
                        FilePatch(src_file, "!rst_n", "rst_n"),
                    ]
                    # Verify both patches can apply independently
                    # (the second !rst_n should be in the if condition)
                    changes.append(DesignChange(
                        change_id=self._next_id(ChangeType.RESET),
                        change_type=ChangeType.RESET,
                        description=(
                            f"Swap reset polarity in {module_name}: "
                            f"active-low -> active-high"
                        ),
                        target_module=module_name,
                        patches=patches,
                        metadata={
                            "old_polarity": "active_low",
                            "new_polarity": "active_high",
                            "variant": "polarity_swap",
                        },
                    ))

            # Async -> sync: remove rst_n from sensitivity list
            if has_async:
                old_sens = "posedge clk or negedge rst_n"
                new_sens = "posedge clk"
                if old_sens in content and module_name not in seen_modules:
                    seen_modules.add(module_name)
                    changes.append(DesignChange(
                        change_id=self._next_id(ChangeType.RESET),
                        change_type=ChangeType.RESET,
                        description=(
                            f"Change async reset to sync in {module_name}"
                        ),
                        target_module=module_name,
                        patches=[FilePatch(src_file, old_sens, new_sens)],
                        metadata={
                            "old_reset_type": "async",
                            "new_reset_type": "sync",
                            "variant": "async_to_sync",
                        },
                    ))

        return changes

    # ------------------------------------------------------------------
    # 5. Module-dependency changes
    # ------------------------------------------------------------------

    def generate_dependency_changes(self) -> list[DesignChange]:
        """Generate module instantiation changes.

        - Remove an instantiation
        - Replace an instantiation with a stub
        """
        changes = []
        G = self.graph

        for node_id, data in G.nodes(data=True):
            if data.get("type") != "Instance":
                continue

            inst_name = data.get("name", "")
            parent_id = data.get("parent_module", "")
            child_id = data.get("child_module", "")
            parent_data = G.nodes.get(parent_id, {})
            child_data = G.nodes.get(child_id, {})
            parent_name = parent_data.get("name", "")
            child_name = child_data.get("name", "")
            src_file = parent_data.get("src_file", "")

            if not src_file:
                continue

            content = self._get_source(src_file)
            if content is None:
                continue

            # Find the instantiation block: child_name ... inst_name (...)  ;
            # Use regex to find the full instantiation
            m = self._find_instantiation(content, child_name, inst_name)
            if not m:
                continue

            inst_text = m.group(0)

            # Generate: comment out the instantiation
            commented = "// REMOVED: " + inst_text.replace("\n", "\n// REMOVED: ")
            changes.append(DesignChange(
                change_id=self._next_id(ChangeType.DEPENDENCY),
                change_type=ChangeType.DEPENDENCY,
                description=(
                    f"Remove instantiation of {child_name} "
                    f"(instance {inst_name}) from {parent_name}"
                ),
                target_module=parent_name,
                patches=[FilePatch(src_file, inst_text, commented)],
                metadata={
                    "instance_name": inst_name,
                    "child_module": child_name,
                    "parent_module": parent_name,
                    "variant": "remove_instance",
                },
            ))

        return changes

    # ------------------------------------------------------------------
    # 6. Constraint changes
    # ------------------------------------------------------------------

    def generate_constraint_changes(self) -> list[DesignChange]:
        """Generate SDC constraint modifications.

        - Tighten clock period (halve)
        - Loosen clock period (double)
        - Remove a false path
        - Change I/O delay values
        """
        changes = []
        G = self.graph

        for node_id, data in G.nodes(data=True):
            if data.get("type") != "Constraint":
                continue

            ctype = data.get("constraint_type", "")
            raw_line = data.get("raw_line", "")
            file_path = data.get("file_path", "")

            if not file_path or not raw_line:
                continue

            content = self._get_sdc_source(file_path)
            if content is None:
                continue

            if raw_line not in content:
                continue

            # Find the target module for this constraint
            target_mod = self._constraint_target_module(node_id)

            if ctype == "create_clock":
                period = data.get("period")
                if period and isinstance(period, (int, float)):
                    # Tighten (halve period)
                    new_period = period / 2
                    old_p_str = str(period)
                    new_p_str = str(new_period)
                    new_line = raw_line.replace(old_p_str, new_p_str)
                    changes.append(DesignChange(
                        change_id=self._next_id(ChangeType.CONSTRAINT),
                        change_type=ChangeType.CONSTRAINT,
                        description=(
                            f"Tighten clock period from {period}ns to {new_period}ns"
                        ),
                        target_module=target_mod,
                        patches=[FilePatch(file_path, raw_line, new_line)],
                        metadata={
                            "constraint_type": ctype,
                            "old_period": period,
                            "new_period": new_period,
                            "old_line": raw_line,
                            "variant": "tighten_clock",
                        },
                    ))

                    # Loosen (double period)
                    loose_period = period * 2
                    loose_p_str = str(loose_period)
                    loose_line = raw_line.replace(old_p_str, loose_p_str)
                    changes.append(DesignChange(
                        change_id=self._next_id(ChangeType.CONSTRAINT),
                        change_type=ChangeType.CONSTRAINT,
                        description=(
                            f"Loosen clock period from {period}ns to {loose_period}ns"
                        ),
                        target_module=target_mod,
                        patches=[FilePatch(file_path, raw_line, loose_line)],
                        metadata={
                            "constraint_type": ctype,
                            "old_period": period,
                            "new_period": loose_period,
                            "old_line": raw_line,
                            "variant": "loosen_clock",
                        },
                    ))

            elif ctype == "set_false_path":
                # Remove the false path
                changes.append(DesignChange(
                    change_id=self._next_id(ChangeType.CONSTRAINT),
                    change_type=ChangeType.CONSTRAINT,
                    description=f"Remove false path: {raw_line.strip()}",
                    target_module=target_mod,
                    patches=[FilePatch(file_path, raw_line,
                                       f"# REMOVED: {raw_line}")],
                    metadata={
                        "constraint_type": ctype,
                        "old_line": raw_line,
                        "variant": "remove_false_path",
                    },
                ))

            elif ctype in ("set_input_delay", "set_output_delay"):
                # Change delay value
                delay = data.get("delay")
                if delay and isinstance(delay, (int, float)):
                    new_delay = delay * 2
                    old_d_str = str(delay)
                    new_d_str = str(new_delay)
                    if old_d_str in raw_line:
                        new_line = raw_line.replace(old_d_str, new_d_str, 1)
                        changes.append(DesignChange(
                            change_id=self._next_id(ChangeType.CONSTRAINT),
                            change_type=ChangeType.CONSTRAINT,
                            description=(
                                f"Change {ctype} from {delay} to {new_delay}"
                            ),
                            target_module=target_mod,
                            patches=[FilePatch(file_path, raw_line, new_line)],
                            metadata={
                                "constraint_type": ctype,
                                "old_delay": delay,
                                "new_delay": new_delay,
                                "old_line": raw_line,
                                "variant": "change_delay",
                            },
                        ))

        return changes

    # ------------------------------------------------------------------
    # 7. Hierarchy modifications
    # ------------------------------------------------------------------

    def generate_hierarchy_changes(self) -> list[DesignChange]:
        """Generate hierarchy modifications.

        - Inline a child module (flatten one level)
          Represented as removing the instantiation and adding a comment.
          A full inline would require duplicating the child logic, which is
          too complex for automated generation.  Instead we produce a
          "flatten marker" change that signals the intent.
        """
        changes = []
        G = self.graph

        for node_id, data in G.nodes(data=True):
            if data.get("type") != "Instance":
                continue

            inst_name = data.get("name", "")
            parent_id = data.get("parent_module", "")
            child_id = data.get("child_module", "")
            parent_data = G.nodes.get(parent_id, {})
            child_data = G.nodes.get(child_id, {})
            parent_name = parent_data.get("name", "")
            child_name = child_data.get("name", "")
            src_file = parent_data.get("src_file", "")

            if not src_file:
                continue

            content = self._get_source(src_file)
            if content is None:
                continue

            # Find the instantiation (same regex as dependency changes)
            m = self._find_instantiation(content, child_name, inst_name)
            if not m:
                continue

            inst_text = m.group(0)

            # Produce a "flatten" change: replace instantiation with
            # a wire-assignment stub representing inlined logic
            stub_lines = [
                f"// FLATTENED: {child_name} instance {inst_name}",
                f"// Original sub-module logic would be inlined here",
            ]
            # Connect output ports of the instance to placeholder assigns
            port_dirs = data.get("port_directions", {})
            port_conns = data.get("port_connections", {})
            for port, direction in port_dirs.items():
                if direction == "output":
                    nets = port_conns.get(port, [])
                    for net in nets:
                        stub_lines.append(
                            f"assign {net} = 1'b0; // stub for {inst_name}.{port}"
                        )

            stub = "\n    ".join(stub_lines)
            changes.append(DesignChange(
                change_id=self._next_id(ChangeType.HIERARCHY),
                change_type=ChangeType.HIERARCHY,
                description=(
                    f"Flatten {child_name} (instance {inst_name}) "
                    f"into {parent_name}"
                ),
                target_module=parent_name,
                patches=[FilePatch(src_file, inst_text, stub)],
                metadata={
                    "instance_name": inst_name,
                    "child_module": child_name,
                    "parent_module": parent_name,
                    "variant": "flatten",
                },
            ))

        return changes

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_source(self, file_path: str) -> str | None:
        """Get RTL source content, trying multiple path resolutions."""
        if file_path in self.rtl_sources:
            return self.rtl_sources[file_path]
        # Try basename match
        basename = Path(file_path).name
        for k, v in self.rtl_sources.items():
            if Path(k).name == basename:
                return v
        return None

    def _get_sdc_source(self, file_path: str) -> str | None:
        """Get SDC source content."""
        if file_path in self.sdc_sources:
            return self.sdc_sources[file_path]
        basename = Path(file_path).name
        for k, v in self.sdc_sources.items():
            if Path(k).name == basename:
                return v
        return None

    def _constraint_target_module(self, constraint_node_id: str) -> str:
        """Find the module most closely associated with a constraint."""
        G = self.graph
        # Follow applies_to edges
        for _, target, d in G.out_edges(constraint_node_id, data=True):
            if d.get("relation") == "applies_to":
                tdata = G.nodes.get(target, {})
                if tdata.get("type") == "Module":
                    return tdata.get("name", "")
                if tdata.get("type") == "Port":
                    mod_id = tdata.get("module", "")
                    mod_data = G.nodes.get(mod_id, {})
                    return mod_data.get("name", "")
                if tdata.get("type") == "Clock":
                    # Clock constraints affect all modules using this clock.
                    # Find a module that has a port constrained by this clock,
                    # or fall back to a module that has a clocked_by edge.
                    for n, nd in G.nodes(data=True):
                        if nd.get("type") == "Register":
                            for _, clk, ed in G.out_edges(n, data=True):
                                if ed.get("relation") == "clocked_by" and clk == target:
                                    reg_mod = nd.get("module", "")
                                    mod_data = G.nodes.get(reg_mod, {})
                                    mod_name = mod_data.get("name", "")
                                    if mod_name:
                                        return mod_name
        # Follow constrained_by reverse edges (port -> constraint)
        for src, tgt, d in G.edges(data=True):
            if tgt == constraint_node_id and d.get("relation") == "constrained_by":
                sdata = G.nodes.get(src, {})
                if sdata.get("type") == "Port":
                    mod_id = sdata.get("module", "")
                    mod_data = G.nodes.get(mod_id, {})
                    return mod_data.get("name", "")
        # Fall back to top module
        for n, nd in G.nodes(data=True):
            if nd.get("type") == "Module" and nd.get("is_top"):
                return nd.get("name", "")
        # Last resort: any module
        for n, nd in G.nodes(data=True):
            if nd.get("type") == "Module":
                return nd.get("name", "")
        return "unknown"

    @staticmethod
    def _find_instantiation(
        content: str, child_name: str, inst_name: str
    ) -> re.Match | None:
        """Find a module instantiation in Verilog source.

        Uses string scanning with balanced-paren matching to avoid
        catastrophic backtracking on large files.

        Returns a proxy object with .group(0) containing the full text,
        or None if not found.
        """
        search_start = 0
        while True:
            idx = content.find(inst_name, search_start)
            if idx < 0:
                return None

            # Verify this is an identifier boundary (not a substring)
            if idx > 0 and (content[idx - 1].isalnum() or content[idx - 1] == '_'):
                search_start = idx + 1
                continue
            end_ch_idx = idx + len(inst_name)
            if end_ch_idx < len(content) and (content[end_ch_idx].isalnum() or content[end_ch_idx] == '_'):
                search_start = idx + 1
                continue

            # inst_name should be followed by whitespace and (
            after = content[end_ch_idx:end_ch_idx + 50].lstrip()
            if not after.startswith('('):
                search_start = idx + 1
                continue

            # Look backwards for child_name — it starts the instantiation
            # Scan backward from idx, skipping whitespace and optional #(...)
            before = content[max(0, idx - 5000):idx]
            before_stripped = before.rstrip()

            # Pattern 1: "child_name inst_name" (no params)
            if before_stripped.endswith(child_name):
                cn_pos = before_stripped.rfind(child_name)
                inst_start = max(0, idx - 5000) + cn_pos
            # Pattern 2: "child_name #(...) inst_name" (with params)
            elif before_stripped.endswith(')'):
                # Scan backwards for matching (
                j = len(before_stripped) - 1
                depth = 0
                while j >= 0:
                    if before_stripped[j] == ')':
                        depth += 1
                    elif before_stripped[j] == '(':
                        depth -= 1
                        if depth == 0:
                            break
                    j -= 1
                if depth != 0:
                    search_start = idx + 1
                    continue
                # Before ( should be #
                pre = before_stripped[:j].rstrip()
                if not pre.endswith('#'):
                    search_start = idx + 1
                    continue
                pre = pre[:-1].rstrip()
                if not pre.endswith(child_name):
                    search_start = idx + 1
                    continue
                cn_pos = pre.rfind(child_name)
                inst_start = max(0, idx - 5000) + cn_pos
            else:
                search_start = idx + 1
                continue

            # Verify not a module definition
            line_start = content.rfind('\n', 0, inst_start) + 1
            line = content[line_start:inst_start].strip()
            if line.startswith('module'):
                search_start = idx + 1
                continue

            # Find the port list opening paren
            paren_idx = content.find('(', end_ch_idx)
            if paren_idx < 0:
                search_start = idx + 1
                continue

            # Scan balanced parens for port connections
            depth = 0
            i = paren_idx
            while i < len(content):
                if content[i] == '(':
                    depth += 1
                elif content[i] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                i += 1

            if depth != 0:
                search_start = idx + 1
                continue

            # Find semicolon
            semi_idx = content.find(';', i + 1)
            end = semi_idx + 1 if semi_idx >= 0 else i + 1

            inst_text = content[inst_start:end]

            class _MatchProxy:
                def __init__(self, text):
                    self._text = text
                def group(self, n=0):
                    return self._text
            return _MatchProxy(inst_text)

    @staticmethod
    def _is_clock_name(name: str) -> bool:
        name_lower = name.lower()
        return any(p in name_lower for p in [
            "clk", "clock", "ck", "sysclk", "pclk",
        ])


# ---------------------------------------------------------------------------
# Patch application & validation
# ---------------------------------------------------------------------------

def apply_changes(
    change: DesignChange,
    sources: dict[str, str],
) -> dict[str, str]:
    """Apply a DesignChange to source files. Returns modified sources dict."""
    result = dict(sources)
    for patch in change.patches:
        basename = Path(patch.file_path).name
        # Find the matching source key
        target_key = None
        for k in result:
            if Path(k).name == basename or k == patch.file_path:
                target_key = k
                break
        if target_key is None:
            raise ValueError(f"Source file not found for patch: {patch.file_path}")

        if patch.old_text not in result[target_key]:
            raise ValueError(
                f"Patch target not found in {target_key}: "
                f"{patch.old_text!r:.120}"
            )

        replace_all = change.metadata.get("replace_all", False)
        if replace_all:
            result[target_key] = result[target_key].replace(
                patch.old_text, patch.new_text
            )
        else:
            result[target_key] = result[target_key].replace(
                patch.old_text, patch.new_text, 1
            )
    return result


def validate_change_synthesis(
    change: DesignChange,
    original_sources: dict[str, str],
    top_module: str | None = None,
) -> bool:
    """Apply a change and check if Yosys can still synthesize the design.

    Returns True if synthesis succeeds, False otherwise.
    Sets change.synthesis_valid.
    """
    import subprocess
    import tempfile

    try:
        modified = apply_changes(change, original_sources)
    except ValueError as e:
        logger.warning("Could not apply change %s: %s", change.change_id, e)
        change.synthesis_valid = False
        return False

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write modified sources
        file_paths = []
        for fname, content in modified.items():
            p = Path(tmpdir) / Path(fname).name
            p.write_text(content)
            file_paths.append(str(p))

        # Run Yosys
        file_args = " ".join(f'"{f}"' for f in file_paths)
        script = f"read_verilog -sv {file_args}"
        if top_module:
            script += f"; hierarchy -top {top_module}"
        script += "; proc; opt"

        try:
            result = subprocess.run(
                ["yosys", "-p", script],
                capture_output=True, text=True, timeout=60,
                cwd=tmpdir,
            )
            change.synthesis_valid = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            change.synthesis_valid = False

    return change.synthesis_valid


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_changeset_summary(cs: ChangeSet):
    """Print a summary of generated changes."""
    print(f"\nChangeSet: {cs.design_name}")
    print(f"  Total changes: {len(cs.changes)}")

    by_type = cs.by_type()
    for ct, changes in sorted(by_type.items(), key=lambda x: x[0].value):
        gt_sizes = [len(c.ground_truth) for c in changes]
        avg_gt = sum(gt_sizes) / len(gt_sizes) if gt_sizes else 0
        print(f"  {ct.value:15s}: {len(changes):3d} changes  "
              f"(avg ground truth: {avg_gt:.1f} artifacts)")

    # Synthesis validity
    valid = sum(1 for c in cs.changes if c.synthesis_valid is True)
    invalid = sum(1 for c in cs.changes if c.synthesis_valid is False)
    unknown = sum(1 for c in cs.changes if c.synthesis_valid is None)
    if valid + invalid > 0:
        print(f"  Synthesis: {valid} valid, {invalid} invalid, {unknown} unknown")

    # Ground truth coverage by type
    all_gt_types: dict[str, int] = {}
    for c in cs.changes:
        for atype, nodes in c.ground_truth_by_type.items():
            all_gt_types[atype] = all_gt_types.get(atype, 0) + len(nodes)
    if all_gt_types:
        print("  Ground truth artifact coverage:")
        for atype, count in sorted(all_gt_types.items()):
            print(f"    {atype}: {count}")
