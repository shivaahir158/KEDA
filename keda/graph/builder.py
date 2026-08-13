"""
Unified EDA Knowledge Graph builder.

Orchestrates all extractors (Yosys, SDC, SVA, Git) to construct a single
cross-artifact knowledge graph for a hardware design.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

from keda.extractors.yosys_extractor import (
    YosysExtractor,
    YosysExtractionResult,
    print_extraction_summary,
    print_graph_summary,
)
from keda.extractors.sdc_extractor import (
    SDCExtractor,
    SDCExtractionResult,
    print_sdc_summary,
)
from keda.extractors.sva_extractor import (
    SVAExtractor,
    SVAExtractionResult,
    print_sva_summary,
)
from keda.extractors.git_extractor import (
    GitExtractor,
    GitExtractionResult,
    print_git_summary,
)

logger = logging.getLogger(__name__)


@dataclass
class DesignConfig:
    """Configuration for a single design to be processed."""
    name: str
    rtl_files: list[str | Path]
    top_module: str | None = None
    sdc_files: list[str | Path] = field(default_factory=list)
    sva_files: list[str | Path] = field(default_factory=list)
    repo_path: str | Path | None = None
    git_branch: str | None = None
    git_max_commits: int = 500
    include_dirs: list[str | Path] = field(default_factory=list)


@dataclass
class BuildResult:
    """Results from a full knowledge graph build."""
    graph: nx.DiGraph
    design_name: str
    rtl_result: YosysExtractionResult | None = None
    sdc_result: SDCExtractionResult | None = None
    sva_result: SVAExtractionResult | None = None
    git_result: GitExtractionResult | None = None


class KGBuilder:
    """Build a unified EDA knowledge graph from multiple artifact sources.

    Usage:
        config = DesignConfig(
            name="uart",
            rtl_files=["uart_top.v", "uart_tx.v", ...],
            top_module="uart_top",
            sdc_files=["uart.sdc"],
            sva_files=["uart_assertions.sv"],
            repo_path="/path/to/repo",
        )
        builder = KGBuilder()
        result = builder.build(config)
        print_build_summary(result)
    """

    def __init__(self, yosys_binary: str = "yosys"):
        self.yosys_ext = YosysExtractor(yosys_binary=yosys_binary)
        self.sdc_ext = SDCExtractor()
        self.sva_ext = SVAExtractor()
        self.git_ext = GitExtractor()

    def build(self, config: DesignConfig) -> BuildResult:
        """Build the full cross-artifact knowledge graph."""
        result = BuildResult(
            graph=nx.DiGraph(),
            design_name=config.name,
        )

        # Step 1: RTL extraction (foundation)
        logger.info("Step 1: Extracting RTL structure with Yosys...")
        rtl_result = self.yosys_ext.extract(
            verilog_files=config.rtl_files,
            top_module=config.top_module,
            design_name=config.name,
            include_dirs=config.include_dirs or None,
        )
        result.rtl_result = rtl_result
        result.graph = self.yosys_ext.build_graph(rtl_result)
        logger.info(
            "  RTL: %d modules, %d instances, %d registers, %d ports",
            len(rtl_result.modules), len(rtl_result.instances),
            len(rtl_result.registers), len(rtl_result.ports),
        )

        # Collect known module names for downstream extractors
        known_modules = {
            m["name"] for m in rtl_result.modules.values()
        }

        # Step 2: SDC constraints
        if config.sdc_files:
            logger.info("Step 2: Extracting SDC constraints...")
            sdc_result = self.sdc_ext.extract(
                sdc_files=config.sdc_files,
                design_name=config.name,
            )
            result.sdc_result = sdc_result
            self.sdc_ext.add_to_graph(sdc_result, result.graph, config.name)
            logger.info("  SDC: %d constraints, %d clocks",
                        len(sdc_result.constraints), len(sdc_result.clocks))
        else:
            logger.info("Step 2: No SDC files provided, skipping.")

        # Step 3: SVA assertions and tests
        sva_files = list(config.sva_files)
        # Also scan RTL files for inline assertions
        for rtl_file in config.rtl_files:
            if str(rtl_file) not in [str(f) for f in sva_files]:
                sva_files.append(rtl_file)

        if sva_files:
            logger.info("Step 3: Extracting SVA assertions and tests...")
            sva_result = self.sva_ext.extract(
                files=sva_files,
                design_name=config.name,
                known_modules=known_modules,
            )
            result.sva_result = sva_result
            self.sva_ext.add_to_graph(sva_result, result.graph, config.name)
            logger.info("  SVA: %d assertions, %d tests",
                        len(sva_result.assertions), len(sva_result.tests))
        else:
            logger.info("Step 3: No SVA/test files provided, skipping.")

        # Step 4: Git history
        if config.repo_path:
            logger.info("Step 4: Extracting Git history...")
            module_file_map = self.git_ext.get_module_file_map(result.graph)
            git_result = self.git_ext.extract(
                repo_path=config.repo_path,
                design_name=config.name,
                max_commits=config.git_max_commits,
                branch=config.git_branch,
                module_file_map=module_file_map,
            )
            result.git_result = git_result
            self.git_ext.add_to_graph(git_result, result.graph, config.name)
            logger.info("  Git: %d commits, %d file histories",
                        git_result.total_commits, len(git_result.file_histories))
        else:
            logger.info("Step 4: No repository path provided, skipping.")

        # Step 5: Post-processing
        logger.info("Step 5: Post-processing graph...")
        self._post_process(result.graph, config.name)

        return result

    def _post_process(self, graph: nx.DiGraph, design_name: str):
        """Add computed attributes and derived edges after all extractors run."""
        # Compute in-degree and out-degree for each module
        for node_id, data in graph.nodes(data=True):
            if data.get("type") == "Module":
                # Count structural relationships
                inst_count = sum(
                    1 for _, _, d in graph.out_edges(node_id, data=True)
                    if d.get("relation") == "instantiates"
                )
                port_count = sum(
                    1 for _, _, d in graph.out_edges(node_id, data=True)
                    if d.get("relation") == "has_port"
                )
                reg_count = sum(
                    1 for _, _, d in graph.out_edges(node_id, data=True)
                    if d.get("relation") == "contains_register"
                )
                assertion_count = sum(
                    1 for _, _, d in graph.out_edges(node_id, data=True)
                    if d.get("relation") == "verified_by"
                )
                constraint_count = sum(
                    1 for _, v, d in graph.edges(data=True)
                    if d.get("relation") == "applies_to"
                    and graph.nodes.get(v, {}).get("module") == node_id
                )

                graph.nodes[node_id]["instantiation_count"] = inst_count
                graph.nodes[node_id]["port_count"] = port_count
                graph.nodes[node_id]["register_count"] = reg_count
                graph.nodes[node_id]["assertion_count"] = assertion_count


def print_build_summary(result: BuildResult):
    """Print a comprehensive summary of the built knowledge graph."""
    print(f"\n{'='*60}")
    print(f"  KEDA Knowledge Graph: {result.design_name}")
    print(f"{'='*60}")

    if result.rtl_result:
        print("\n--- RTL Extraction ---")
        print_extraction_summary(result.rtl_result)

    if result.sdc_result:
        print("\n--- SDC Constraints ---")
        print_sdc_summary(result.sdc_result)

    if result.sva_result:
        print("\n--- SVA Assertions & Tests ---")
        print_sva_summary(result.sva_result)

    if result.git_result:
        print("\n--- Git History ---")
        print_git_summary(result.git_result)

    print("\n--- Knowledge Graph ---")
    print_graph_summary(result.graph)
    print()


def export_graph(graph: nx.DiGraph, path: str | Path, format: str = "graphml"):
    """Export the knowledge graph to a file.

    Supported formats: graphml, gexf, json (node-link).
    """
    path = Path(path)

    # Clean graph for serialization: convert sets/lists in node attrs
    G = graph.copy()
    for node_id in G.nodes():
        for key, val in list(G.nodes[node_id].items()):
            if isinstance(val, (set, list)):
                G.nodes[node_id][key] = str(val)
            elif isinstance(val, dict):
                G.nodes[node_id][key] = json.dumps(val)
            elif val is None:
                G.nodes[node_id][key] = ""
    for u, v in G.edges():
        for key, val in list(G[u][v].items()):
            if isinstance(val, (set, list)):
                G[u][v][key] = str(val)
            elif isinstance(val, dict):
                G[u][v][key] = json.dumps(val)
            elif val is None:
                G[u][v][key] = ""

    if format == "graphml":
        nx.write_graphml(G, str(path))
    elif format == "gexf":
        nx.write_gexf(G, str(path))
    elif format == "json":
        data = nx.node_link_data(G)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info("Exported graph to %s (%s)", path, format)
