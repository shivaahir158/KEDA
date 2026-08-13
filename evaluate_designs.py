#!/usr/bin/env python3
"""
KEDA-Bench: Evaluate the full pipeline on real open-source designs.

Designs:
  1. PicoRV32   — RISC-V CPU (8 modules, 3049 LoC, single file)
  2. Verilog-UART — UART IP core (3 modules, multi-file)
  3. AXI-DMA    — AXI DMA engine (4 modules, complex parameterization)
  4. UART (fixture) — Our test UART (4 modules, full cross-artifact)

Pipeline per design:
  Step 1: Build knowledge graph (Yosys + SDC + SVA + Git)
  Step 2: Generate controlled changes (7 types)
  Step 3: Compute ground truth via graph traversal
  Step 4: Run impact analysis (4 methods)
  Step 5: Evaluate against ground truth (P/R/F1, Recall@K, MRR)
"""

import sys
import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from keda.graph.builder import KGBuilder, DesignConfig, print_build_summary
from keda.analysis.changegen import (
    ChangeType, ChangeGenerator, GroundTruthComputer, ChangeSet,
    print_changeset_summary,
)
from keda.analysis.impact import ChangeImpactAnalyzer, compute_metrics

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DESIGNS_DIR = Path(__file__).parent / "designs"
FIXTURE_DIR = Path(__file__).parent / "tests" / "fixtures"
RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Design configurations
# ---------------------------------------------------------------------------

DESIGN_CONFIGS = {
    "uart_fixture": {
        "name": "uart_fixture",
        "rtl_files": [
            FIXTURE_DIR / "uart_top.v",
            FIXTURE_DIR / "uart_tx.v",
            FIXTURE_DIR / "uart_rx.v",
            FIXTURE_DIR / "baud_gen.v",
        ],
        "top_module": "uart_top",
        "sdc_files": [FIXTURE_DIR / "uart.sdc"],
        "sva_files": [FIXTURE_DIR / "uart_assertions.sv"],
        "repo_path": FIXTURE_DIR / "git_repo",
    },
    "picorv32": {
        "name": "picorv32",
        "rtl_files": [DESIGNS_DIR / "picorv32" / "picorv32.v"],
        "top_module": "",  # empty = skip hierarchy, keep all 8 modules
        "sdc_files": [DESIGNS_DIR / "picorv32" / "picorv32.sdc"],
        "sva_files": [],
        "repo_path": DESIGNS_DIR / "picorv32",
    },
    "verilog_uart": {
        "name": "verilog_uart",
        "rtl_files": [
            DESIGNS_DIR / "verilog-uart" / "rtl" / "uart.v",
            DESIGNS_DIR / "verilog-uart" / "rtl" / "uart_rx.v",
            DESIGNS_DIR / "verilog-uart" / "rtl" / "uart_tx.v",
        ],
        "top_module": "uart",
        "sdc_files": [DESIGNS_DIR / "verilog-uart" / "rtl" / "uart.sdc"],
        "sva_files": [],
        "repo_path": DESIGNS_DIR / "verilog-uart",
    },
    "axi_dma": {
        "name": "axi_dma",
        "rtl_files": [
            DESIGNS_DIR / "verilog-axi" / "rtl" / "axi_dma.v",
            DESIGNS_DIR / "verilog-axi" / "rtl" / "axi_dma_rd.v",
            DESIGNS_DIR / "verilog-axi" / "rtl" / "axi_dma_wr.v",
            DESIGNS_DIR / "verilog-axi" / "rtl" / "axi_dma_desc_mux.v",
        ],
        "top_module": "axi_dma",
        "sdc_files": [DESIGNS_DIR / "verilog-axi" / "rtl" / "axi_dma.sdc"],
        "sva_files": [],
        "repo_path": DESIGNS_DIR / "verilog-axi",
    },
    "mor1kx": {
        "name": "mor1kx",
        "rtl_glob": str(DESIGNS_DIR / "mor1kx" / "rtl" / "verilog" / "**" / "*.v"),
        "top_module": "mor1kx",
        "sdc_files": [DESIGNS_DIR / "mor1kx" / "mor1kx.sdc"],
        "sva_files": [],
        "repo_path": DESIGNS_DIR / "mor1kx",
        "include_dirs": [DESIGNS_DIR / "mor1kx" / "rtl" / "verilog"],
        "git_max_commits": 50,
    },
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@dataclass
class DesignResult:
    name: str
    num_nodes: int = 0
    num_edges: int = 0
    num_modules: int = 0
    num_changes: int = 0
    changes_by_type: dict[str, int] = field(default_factory=dict)
    build_time: float = 0.0
    gen_time: float = 0.0
    eval_time: float = 0.0
    per_type_metrics: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    overall_metrics: dict[str, dict[str, float]] = field(default_factory=dict)


def evaluate_design(design_key: str) -> DesignResult | None:
    """Run the full KEDA pipeline on one design."""
    import glob as glob_mod
    cfg = DESIGN_CONFIGS[design_key]
    result = DesignResult(name=cfg["name"])

    # Resolve RTL files (support glob patterns)
    if "rtl_glob" in cfg:
        rtl_files = sorted(glob_mod.glob(cfg["rtl_glob"], recursive=True))
        # Exclude defines/header files that aren't modules
        rtl_files = [f for f in rtl_files if Path(f).suffix == ".v"]
    else:
        rtl_files = [str(f) for f in cfg["rtl_files"]]

    # Check files exist
    if not rtl_files:
        print(f"  SKIP: No RTL files found")
        return None
    for f in rtl_files:
        if not Path(f).exists():
            print(f"  SKIP: RTL file not found: {f}")
            return None

    # Step 1: Build KG
    print(f"\n  Step 1: Building knowledge graph ({len(rtl_files)} RTL files)...")
    t0 = time.time()

    config = DesignConfig(
        name=cfg["name"],
        rtl_files=rtl_files,
        top_module=cfg["top_module"],
        sdc_files=[f for f in cfg.get("sdc_files", []) if Path(f).exists()],
        sva_files=[f for f in cfg.get("sva_files", []) if Path(f).exists()],
        repo_path=cfg.get("repo_path") if cfg.get("repo_path") and Path(cfg["repo_path"]).exists() else None,
        include_dirs=cfg.get("include_dirs", []),
        git_max_commits=cfg.get("git_max_commits", 50),
    )
    try:
        builder = KGBuilder()
        build = builder.build(config)
    except Exception as e:
        print(f"  FAILED to build KG: {e}")
        return None

    G = build.graph
    result.build_time = time.time() - t0
    result.num_nodes = G.number_of_nodes()
    result.num_edges = G.number_of_edges()

    # Count modules
    result.num_modules = sum(
        1 for _, d in G.nodes(data=True) if d.get("type") == "Module"
    )
    node_types = {}
    for _, d in G.nodes(data=True):
        t = d.get("type", "Unknown")
        node_types[t] = node_types.get(t, 0) + 1

    print(f"    Graph: {result.num_nodes} nodes, {result.num_edges} edges, "
          f"{result.num_modules} modules")
    print(f"    Node types: {dict(sorted(node_types.items()))}")
    print(f"    Build time: {result.build_time:.2f}s")

    # Step 2: Generate changes
    print(f"  Step 2: Generating controlled changes...")
    t0 = time.time()

    rtl_sources = {}
    for f in rtl_files:
        rtl_sources[str(f)] = Path(f).read_text()

    sdc_sources = {}
    for f in cfg.get("sdc_files", []):
        if Path(f).exists():
            sdc_sources[str(f)] = Path(f).read_text()

    gen = ChangeGenerator(G, rtl_sources, cfg["name"], sdc_sources)
    # Limit changes per type for large designs to keep evaluation tractable
    max_per = cfg.get("max_per_type", 10)
    changeset = gen.generate_all(max_per_type=max_per)

    result.gen_time = time.time() - t0
    result.num_changes = len(changeset.changes)

    by_type = changeset.by_type()
    result.changes_by_type = {ct.value: len(cs) for ct, cs in by_type.items()}
    print(f"    Generated {result.num_changes} changes in {result.gen_time:.2f}s")
    for ct, count in sorted(result.changes_by_type.items()):
        print(f"      {ct:15s}: {count}")

    if result.num_changes == 0:
        print("  No changes generated — skipping evaluation.")
        return result

    # Step 3: Compute ground truth
    print(f"  Step 3: Computing ground truth...")
    gtc = GroundTruthComputer(G)
    gtc.compute_all(changeset)

    gt_sizes = [len(c.ground_truth) for c in changeset.changes]
    avg_gt = sum(gt_sizes) / len(gt_sizes) if gt_sizes else 0
    print(f"    Avg ground truth size: {avg_gt:.1f} artifacts")

    # Step 4 & 5: Run impact analysis and evaluate
    print(f"  Step 4-5: Running impact analysis & evaluation...")
    t0 = time.time()

    analyzer = ChangeImpactAnalyzer(G)
    methods = ["weighted_bfs", "bfs", "pagerank", "structural"]

    # Per-change-type evaluation
    for ctype, changes in sorted(by_type.items(), key=lambda x: x[0].value):
        type_key = ctype.value
        result.per_type_metrics[type_key] = {}

        for method in methods:
            all_metrics = []
            for c in changes:
                if not c.ground_truth:
                    continue
                module_id = f"{cfg['name']}::{c.target_module}"
                if module_id not in G:
                    continue
                try:
                    if method == "weighted_bfs":
                        impact = analyzer.weighted_bfs(
                            [module_id], change_type=c.change_type.value
                        )
                    elif method == "bfs":
                        impact = analyzer.bfs([module_id])
                    elif method == "pagerank":
                        impact = analyzer.pagerank([module_id])
                    elif method == "structural":
                        impact = analyzer.structural([module_id])
                    else:
                        continue

                    m = compute_metrics(impact, c.ground_truth)
                    all_metrics.append(m)
                except Exception:
                    pass

            if all_metrics:
                avg = {}
                for key in ["precision", "recall", "f1", "mrr"]:
                    vals = [m[key] for m in all_metrics if key in m]
                    avg[key] = sum(vals) / len(vals) if vals else 0.0
                for key in ["recall@5", "recall@10", "recall@20"]:
                    vals = [m[key] for m in all_metrics if key in m]
                    avg[key] = sum(vals) / len(vals) if vals else 0.0
                avg["n"] = len(all_metrics)
                result.per_type_metrics[type_key][method] = avg

    # Overall evaluation
    for method in methods:
        all_metrics = []
        for c in changeset.changes:
            if not c.ground_truth:
                continue
            module_id = f"{cfg['name']}::{c.target_module}"
            if module_id not in G:
                continue
            try:
                if method == "weighted_bfs":
                    impact = analyzer.weighted_bfs(
                        [module_id], change_type=c.change_type.value
                    )
                elif method == "bfs":
                    impact = analyzer.bfs([module_id])
                elif method == "pagerank":
                    impact = analyzer.pagerank([module_id])
                elif method == "structural":
                    impact = analyzer.structural([module_id])
                else:
                    continue
                m = compute_metrics(impact, c.ground_truth)
                all_metrics.append(m)
            except Exception:
                pass

        if all_metrics:
            avg = {}
            for key in ["precision", "recall", "f1", "mrr"]:
                vals = [m[key] for m in all_metrics if key in m]
                avg[key] = sum(vals) / len(vals) if vals else 0.0
            for key in ["recall@5", "recall@10", "recall@20"]:
                vals = [m[key] for m in all_metrics if key in m]
                avg[key] = sum(vals) / len(vals) if vals else 0.0
            avg["n"] = len(all_metrics)
            result.overall_metrics[method] = avg

    result.eval_time = time.time() - t0
    print(f"    Evaluation time: {result.eval_time:.2f}s")

    # Save changeset
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    changeset.save(RESULTS_DIR / f"{cfg['name']}_changeset.json")

    return result


def print_results_table(results: list[DesignResult]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 100)
    print("  KEDA-Bench: Cross-Design Evaluation Results")
    print("=" * 100)

    # Design summary table
    print(f"\n{'Design':<18} {'Modules':>8} {'Nodes':>8} {'Edges':>8} {'Changes':>8} {'Build(s)':>8}")
    print("-" * 68)
    for r in results:
        print(f"{r.name:<18} {r.num_modules:>8} {r.num_nodes:>8} {r.num_edges:>8} "
              f"{r.num_changes:>8} {r.build_time:>8.2f}")

    # Overall metrics per method
    print(f"\n{'='*100}")
    print(f"  Overall Metrics (averaged across all changes)")
    print(f"{'='*100}")

    methods = ["weighted_bfs", "bfs", "pagerank", "structural"]
    print(f"\n{'Design':<18} {'Method':<18} {'P':>7} {'R':>7} {'F1':>7} "
          f"{'R@5':>7} {'R@10':>7} {'R@20':>7} {'MRR':>7} {'n':>5}")
    print("-" * 100)

    for r in results:
        if not r.overall_metrics:
            continue
        for i, method in enumerate(methods):
            if method not in r.overall_metrics:
                continue
            m = r.overall_metrics[method]
            name_col = r.name if i == 0 else ""
            print(f"{name_col:<18} {method:<18} "
                  f"{m.get('precision',0):>7.3f} {m.get('recall',0):>7.3f} "
                  f"{m.get('f1',0):>7.3f} "
                  f"{m.get('recall@5',0):>7.3f} {m.get('recall@10',0):>7.3f} "
                  f"{m.get('recall@20',0):>7.3f} "
                  f"{m.get('mrr',0):>7.3f} {int(m.get('n',0)):>5}")
        print()

    # Per-change-type breakdown for each design
    print(f"\n{'='*100}")
    print(f"  Per-Change-Type F1 Scores (weighted_bfs vs structural baseline)")
    print(f"{'='*100}")

    all_types = sorted(set(
        ct for r in results for ct in r.per_type_metrics.keys()
    ))

    header = f"{'Design':<18}"
    for ct in all_types:
        header += f" {ct[:8]:>9}"
    print(f"\n{header}")
    print("-" * (18 + 10 * len(all_types)))

    for r in results:
        if not r.per_type_metrics:
            continue
        # weighted_bfs row
        row = f"{r.name:<18}"
        for ct in all_types:
            m = r.per_type_metrics.get(ct, {}).get("weighted_bfs", {})
            f1 = m.get("f1", -1)
            row += f" {f1:>9.3f}" if f1 >= 0 else f" {'—':>9}"
        print(f"{row}  (weighted_bfs)")

        # structural row
        row = f"{'':18}"
        for ct in all_types:
            m = r.per_type_metrics.get(ct, {}).get("structural", {})
            f1 = m.get("f1", -1)
            row += f" {f1:>9.3f}" if f1 >= 0 else f" {'—':>9}"
        print(f"{row}  (structural)")
        print()

    # Key findings
    print(f"\n{'='*100}")
    print(f"  Key Findings")
    print(f"{'='*100}")

    for r in results:
        if not r.overall_metrics:
            continue
        wbfs = r.overall_metrics.get("weighted_bfs", {})
        struct = r.overall_metrics.get("structural", {})
        if wbfs and struct:
            recall_gain = wbfs.get("recall", 0) - struct.get("recall", 0)
            f1_gain = wbfs.get("f1", 0) - struct.get("f1", 0)
            print(f"  {r.name}: weighted_bfs recall={wbfs.get('recall',0):.3f} vs "
                  f"structural recall={struct.get('recall',0):.3f} "
                  f"(+{recall_gain:.3f}), "
                  f"F1 gain: +{f1_gain:.3f}")


def save_results_json(results: list[DesignResult]):
    """Save results as JSON for further analysis."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for r in results:
        out.append({
            "name": r.name,
            "num_modules": r.num_modules,
            "num_nodes": r.num_nodes,
            "num_edges": r.num_edges,
            "num_changes": r.num_changes,
            "changes_by_type": r.changes_by_type,
            "build_time": r.build_time,
            "gen_time": r.gen_time,
            "eval_time": r.eval_time,
            "per_type_metrics": r.per_type_metrics,
            "overall_metrics": r.overall_metrics,
        })
    with open(RESULTS_DIR / "keda_bench_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / 'keda_bench_results.json'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  KEDA-Bench: Knowledge-Graph-Enhanced EDA Evaluation")
    print("=" * 70)

    results = []
    for design_key in DESIGN_CONFIGS:
        print(f"\n{'='*70}")
        print(f"  Processing: {design_key}")
        print(f"{'='*70}")

        r = evaluate_design(design_key)
        if r is not None:
            results.append(r)

    if results:
        print_results_table(results)
        save_results_json(results)

    # Total summary
    total_changes = sum(r.num_changes for r in results)
    total_modules = sum(r.num_modules for r in results)
    print(f"\n  Total: {len(results)} designs, {total_modules} modules, "
          f"{total_changes} controlled changes")


if __name__ == "__main__":
    main()
