#!/usr/bin/env python3 -u
"""
KEDA QA Evaluation — End-to-end evaluation of the GraphRAG QA engine.

Builds knowledge graphs for real designs, generates engineering questions,
runs all 5 QA methods, and evaluates with metrics.

Usage:
    python evaluate_qa.py                    # Run with all designs, no LLM
    python evaluate_qa.py --with-llm         # Run with Anthropic Claude API
    python evaluate_qa.py --design picorv32  # Run on a single design
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import networkx as nx

from keda.graph.builder import KGBuilder, DesignConfig, print_build_summary
from keda.llm.qa_engine import QAEngine
from keda.llm.question_gen import QuestionGenerator, save_questions, EngineeringQuestion
from keda.llm.evaluator import QAEvaluator, print_qa_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ── Design configurations ──────────────────────────────────────────────

DESIGNS = {
    "uart_fixture": {
        "rtl_glob": "tests/fixtures/*.v",
        "sdc_files": [],
        "top_module": None,
        "repo_path": None,
    },
    "picorv32": {
        "rtl_glob": "designs/picorv32/picorv32.v",
        "sdc_files": ["designs/picorv32/picorv32.sdc"],
        "top_module": "",
        "repo_path": "designs/picorv32",
        "git_max_commits": 100,
    },
    "verilog_uart": {
        "rtl_glob": "designs/verilog-uart/rtl/*.v",
        "sdc_files": ["designs/verilog-uart/rtl/uart.sdc"],
        "top_module": "",
        "repo_path": "designs/verilog-uart",
        "git_max_commits": 100,
    },
}


def resolve_design(name: str, cfg: dict) -> DesignConfig | None:
    """Resolve a design config, expanding globs."""
    import glob

    rtl_files = sorted(glob.glob(cfg["rtl_glob"]))
    if not rtl_files:
        logger.warning("No RTL files for %s: %s", name, cfg["rtl_glob"])
        return None

    sdc_files = [f for f in cfg.get("sdc_files", []) if Path(f).exists()]

    return DesignConfig(
        name=name,
        rtl_files=rtl_files,
        top_module=cfg.get("top_module"),
        sdc_files=sdc_files,
        repo_path=cfg.get("repo_path"),
        git_max_commits=cfg.get("git_max_commits", 200),
        include_dirs=cfg.get("include_dirs", []),
    )


def run_evaluation(
    designs: dict[str, dict],
    with_llm: bool = False,
    max_questions_per_cat: int = 10,
):
    """Run the full QA evaluation pipeline."""
    builder = KGBuilder()
    evaluator = QAEvaluator()

    llm_client = None
    llm_model = "gpt-4o-mini"
    llm_provider = "openai"
    if with_llm:
        try:
            import openai
            api_key = os.environ.get("OPENAI_API_KEY", "")
            llm_client = openai.OpenAI(api_key=api_key)
            logger.info("LLM client initialized (OpenAI %s)", llm_model)
        except Exception as e:
            logger.warning("Could not initialize LLM client: %s", e)

    all_results: dict[str, dict] = {}
    total_questions = 0

    for design_name, cfg in designs.items():
        print(f"\n{'='*70}")
        print(f"  Design: {design_name}")
        print(f"{'='*70}")

        config = resolve_design(design_name, cfg)
        if config is None:
            continue

        # Build KG
        t0 = time.time()
        try:
            build_result = builder.build(config)
        except Exception as e:
            logger.error("Failed to build KG for %s: %s", design_name, e)
            continue
        build_time = time.time() - t0
        print(f"  KG built in {build_time:.1f}s: {build_result.graph.number_of_nodes()} nodes, "
              f"{build_result.graph.number_of_edges()} edges")

        graph = build_result.graph

        # Generate questions
        gen = QuestionGenerator(graph, design_name)
        questions = gen.generate_all(max_per_category=max_questions_per_cat)
        print(f"  Generated {len(questions)} questions")

        # Count by category
        cat_counts: dict[str, int] = {}
        for q in questions:
            cat_counts[q.category] = cat_counts.get(q.category, 0) + 1
        for cat, cnt in sorted(cat_counts.items()):
            print(f"    {cat}: {cnt}")

        # Save questions
        os.makedirs("results", exist_ok=True)
        save_questions(questions, f"results/{design_name}_questions.json")

        # Initialize QA engine
        qa = QAEngine(
            graph=graph,
            design_name=design_name,
            llm_client=llm_client,
            llm_model=llm_model,
            llm_provider=llm_provider,
        )

        # Determine which methods to run
        methods = ["kg_only", "keda_qa", "vector_rag", "keda_full"]
        if with_llm:
            methods = ["llm_only"] + methods

        # Run all methods
        method_results: dict[str, list] = {m: [] for m in methods}
        method_metrics: dict[str, list] = {m: [] for m in methods}

        total_calls = len(questions) * len(methods)
        call_num = 0
        for i, q in enumerate(questions):
            for method in methods:
                call_num += 1
                if call_num % 10 == 0 or call_num == 1:
                    print(f"    Progress: {call_num}/{total_calls} "
                          f"(Q{i+1}/{len(questions)}, {method})", flush=True)
                try:
                    result = qa.answer(q.question_text, method=method)
                    method_results[method].append(result)
                    m = evaluator.evaluate(q, result)
                    method_metrics[method].append(m)
                except Exception as e:
                    logger.warning("Q%d method %s failed: %s", i, method, e)
                    from keda.llm.qa_engine import QAResult
                    method_results[method].append(QAResult(
                        question=q.question_text, answer=f"ERROR: {e}", method=method
                    ))
                    from keda.llm.evaluator import QAMetrics
                    method_metrics[method].append(QAMetrics(
                        question_id=q.question_id, method=method
                    ))

        # Print evaluation
        print_qa_evaluation(method_metrics, questions)

        # Store results
        design_summary = {}
        for method, metrics_list in method_metrics.items():
            agg = evaluator.aggregate(metrics_list)
            design_summary[method] = agg
            by_cat = evaluator.aggregate_by_category(metrics_list, questions)
            design_summary[f"{method}_by_category"] = by_cat

        all_results[design_name] = {
            "num_questions": len(questions),
            "categories": cat_counts,
            "graph_nodes": graph.number_of_nodes(),
            "graph_edges": graph.number_of_edges(),
            "build_time_s": build_time,
            "metrics": design_summary,
        }
        total_questions += len(questions)

        # Print a few example Q&A pairs
        print(f"\n  --- Example Q&A (kg_only) ---")
        for q, r in list(zip(questions, method_results.get("kg_only", [])))[:3]:
            print(f"\n  Q: {q.question_text}")
            print(f"  Gold: {q.gold_answer[:120]}")
            answer_preview = r.answer[:200].replace('\n', ' ')
            print(f"  Pred: {answer_preview}")

        if "keda_qa" in method_results and method_results["keda_qa"]:
            print(f"\n  --- Example Q&A (keda_qa) ---")
            for q, r in list(zip(questions, method_results["keda_qa"]))[:2]:
                print(f"\n  Q: {q.question_text}")
                answer_preview = r.answer[:200].replace('\n', ' ')
                print(f"  A: {answer_preview}")

    # ── Summary across designs ──────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  OVERALL SUMMARY ({total_questions} questions across {len(all_results)} designs)")
    print(f"{'='*70}")

    # Aggregate across designs
    overall: dict[str, dict[str, list]] = {}
    for design_name, data in all_results.items():
        for method, agg in data["metrics"].items():
            if isinstance(agg, dict) and "evidence_recall" in agg:
                if method not in overall:
                    overall[method] = {k: [] for k in agg}
                for k, v in agg.items():
                    if isinstance(v, (int, float)):
                        overall[method][k].append(v)

    print(f"\n{'Method':<15} {'Ev.Recall':>10} {'Ev.Prec':>10} {'TokenF1':>10} "
          f"{'KeyEntity':>10} {'Latency':>10}")
    print("-" * 70)
    for method, values in sorted(overall.items()):
        def avg(key):
            v = values.get(key, [])
            return sum(v) / len(v) if v else 0
        print(f"{method:<15} {avg('evidence_recall'):>10.3f} "
              f"{avg('evidence_precision'):>10.3f} "
              f"{avg('token_overlap_f1'):>10.3f} "
              f"{avg('key_entity_recall'):>10.3f} "
              f"{avg('avg_latency_ms'):>8.1f}ms")

    # Save results
    with open("results/keda_qa_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to results/keda_qa_results.json")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="KEDA QA Evaluation")
    parser.add_argument("--with-llm", action="store_true",
                        help="Enable LLM synthesis (requires ANTHROPIC_API_KEY)")
    parser.add_argument("--design", type=str, default=None,
                        help="Run on a single design")
    parser.add_argument("--max-questions", type=int, default=10,
                        help="Max questions per category")
    args = parser.parse_args()

    designs = DESIGNS
    if args.design:
        if args.design not in DESIGNS:
            print(f"Unknown design: {args.design}. Available: {list(DESIGNS.keys())}")
            sys.exit(1)
        designs = {args.design: DESIGNS[args.design]}

    run_evaluation(designs, with_llm=args.with_llm,
                   max_questions_per_cat=args.max_questions)


if __name__ == "__main__":
    main()
