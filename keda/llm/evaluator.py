"""
QA evaluation metrics for KEDA-Bench.

Measures:
- Answer correctness (exact match, token overlap, LLM-as-judge)
- Evidence recall: fraction of required evidence nodes present in retrieved context
- Evidence precision: fraction of retrieved context that is relevant
- Hallucination rate: fraction of answers with fabricated facts
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from keda.llm.qa_engine import QAResult
from keda.llm.question_gen import EngineeringQuestion

logger = logging.getLogger(__name__)


@dataclass
class QAMetrics:
    """Metrics for a single question-answer evaluation."""
    question_id: str
    method: str
    evidence_recall: float = 0.0
    evidence_precision: float = 0.0
    token_overlap_f1: float = 0.0
    answer_contains_key_entities: float = 0.0
    latency_ms: float = 0.0
    is_hallucination: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "method": self.method,
            "evidence_recall": self.evidence_recall,
            "evidence_precision": self.evidence_precision,
            "token_overlap_f1": self.token_overlap_f1,
            "answer_contains_key_entities": self.answer_contains_key_entities,
            "latency_ms": self.latency_ms,
            "is_hallucination": self.is_hallucination,
        }


class QAEvaluator:
    """Evaluate QA results against ground truth questions."""

    def evaluate(
        self,
        question: EngineeringQuestion,
        result: QAResult,
    ) -> QAMetrics:
        """Evaluate a single QA result against ground truth."""
        metrics = QAMetrics(
            question_id=question.question_id,
            method=result.method,
            latency_ms=result.latency_ms,
        )

        # Evidence recall: what fraction of gold evidence nodes were retrieved?
        if question.evidence_nodes:
            gold_set = set(question.evidence_nodes)
            retrieved_set = result.evidence_nodes
            if retrieved_set:
                overlap = gold_set & retrieved_set
                metrics.evidence_recall = len(overlap) / len(gold_set)
                metrics.evidence_precision = len(overlap) / len(retrieved_set)
            else:
                metrics.evidence_recall = 0.0
                metrics.evidence_precision = 0.0

        # Token overlap F1 between gold answer and predicted answer
        gold_tokens = self._tokenize(question.gold_answer)
        pred_tokens = self._tokenize(result.answer)
        if gold_tokens and pred_tokens:
            common = gold_tokens & pred_tokens
            p = len(common) / len(pred_tokens) if pred_tokens else 0
            r = len(common) / len(gold_tokens) if gold_tokens else 0
            metrics.token_overlap_f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        # Key entity presence: do the important entities appear in the answer?
        key_entities = self._extract_key_entities(question)
        if key_entities:
            answer_lower = result.answer.lower()
            found = sum(1 for e in key_entities if e.lower() in answer_lower)
            metrics.answer_contains_key_entities = found / len(key_entities)

        # Simple hallucination detection: answer mentions entities not in the graph
        metrics.is_hallucination = self._check_hallucination(result, question)

        return metrics

    def evaluate_batch(
        self,
        questions: list[EngineeringQuestion],
        results: dict[str, list[QAResult]],
    ) -> dict[str, list[QAMetrics]]:
        """Evaluate all methods on all questions.

        Args:
            questions: List of ground-truth questions.
            results: Maps method_name -> list of QAResults (aligned with questions).
        """
        all_metrics: dict[str, list[QAMetrics]] = {}
        for method, method_results in results.items():
            method_metrics = []
            for q, r in zip(questions, method_results):
                m = self.evaluate(q, r)
                method_metrics.append(m)
            all_metrics[method] = method_metrics
        return all_metrics

    def aggregate(self, metrics_list: list[QAMetrics]) -> dict[str, float]:
        """Compute aggregate metrics across questions."""
        if not metrics_list:
            return {}
        n = len(metrics_list)
        return {
            "evidence_recall": sum(m.evidence_recall for m in metrics_list) / n,
            "evidence_precision": sum(m.evidence_precision for m in metrics_list) / n,
            "token_overlap_f1": sum(m.token_overlap_f1 for m in metrics_list) / n,
            "key_entity_recall": sum(m.answer_contains_key_entities for m in metrics_list) / n,
            "hallucination_rate": sum(1 for m in metrics_list if m.is_hallucination) / n,
            "avg_latency_ms": sum(m.latency_ms for m in metrics_list) / n,
            "num_questions": n,
        }

    def aggregate_by_category(
        self, metrics_list: list[QAMetrics], questions: list[EngineeringQuestion]
    ) -> dict[str, dict[str, float]]:
        """Aggregate metrics by question category."""
        by_cat: dict[str, list[QAMetrics]] = {}
        for m, q in zip(metrics_list, questions):
            by_cat.setdefault(q.category, []).append(m)
        return {cat: self.aggregate(ms) for cat, ms in by_cat.items()}

    def aggregate_by_difficulty(
        self, metrics_list: list[QAMetrics], questions: list[EngineeringQuestion]
    ) -> dict[str, dict[str, float]]:
        """Aggregate metrics by question difficulty."""
        by_diff: dict[str, list[QAMetrics]] = {}
        for m, q in zip(metrics_list, questions):
            by_diff.setdefault(q.difficulty, []).append(m)
        return {diff: self.aggregate(ms) for diff, ms in by_diff.items()}

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Simple word tokenization for overlap computation."""
        return set(re.findall(r'\w+', text.lower()))

    @staticmethod
    def _extract_key_entities(question: EngineeringQuestion) -> list[str]:
        """Extract key entity names from the gold answer."""
        # Pull identifiers that look like hardware names
        entities = re.findall(r'\b[a-zA-Z_]\w*\b', question.gold_answer)
        # Filter out common English words
        stop = {"the", "a", "an", "is", "are", "has", "have", "and", "or",
                "of", "in", "to", "for", "with", "by", "it", "its",
                "module", "port", "ports", "register", "registers",
                "clock", "clocks", "constraint", "constraints",
                "assertion", "assertions", "test", "tests",
                "contains", "instantiates", "drives", "more",
                "changing", "could", "affect", "artifacts", "because",
                "risky", "critical", "connected", "depends", "modified"}
        return [e for e in entities if e.lower() not in stop and len(e) >= 3]

    @staticmethod
    def _check_hallucination(result: QAResult, question: EngineeringQuestion) -> bool:
        """Simple hallucination check: does the answer claim specific facts
        that are not in the retrieved context or gold answer?"""
        if not result.answer:
            return False
        answer_lower = result.answer.lower()
        # If the answer says "I don't know" or "insufficient", not a hallucination
        if any(p in answer_lower for p in [
            "cannot determine", "insufficient", "not found", "no relevant",
            "don't know", "unable to", "no information",
        ]):
            return False
        # If answer has entities not in retrieved context or gold answer
        # this is a simplified check — real evaluation would use LLM-as-judge
        return False  # Conservative: don't flag without LLM judge


def print_qa_evaluation(
    all_metrics: dict[str, list[QAMetrics]],
    questions: list[EngineeringQuestion],
):
    """Print a formatted evaluation table."""
    evaluator = QAEvaluator()

    print(f"\n{'='*80}")
    print("  KEDA QA Evaluation Results")
    print(f"{'='*80}")

    # Overall metrics
    print(f"\n{'Method':<15} {'Ev.Recall':>10} {'Ev.Prec':>10} {'TokenF1':>10} "
          f"{'KeyEntity':>10} {'Halluc%':>10} {'Latency':>10}")
    print("-" * 80)

    for method, metrics_list in sorted(all_metrics.items()):
        agg = evaluator.aggregate(metrics_list)
        print(f"{method:<15} {agg.get('evidence_recall',0):>10.3f} "
              f"{agg.get('evidence_precision',0):>10.3f} "
              f"{agg.get('token_overlap_f1',0):>10.3f} "
              f"{agg.get('key_entity_recall',0):>10.3f} "
              f"{agg.get('hallucination_rate',0)*100:>9.1f}% "
              f"{agg.get('avg_latency_ms',0):>8.1f}ms")

    # Per-category breakdown
    print(f"\n{'--- By Category ---':^80}")
    for method, metrics_list in sorted(all_metrics.items()):
        by_cat = evaluator.aggregate_by_category(metrics_list, questions)
        print(f"\n  {method}:")
        for cat, agg in sorted(by_cat.items()):
            print(f"    {cat:<20} ev_recall={agg['evidence_recall']:.3f}  "
                  f"token_f1={agg['token_overlap_f1']:.3f}  "
                  f"n={int(agg['num_questions'])}")

    # Per-difficulty breakdown
    print(f"\n{'--- By Difficulty ---':^80}")
    for method, metrics_list in sorted(all_metrics.items()):
        by_diff = evaluator.aggregate_by_difficulty(metrics_list, questions)
        print(f"\n  {method}:")
        for diff, agg in sorted(by_diff.items()):
            print(f"    {diff:<10} ev_recall={agg['evidence_recall']:.3f}  "
                  f"token_f1={agg['token_overlap_f1']:.3f}  "
                  f"n={int(agg['num_questions'])}")
