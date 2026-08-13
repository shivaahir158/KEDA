"""
Engineering question generator for KEDA-Bench evaluation.

Generates questions across categories:
- Structural (1-hop): ports, submodules, registers, parameters
- Cross-artifact (2-3 hop): constraints, clocks, assertions, tests
- Temporal (Git): commit history, recent changes
- What-if (impact): change impact, risk assessment

Each question has a gold answer and evidence nodes derived from the KG.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class EngineeringQuestion:
    """A single benchmark question with ground truth."""
    question_id: str
    question_text: str
    category: str          # structural, cross_artifact, temporal, what_if
    difficulty: str        # easy, medium, hard
    hops_required: int
    gold_answer: str
    evidence_nodes: list[str] = field(default_factory=list)
    evidence_edges: list[tuple[str, str, str]] = field(default_factory=list)
    design_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question_text": self.question_text,
            "category": self.category,
            "difficulty": self.difficulty,
            "hops_required": self.hops_required,
            "gold_answer": self.gold_answer,
            "evidence_nodes": self.evidence_nodes,
            "evidence_edges": [(s, r, t) for s, r, t in self.evidence_edges],
            "design_name": self.design_name,
        }


class QuestionGenerator:
    """Generate engineering questions from a knowledge graph."""

    def __init__(self, graph: nx.DiGraph, design_name: str = "design"):
        self.graph = graph
        self.design_name = design_name
        self._modules = self._get_nodes_by_type("Module")
        self._ports = self._get_nodes_by_type("Port")
        self._registers = self._get_nodes_by_type("Register")
        self._clocks = self._get_nodes_by_type("Clock")
        self._constraints = self._get_nodes_by_type("Constraint")
        self._assertions = self._get_nodes_by_type("Assertion")
        self._tests = self._get_nodes_by_type("Test")
        self._commits = self._get_nodes_by_type("Commit")
        self._parameters = self._get_nodes_by_type("Parameter")

    def _get_nodes_by_type(self, node_type: str) -> list[str]:
        return [
            nid for nid, d in self.graph.nodes(data=True)
            if d.get("type") == node_type
        ]

    def generate_all(
        self, max_per_category: int = 20, seed: int = 42
    ) -> list[EngineeringQuestion]:
        """Generate questions across all categories."""
        rng = random.Random(seed)
        questions: list[EngineeringQuestion] = []
        qid = 0

        # Structural questions (easy, 1-hop)
        for q in self._gen_structural(rng, max_per_category):
            qid += 1
            q.question_id = f"{self.design_name}::q::{qid:04d}"
            questions.append(q)

        # Cross-artifact questions (medium, 2-3 hops)
        for q in self._gen_cross_artifact(rng, max_per_category):
            qid += 1
            q.question_id = f"{self.design_name}::q::{qid:04d}"
            questions.append(q)

        # Temporal questions (medium, Git-based)
        for q in self._gen_temporal(rng, max_per_category):
            qid += 1
            q.question_id = f"{self.design_name}::q::{qid:04d}"
            questions.append(q)

        # What-if questions (hard, multi-hop)
        for q in self._gen_what_if(rng, max_per_category):
            qid += 1
            q.question_id = f"{self.design_name}::q::{qid:04d}"
            questions.append(q)

        return questions

    # ------------------------------------------------------------------
    # Structural (1-hop, easy)
    # ------------------------------------------------------------------

    def _gen_structural(self, rng: random.Random, max_n: int) -> list[EngineeringQuestion]:
        G = self.graph
        questions: list[EngineeringQuestion] = []
        modules = list(self._modules)
        rng.shuffle(modules)

        for mod_id in modules[:max_n]:
            mod_name = G.nodes[mod_id].get("name", mod_id.split("::")[-1])

            # Q: "What ports does module X have?"
            ports = [
                (v, G.nodes[v].get("name", v))
                for _, v, d in G.out_edges(mod_id, data=True)
                if d.get("relation") == "has_port"
            ]
            if ports:
                port_names = sorted(set(p[1] for p in ports if p[1]))
                answer = f"Module {mod_name} has {len(port_names)} ports: {', '.join(port_names[:15])}"
                if len(port_names) > 15:
                    answer += f"... and {len(port_names) - 15} more"
                questions.append(EngineeringQuestion(
                    question_id="",
                    question_text=f"What ports does module {mod_name} have?",
                    category="structural",
                    difficulty="easy",
                    hops_required=1,
                    gold_answer=answer,
                    evidence_nodes=[mod_id] + [p[0] for p in ports],
                    evidence_edges=[(mod_id, "has_port", p[0]) for p in ports],
                    design_name=self.design_name,
                ))

            # Q: "What submodules does X instantiate?"
            children = [
                (v, G.nodes[v].get("name", v))
                for _, v, d in G.out_edges(mod_id, data=True)
                if d.get("relation") == "instantiates"
                and G.nodes.get(v, {}).get("type") == "Module"
            ]
            if children:
                child_names = sorted(set(c[1] for c in children if c[1]))
                questions.append(EngineeringQuestion(
                    question_id="",
                    question_text=f"What submodules does {mod_name} instantiate?",
                    category="structural",
                    difficulty="easy",
                    hops_required=1,
                    gold_answer=f"{mod_name} instantiates: {', '.join(child_names)}",
                    evidence_nodes=[mod_id] + [c[0] for c in children],
                    evidence_edges=[(mod_id, "instantiates", c[0]) for c in children],
                    design_name=self.design_name,
                ))

            # Q: "What registers are in module X?"
            regs = [
                (v, G.nodes[v].get("name", v))
                for _, v, d in G.out_edges(mod_id, data=True)
                if d.get("relation") == "contains_register"
            ]
            if regs:
                reg_names = sorted(set(r[1] for r in regs if r[1]))[:10]
                questions.append(EngineeringQuestion(
                    question_id="",
                    question_text=f"What registers does module {mod_name} contain?",
                    category="structural",
                    difficulty="easy",
                    hops_required=1,
                    gold_answer=f"{mod_name} contains {len(regs)} registers: {', '.join(reg_names)}",
                    evidence_nodes=[mod_id] + [r[0] for r in regs],
                    evidence_edges=[(mod_id, "contains_register", r[0]) for r in regs],
                    design_name=self.design_name,
                ))

            # Q: "What parameters does module X have?"
            params = [
                (v, G.nodes[v].get("name", v))
                for _, v, d in G.out_edges(mod_id, data=True)
                if d.get("relation") == "has_parameter"
            ]
            if params:
                param_info = []
                for pid, pname in params:
                    val = G.nodes.get(pid, {}).get("default_value", "?")
                    param_info.append(f"{pname}={val}")
                questions.append(EngineeringQuestion(
                    question_id="",
                    question_text=f"What parameters does module {mod_name} have?",
                    category="structural",
                    difficulty="easy",
                    hops_required=1,
                    gold_answer=f"{mod_name} has parameters: {', '.join(param_info)}",
                    evidence_nodes=[mod_id] + [p[0] for p in params],
                    evidence_edges=[(mod_id, "has_parameter", p[0]) for p in params],
                    design_name=self.design_name,
                ))

        return questions

    # ------------------------------------------------------------------
    # Cross-artifact (2-3 hop, medium)
    # ------------------------------------------------------------------

    def _gen_cross_artifact(self, rng: random.Random, max_n: int) -> list[EngineeringQuestion]:
        G = self.graph
        questions: list[EngineeringQuestion] = []
        modules = list(self._modules)
        rng.shuffle(modules)

        for mod_id in modules[:max_n]:
            mod_name = G.nodes[mod_id].get("name", mod_id.split("::")[-1])

            # Q: "What clocks drive registers in module X?"
            reg_clocks: list[tuple[str, str, str]] = []  # (reg_id, clk_id, clk_name)
            evidence_edges = []
            for _, reg_id, d in G.out_edges(mod_id, data=True):
                if d.get("relation") != "contains_register":
                    continue
                for _, clk_id, d2 in G.out_edges(reg_id, data=True):
                    if d2.get("relation") == "clocked_by":
                        clk_name = G.nodes.get(clk_id, {}).get("name", clk_id)
                        reg_clocks.append((reg_id, clk_id, clk_name))
                        evidence_edges.append((mod_id, "contains_register", reg_id))
                        evidence_edges.append((reg_id, "clocked_by", clk_id))

            if reg_clocks:
                clk_names = sorted(set(c[2] for c in reg_clocks))
                all_nodes = [mod_id] + list(set(
                    [c[0] for c in reg_clocks] + [c[1] for c in reg_clocks]
                ))
                questions.append(EngineeringQuestion(
                    question_id="",
                    question_text=f"What clocks drive registers in module {mod_name}?",
                    category="cross_artifact",
                    difficulty="medium",
                    hops_required=2,
                    gold_answer=f"Clocks driving registers in {mod_name}: {', '.join(clk_names)}",
                    evidence_nodes=all_nodes,
                    evidence_edges=evidence_edges,
                    design_name=self.design_name,
                ))

            # Q: "What constraints apply to module X?"
            port_constraints: list[tuple[str, str, str]] = []
            evidence_edges_c = []
            for _, port_id, d in G.out_edges(mod_id, data=True):
                if d.get("relation") != "has_port":
                    continue
                for cst_id, _, d2 in G.in_edges(port_id, data=True):
                    if d2.get("relation") == "applies_to":
                        cst_data = G.nodes.get(cst_id, {})
                        cst_type = cst_data.get("constraint_type", "?")
                        port_constraints.append((port_id, cst_id, cst_type))
                        evidence_edges_c.append((mod_id, "has_port", port_id))
                        evidence_edges_c.append((cst_id, "applies_to", port_id))

            if port_constraints:
                cst_types = {}
                for _, _, ct in port_constraints:
                    cst_types[ct] = cst_types.get(ct, 0) + 1
                cst_summary = ", ".join(f"{c} ({n})" for c, n in sorted(cst_types.items()))
                all_nodes = [mod_id] + list(set(
                    [c[0] for c in port_constraints] + [c[1] for c in port_constraints]
                ))
                questions.append(EngineeringQuestion(
                    question_id="",
                    question_text=f"What constraints apply to module {mod_name}?",
                    category="cross_artifact",
                    difficulty="medium",
                    hops_required=2,
                    gold_answer=f"Constraints on {mod_name}: {cst_summary}",
                    evidence_nodes=all_nodes,
                    evidence_edges=evidence_edges_c,
                    design_name=self.design_name,
                ))

            # Q: "What assertions verify module X?"
            assertions = [
                (v, G.nodes[v].get("name", v), G.nodes[v].get("assertion_type", "?"))
                for _, v, d in G.out_edges(mod_id, data=True)
                if d.get("relation") == "verified_by"
            ]
            if assertions:
                asrt_info = [f"{a[1]} ({a[2]})" for a in assertions]
                questions.append(EngineeringQuestion(
                    question_id="",
                    question_text=f"What assertions verify module {mod_name}?",
                    category="cross_artifact",
                    difficulty="medium",
                    hops_required=1,
                    gold_answer=f"Assertions for {mod_name}: {', '.join(asrt_info)}",
                    evidence_nodes=[mod_id] + [a[0] for a in assertions],
                    evidence_edges=[(mod_id, "verified_by", a[0]) for a in assertions],
                    design_name=self.design_name,
                ))

        return questions

    # ------------------------------------------------------------------
    # Temporal (Git-based, medium)
    # ------------------------------------------------------------------

    def _gen_temporal(self, rng: random.Random, max_n: int) -> list[EngineeringQuestion]:
        G = self.graph
        questions: list[EngineeringQuestion] = []

        if not self._commits:
            return questions

        modules = list(self._modules)
        rng.shuffle(modules)

        for mod_id in modules[:max_n]:
            mod_name = G.nodes[mod_id].get("name", mod_id.split("::")[-1])

            # Q: "Which commits modified module X?"
            commits = []
            for src, _, d in G.in_edges(mod_id, data=True):
                if d.get("relation") == "modifies":
                    cdata = G.nodes.get(src, {})
                    commits.append((src, cdata.get("summary", cdata.get("message", src))))

            if commits:
                commit_strs = [f"{c[0].split('::')[-1][:8]}: {c[1][:60]}" for c in commits[:5]]
                questions.append(EngineeringQuestion(
                    question_id="",
                    question_text=f"Which commits modified module {mod_name}?",
                    category="temporal",
                    difficulty="medium",
                    hops_required=1,
                    gold_answer=f"{len(commits)} commits modified {mod_name}: {'; '.join(commit_strs)}",
                    evidence_nodes=[mod_id] + [c[0] for c in commits],
                    evidence_edges=[(c[0], "modifies", mod_id) for c in commits],
                    design_name=self.design_name,
                ))

        return questions

    # ------------------------------------------------------------------
    # What-if (impact, hard)
    # ------------------------------------------------------------------

    def _gen_what_if(self, rng: random.Random, max_n: int) -> list[EngineeringQuestion]:
        G = self.graph
        questions: list[EngineeringQuestion] = []
        modules = list(self._modules)
        rng.shuffle(modules)

        for mod_id in modules[:max_n]:
            mod_name = G.nodes[mod_id].get("name", mod_id.split("::")[-1])

            # Collect all neighbors within 2 hops
            neighbors_1: set[str] = set()
            for _, v, d in G.out_edges(mod_id, data=True):
                neighbors_1.add(v)
            for u, _, d in G.in_edges(mod_id, data=True):
                neighbors_1.add(u)

            neighbors_2: set[str] = set()
            for n1 in neighbors_1:
                for _, v, d in G.out_edges(n1, data=True):
                    neighbors_2.add(v)
                for u, _, d in G.in_edges(n1, data=True):
                    neighbors_2.add(u)
            neighbors_2 -= {mod_id}
            all_affected = neighbors_1 | neighbors_2

            # Count affected by type
            type_counts: dict[str, int] = {}
            for nid in all_affected:
                t = G.nodes.get(nid, {}).get("type", "Unknown")
                type_counts[t] = type_counts.get(t, 0) + 1

            if len(all_affected) >= 3:
                impact_str = ", ".join(f"{c} {t}s" for t, c in sorted(type_counts.items()) if c > 0)
                questions.append(EngineeringQuestion(
                    question_id="",
                    question_text=f"What is the impact of changing module {mod_name}?",
                    category="what_if",
                    difficulty="hard",
                    hops_required=2,
                    gold_answer=f"Changing {mod_name} could affect {len(all_affected)} artifacts: {impact_str}",
                    evidence_nodes=[mod_id] + list(all_affected),
                    evidence_edges=[],  # Too many to enumerate
                    design_name=self.design_name,
                ))

            # Q: "Why is module X critical/risky?"
            parent_modules = [
                G.nodes.get(u, {}).get("name", u)
                for u, _, d in G.in_edges(mod_id, data=True)
                if d.get("relation") == "instantiates"
            ]
            if parent_modules or len(all_affected) >= 5:
                risk_factors = []
                if parent_modules:
                    risk_factors.append(f"instantiated by {len(parent_modules)} parent modules ({', '.join(parent_modules[:3])})")
                if type_counts.get("Port", 0) > 5:
                    risk_factors.append(f"has {type_counts['Port']} connected ports")
                if type_counts.get("Constraint", 0) > 0:
                    risk_factors.append(f"{type_counts['Constraint']} constraints depend on it")
                if type_counts.get("Assertion", 0) > 0:
                    risk_factors.append(f"{type_counts['Assertion']} assertions verify it")

                if risk_factors:
                    questions.append(EngineeringQuestion(
                        question_id="",
                        question_text=f"Why is module {mod_name} risky to modify?",
                        category="what_if",
                        difficulty="hard",
                        hops_required=2,
                        gold_answer=f"{mod_name} is risky because: {'; '.join(risk_factors)}",
                        evidence_nodes=[mod_id] + list(neighbors_1),
                        evidence_edges=[],
                        design_name=self.design_name,
                    ))

        return questions


def save_questions(questions: list[EngineeringQuestion], path: str | Path):
    """Save questions to a JSON file."""
    data = [q.to_dict() for q in questions]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved %d questions to %s", len(questions), path)


def load_questions(path: str | Path) -> list[EngineeringQuestion]:
    """Load questions from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return [
        EngineeringQuestion(
            question_id=d["question_id"],
            question_text=d["question_text"],
            category=d["category"],
            difficulty=d["difficulty"],
            hops_required=d["hops_required"],
            gold_answer=d["gold_answer"],
            evidence_nodes=d.get("evidence_nodes", []),
            evidence_edges=[tuple(e) for e in d.get("evidence_edges", [])],
            design_name=d.get("design_name", ""),
        )
        for d in data
    ]
