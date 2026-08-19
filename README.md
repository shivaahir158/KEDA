# KEDA: Knowledge-Graph-Enhanced Reasoning for Cross-Artifact EDA

A research framework that builds unified knowledge graphs from hardware design artifacts (RTL, constraints, assertions, Git history) and uses them for change-impact analysis and engineering question answering.

# Author: Shiva Ahir, PhD Candidate, Stony Brook University

## Problem

Semiconductor designs produce tightly coupled artifacts: RTL source code, timing constraints (SDC), formal assertions (SVA), testbenches, and version control history. A change to one artifact can invalidate others across multiple categories. Existing tools treat these artifacts in isolation.

KEDA constructs a cross-artifact knowledge graph that captures semantic relationships between all artifact types, then uses graph traversal and LLM-grounded retrieval (GraphRAG) to reason about design changes and answer engineering questions.

## Results

### Change-Impact Analysis (RQ1)

Evaluated on 5 open-source designs (207 controlled changes, 38 modules):

| Method | Avg Recall | Avg Precision | Avg F1 |
|--------|-----------|--------------|--------|
| **weighted_bfs (KEDA)** | **0.94** | 0.28 | 0.42 |
| structural (baseline) | 0.02 | 0.52 | 0.04 |
| F1 gain | | | **+0.32 to +0.51** |

### Engineering QA (RQ2)

Evaluated on 81 questions across 3 designs using GPT-4o-mini:

| Method | Evidence Recall | Evidence Precision | Key Entity Recall |
|--------|----------------|-------------------|-------------------|
| LLM-only | 0.000 | 0.000 | 0.343 |
| Vector RAG | 0.633 | 0.404 | 0.714 |
| **KG + LLM (KEDA-QA)** | **0.833** | **0.899** | **0.770** |
| KG + Vector + LLM | 0.859 | 0.487 | 0.817 |

Key finding: LLM-only retrieves zero evidence nodes (cannot ground answers without the KG). KG-based methods achieve 90% evidence precision.

## Architecture

```
                    Graph Construction Pipeline
    +-----------+  +-----------+  +-----------+  +-----------+
    |   Yosys   |  |    SDC    |  |    SVA    |  |    Git    |
    | Extractor |  | Extractor |  | Extractor |  | Extractor |
    +-----------+  +-----------+  +-----------+  +-----------+
          |              |              |              |
          v              v              v              v
    +----------------------------------------------------+
    |              Unified KG Builder                     |
    |         (NetworkX DiGraph, 10 node types,           |
    |          17 edge relations)                         |
    +----------------------------------------------------+
                         |
          +--------------+--------------+
          v              v              v
    +-----------+  +-----------+  +-----------+
    |  Change   |  | GraphRAG  |  |   Risk    |
    |  Impact   |  |    QA     |  |Propagation|
    | Analyzer  |  |  Engine   |  |  Engine   |
    +-----------+  +-----------+  +-----------+
```

### Node Types
Module, Port, Register, Clock, Parameter, Instance, Constraint, Assertion, Test, Commit

### Edge Relations
instantiates, has_port, contains_register, clocked_by, constrained_by, applies_to, verified_by, covers, modifies, drives, has_parameter, and more.

## Project Structure

```
keda/
  extractors/
    yosys_extractor.py    # RTL structure via Yosys JSON
    sdc_extractor.py      # SDC timing constraint parsing
    sva_extractor.py      # SVA assertions + testbench detection
    git_extractor.py      # Git commit history + module linking
  graph/
    builder.py            # Unified KGBuilder orchestrating all extractors
  analysis/
    impact.py             # Change-impact analysis (5 methods)
    changegen.py          # Controlled change generation (7 types)
  llm/
    qa_engine.py          # GraphRAG QA engine (5 methods)
    graph_retriever.py    # Subgraph extraction strategies
    query_engine.py       # NL-to-graph-query parser
    question_gen.py       # Engineering question generator
    evaluator.py          # QA evaluation metrics

evaluate_designs.py       # Change-impact benchmark runner
evaluate_qa.py            # QA benchmark runner
tests/                    # 186 tests
results/                  # Benchmark results (JSON)
```

## Quick Start

### Prerequisites

- Python 3.10+
- [Yosys](https://github.com/YosysHQ/yosys) (for RTL parsing)
- Python packages: `networkx`, `gitpython`, `numpy`, `sentence-transformers`
- Optional: `openai` or `anthropic` (for LLM-based QA)

### Installation

```bash
pip install networkx gitpython numpy sentence-transformers openai
```

### Build a Knowledge Graph

```python
from keda.graph.builder import KGBuilder, DesignConfig

config = DesignConfig(
    name="my_design",
    rtl_files=["src/top.v", "src/uart.v"],
    top_module="top",
    sdc_files=["constraints/timing.sdc"],
    repo_path=".",
)

builder = KGBuilder()
result = builder.build(config)
print(f"Graph: {result.graph.number_of_nodes()} nodes, {result.graph.number_of_edges()} edges")
```

### Run Change-Impact Analysis

```python
from keda.analysis.impact import ChangeImpactAnalyzer

analyzer = ChangeImpactAnalyzer(result.graph)
impact = analyzer.from_modules(["uart_tx"], design_name="my_design")

for artifact in impact.top_k(10):
    print(f"  [{artifact.artifact_type}] {artifact.name} risk={artifact.risk_score:.3f}")
```

### Ask Questions with GraphRAG

```python
from keda.llm.qa_engine import QAEngine
import openai

qa = QAEngine(
    graph=result.graph,
    design_name="my_design",
    llm_client=openai.OpenAI(),
    llm_model="gpt-4o-mini",
    llm_provider="openai",
)

answer = qa.answer("What clocks drive registers in module uart_tx?", method="keda_qa")
print(answer.answer)
```

### Run Benchmarks

```bash
# Change-impact analysis benchmark (no LLM needed)
python evaluate_designs.py

# QA benchmark (requires OpenAI API key)
OPENAI_API_KEY=your-key python evaluate_qa.py --with-llm
```

### Run Tests

```bash
pytest tests/ -v
```

## Evaluated Designs

| Design | Modules | KG Nodes | KG Edges | Source |
|--------|---------|----------|----------|--------|
| UART fixture | 4 | 89 | 178 | Custom test fixture |
| PicoRV32 | 8 | 525 | 994 | [github.com/YosysHQ/picorv32](https://github.com/YosysHQ/picorv32) |
| verilog-uart | 3 | 124 | 173 | [github.com/alexforencich/verilog-uart](https://github.com/alexforencich/verilog-uart) |
| AXI DMA | 3 | 154 | 219 | [github.com/alexforencich/verilog-axi](https://github.com/alexforencich/verilog-axi) |
| mor1kx | 20 | 1784 | 2979 | [github.com/openrisc/mor1kx](https://github.com/openrisc/mor1kx) |

## Change Types

The controlled change generator produces 7 types of design changes with automatically computed ground truth:

1. **PARAMETER** - Module parameter modifications
2. **WIDTH** - Signal/port width changes
3. **CLOCK** - Clock signal modifications
4. **RESET** - Reset signal changes
5. **DEPENDENCY** - Module instantiation modifications
6. **CONSTRAINT** - SDC timing constraint changes
7. **HIERARCHY** - Module hierarchy restructuring

## QA Methods Compared

| Method | Description |
|--------|-------------|
| **llm_only** | LLM with graph schema only (no retrieval) |
| **kg_only** | Structured graph query, formatted answer (no LLM) |
| **keda_qa** | Graph retrieval + LLM synthesis (primary method) |
| **vector_rag** | Sentence-transformer embeddings + LLM |
| **keda_full** | Graph + vector retrieval + LLM |

## License

Research use. See individual design repositories for their respective licenses.
