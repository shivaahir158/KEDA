# KEDA: Knowledge-Graph-Enhanced Reasoning for Cross-Artifact Electronic Design Automation

## Complete Research Plan

---

## 1. Research Problem Statement

Semiconductor design produces a heterogeneous web of tightly coupled artifacts: RTL source code (Verilog/SystemVerilog), module hierarchies, timing constraints (SDC), clock-domain-crossing (CDC) reports, formal assertions (SVA), functional verification tests, specification/requirements documents, synthesis outputs, and version-control history (Git commits and pull requests). These artifacts are semantically interdependent—a change to an RTL module may invalidate timing constraints, require re-verification of assertions, affect downstream clock domains, and violate specification requirements—yet existing EDA tools and emerging LLM-based EDA assistants treat them largely in isolation.

**Problem:** There is no unified, queryable representation that captures cross-artifact semantic relationships in a hardware design, and consequently no systematic method for reasoning about the transitive impact of design changes across artifact boundaries. Current approaches to change-impact analysis, design-risk assessment, and engineering question answering in EDA are limited to single-artifact analysis (e.g., RTL dependency graphs, constraint checking, or code-level version diffs) and fail to propagate reasoning across the full artifact graph.

**This research investigates whether a multi-artifact EDA knowledge graph, combined with LLM-based reasoning, can improve cross-artifact change-impact analysis, engineering question answering, and design-risk propagation compared to single-artifact, lexical, embedding-based, and LLM-only approaches.**

---

## 2. Research Questions

**RQ1 (Change-Impact Analysis):** Does a cross-artifact EDA knowledge graph improve precision and recall of change-impact analysis compared to lexical search, embedding-based retrieval, and LLM-only reasoning, when a design change (e.g., RTL modification) may affect artifacts across multiple categories (constraints, assertions, tests, timing paths, requirements)?

**RQ2 (Engineering QA):** Does KG-grounded retrieval (GraphRAG) reduce hallucination and improve answer correctness for cross-artifact engineering questions compared to raw-context LLM prompting and vector-based RAG?

**RQ3 (Risk Propagation):** Can multi-hop graph traversal identify indirect downstream risks (e.g., a module change affecting a distant timing path through intermediate clock and constraint relationships) that file-level and lexical approaches systematically miss?

**RQ4 (Ablation — Artifact Contribution):** Which artifact categories (constraints, requirements, verification, Git history) contribute most to cross-artifact reasoning accuracy, and at what graph depth does impact propagation saturate?

**RQ5 (Scalability):** Does the approach scale to medium-to-large open-source SoC designs (100K+ lines of RTL) without prohibitive graph construction or query costs?

---

## 3. Novelty Analysis

### What this is NOT:
- Simply building a knowledge graph from RTL (RTL dependency graphs, module hierarchies, and netlist representations already exist in every synthesis tool)
- Simply applying LLMs to EDA tasks (a rapidly growing field with many existing contributions)
- Simply building a GraphRAG system (generic GraphRAG methods exist for text corpora)

### What prior work already does:

| Area | Existing Work | What They Do |
|------|--------------|--------------|
| RTL graphs | Synthesis tools (Yosys, Synopsys DC), academic work on circuit graph representations | Build gate-level or module-level dependency graphs from RTL |
| KGs for EDA | Limited: some work on IC manufacturing KGs, PDK KGs, analog circuit KGs | Focus on manufacturing/process knowledge, not cross-artifact design reasoning |
| Semiconductor KGs | Industry efforts (Samsung, TSMC internal tools) | Proprietary, manufacturing-focused, not open-source or cross-artifact |
| LLMs for EDA | ChipGPT, RTLCoder, ChipChat, VerilogEval, RTLLM, AutoChip, ChatEDA | Focus on RTL generation, bug detection, or single-artifact QA; do not use structured cross-artifact KGs |
| Graph-based HW reasoning | GNN-based power/timing prediction, circuit-GNNs | Operate on gate-level netlist graphs for prediction tasks; do not span artifact types |
| RAG for EDA | Emerging work on retrieval-augmented EDA assistants | Use vector retrieval over documentation; do not exploit structured graph relationships |
| Cross-artifact SE graphs | Software engineering: code-requirement traceability, change-impact analysis | Focus on software (Java/C++), not hardware; do not include timing constraints, clock domains, SDC, CDC |
| Traceability | Requirements-to-test traceability in safety-critical systems (DO-254, ISO 26262) | Manual or semi-automated; not unified with RTL structure and timing analysis |

### What KEDA contributes that prior work does not:

1. **Cross-artifact ontology for hardware design** that unifies RTL structure, timing constraints, clock domains, verification artifacts, requirements, and version history in a single queryable graph. No prior work provides this unified schema.

2. **Automated graph construction pipeline** from open-source tooling (Yosys, Verible, Git) rather than proprietary tool APIs.

3. **Cross-artifact change-impact analysis** that propagates through multiple artifact types (RTL → constraint → timing path → requirement), which no existing EDA tool or LLM approach provides systematically.

4. **GraphRAG for EDA** — applying structured graph retrieval to augment LLM reasoning specifically for hardware engineering questions, with domain-specific graph schemas rather than generic text graphs.

5. **Quantitative evaluation** on open-source designs with controlled design changes and ground-truth impact sets, providing the first benchmark for cross-artifact EDA reasoning.

### Critical honesty about novelty risks:

- **Risk 1:** The ontology/schema contribution alone is insufficient for a top venue. The paper MUST have strong experimental results showing quantitative improvement.
- **Risk 2:** Concurrent work in LLMs-for-EDA is advancing rapidly. By submission time, others may have explored graph-augmented EDA reasoning.
- **Risk 3:** The cross-artifact idea is conceptually natural. The contribution must be in the rigorous implementation, evaluation, and demonstrated utility—not just the idea.

---

## 4. EDA Ontology / Schema

### Node Types

| Node Type | Description | Key Attributes |
|-----------|-------------|----------------|
| `Module` | RTL module definition | name, file_path, line_range, language, LOC, parameter_list |
| `Instance` | Instantiation of a module | instance_name, parent_module, position |
| `Port` | Module port | name, direction (input/output/inout), width, type |
| `Net` | Internal wire/signal | name, width, type (wire/reg/logic) |
| `Register` | Sequential element (flip-flop) | name, width, reset_value, clock_edge |
| `Clock` | Clock signal/domain | name, frequency, source, domain_id |
| `ClockDomain` | Group of signals sharing a clock | domain_name, clock_name |
| `Requirement` | Specification requirement | req_id, text, category, priority |
| `Constraint` | SDC/timing constraint | constraint_type (create_clock, set_input_delay, etc.), value, target |
| `Assertion` | SVA/formal assertion | name, type (assert/assume/cover), file_path, module_scope |
| `Test` | Verification test/testbench | name, type (unit/integration/formal), file_path, modules_exercised |
| `TimingPath` | Timing path from STA | startpoint, endpoint, slack, path_group |
| `Violation` | Design rule or timing violation | type, severity, description |
| `PullRequest` | Git PR / merge request | pr_id, author, date, files_changed, description |
| `Commit` | Git commit | sha, author, date, message, files_changed |
| `Parameter` | Module parameter/generic | name, type, default_value, module |
| `FSM` | Finite state machine | name, states, module |
| `CDCCrossing` | Clock-domain crossing point | source_domain, dest_domain, synchronizer_type |

### Edge Types

| Edge Type | Source → Target | Attributes |
|-----------|----------------|------------|
| `instantiates` | Module → Module | instance_name |
| `has_port` | Module → Port | |
| `has_net` | Module → Net | |
| `connected_to` | Port → Net, Port → Port | |
| `drives` | Net → Net, Port → Net | |
| `driven_by` | Net → Port, Net → Net | |
| `contains_register` | Module → Register | |
| `clocked_by` | Register → Clock | edge_type (posedge/negedge) |
| `in_domain` | Register/Module → ClockDomain | |
| `crosses_domain` | CDCCrossing → ClockDomain (×2) | |
| `constrained_by` | Port/Clock/TimingPath → Constraint | |
| `applies_to` | Constraint → Port/Clock/Module | |
| `implemented_by` | Requirement → Module | confidence |
| `verifies` | Assertion → Requirement/Module | |
| `covers` | Test → Module/Requirement | coverage_type |
| `depends_on` | Module → Module | dependency_type (structural/data/control) |
| `modifies` | PullRequest/Commit → Module | lines_changed |
| `affects` | Violation → Module/Instance/TimingPath | |
| `parameter_of` | Parameter → Module | |
| `uses_parameter` | Instance → Parameter | override_value |
| `part_of_fsm` | Register → FSM | |
| `child_of` | Instance → Instance | (hierarchy) |
| `same_signal` | Net → Net | (cross-hierarchy) |

### Schema Diagram (ASCII)

```
Requirement ──implemented_by──► Module ◄──modifies── PullRequest/Commit
    │                            │  │  │
    │                            │  │  └──has_port──► Port ◄──applies_to── Constraint
  verifies                       │  │                  │
    │                            │  instantiates       connected_to
    ▼                            │  │                  │
Assertion                        │  ▼                  ▼
    │                            │ Instance            Net
  verifies                       │                     │
    │                            │                   drives/driven_by
    ▼                            │                     │
  Module ◄──covers── Test        └──contains_register──► Register
                                                          │
                                                        clocked_by
                                                          │
                                                          ▼
Violation ──affects──► TimingPath ◄──constrained_by── Constraint ──applies_to──► Clock
```

---

## 5. System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        KEDA System                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────┐          │
│  │           Graph Construction Pipeline              │          │
│  │                                                    │          │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐          │          │
│  │  │ RTL      │ │Constraint│ │Verification│          │          │
│  │  │ Extractor│ │ Extractor│ │ Extractor  │          │          │
│  │  │(Yosys/   │ │(SDC      │ │(SVA/Test   │          │          │
│  │  │ Verible) │ │ Parser)  │ │ Parser)    │          │          │
│  │  └────┬─────┘ └────┬─────┘ └─────┬─────┘          │          │
│  │       │             │             │                │          │
│  │  ┌────┴─────┐ ┌────┴─────┐ ┌─────┴─────┐          │          │
│  │  │ Git      │ │Requirement│ │ Synthesis │          │          │
│  │  │ Extractor│ │ Linker   │ │ Extractor  │          │          │
│  │  │(GitPython│ │(LLM-     │ │(Yosys JSON)│          │          │
│  │  │)        │ │ assisted)│ │            │          │          │
│  │  └────┬─────┘ └────┬─────┘ └─────┬─────┘          │          │
│  │       │             │             │                │          │
│  │       └─────────────┼─────────────┘                │          │
│  │                     ▼                              │          │
│  │            ┌────────────────┐                      │          │
│  │            │  Graph Builder  │                      │          │
│  │            │  & Linker       │                      │          │
│  │            └───────┬────────┘                      │          │
│  └────────────────────┼───────────────────────────────┘          │
│                       ▼                                         │
│  ┌────────────────────────────────────────────┐                 │
│  │         EDA Knowledge Graph                 │                 │
│  │      (NetworkX / Neo4j / igraph)            │                 │
│  └──────────────┬──────────────────────────────┘                 │
│                 │                                                │
│    ┌────────────┼────────────┐                                  │
│    ▼            ▼            ▼                                  │
│ ┌────────┐ ┌────────┐ ┌──────────┐                             │
│ │Change  │ │GraphRAG│ │Risk      │                              │
│ │Impact  │ │QA      │ │Propagation│                             │
│ │Analyzer│ │Engine  │ │Engine     │                             │
│ └───┬────┘ └───┬────┘ └────┬─────┘                             │
│     │          │           │                                    │
│     └──────────┼───────────┘                                    │
│                ▼                                                │
│  ┌─────────────────────────────────┐                            │
│  │   LLM Reasoning Layer           │                            │
│  │   (Query generation, answer     │                            │
│  │    synthesis, explanation)       │                            │
│  └─────────────────────────────────┘                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Component Details

**Graph Construction Pipeline:**
1. **RTL Extractor:** Parses Verilog/SystemVerilog using Yosys (JSON netlist export) and Verible (AST/syntax analysis). Extracts modules, instances, ports, nets, registers, parameters, FSMs, and structural relationships.
2. **Constraint Extractor:** Parses SDC files to extract clock definitions, timing constraints, I/O delays, false/multicycle paths. Links constraints to ports/clocks/modules.
3. **Verification Extractor:** Parses SVA assertions from RTL files, identifies testbench files and test names, extracts coverage relationships.
4. **Git Extractor:** Uses GitPython to extract commit/PR history, file-change mappings, and links changes to modules.
5. **Requirement Linker:** Maps natural-language requirements to modules using keyword matching and LLM-assisted linking.
6. **Synthesis Extractor:** Runs Yosys synthesis to obtain gate counts, timing estimates, and elaborated hierarchy.
7. **Graph Builder & Linker:** Merges all extracted relationships into a unified graph, resolving cross-references between artifacts.

**Reasoning Engines:**
1. **Change Impact Analyzer:** Given a changed module/file, performs graph traversal to find all transitively affected artifacts.
2. **GraphRAG QA Engine:** Converts natural-language questions to graph queries, retrieves relevant subgraphs, and feeds them to an LLM for grounded answers.
3. **Risk Propagation Engine:** Computes risk scores across the graph using weighted traversal or Personalized PageRank.

---

## 6. Extraction Pipeline

### Step-by-step pipeline:

```
Step 1: Clone repository
  └─► git clone <repo_url>

Step 2: RTL Structure Extraction
  ├─► Verible: verible-verilog-syntax --export_json *.v *.sv
  │     → Module definitions, port lists, parameter lists, instance names
  │     → Line ranges, file locations
  ├─► Yosys: read_verilog *.v; hierarchy; proc; write_json design.json
  │     → Elaborated hierarchy, module graph, net connections
  │     → Register identification (cells with CLK inputs)
  │     → Clock tree structure
  └─► Custom Python parser (regex-based fallback for constructs
      Yosys/Verible miss): FSM detection, always-block analysis

Step 3: Constraint Extraction
  └─► Custom SDC parser (Python):
        Parse create_clock, create_generated_clock,
        set_input_delay, set_output_delay,
        set_false_path, set_multicycle_path,
        set_max_delay, set_min_delay
        → Create Constraint nodes, link to Port/Clock/Module nodes

Step 4: Assertion/Verification Extraction
  ├─► Regex/Verible parser for SVA: assert property, assume, cover
  │     → Create Assertion nodes, link to containing Module
  └─► Testbench scanner: identify test files, extract test names,
      parse module instantiations in testbenches
        → Create Test nodes, link to covered Modules

Step 5: Git History Extraction
  └─► GitPython:
        For each commit: extract changed files, map to Module nodes
        For PRs (GitHub API if available): extract PR metadata
        → Create Commit/PR nodes, create `modifies` edges

Step 6: Requirement Extraction (synthetic or from docs)
  ├─► Parse README, specification docs (if available)
  │     → Create Requirement nodes
  └─► LLM-assisted linking: for each Requirement, identify which
      Module(s) implement it
        → Create `implemented_by` edges with confidence scores

Step 7: Synthesis & Static Analysis
  └─► Yosys synthesis (generic target):
        → Gate counts per module
        → Critical paths (approximate)
        → Area/resource estimates

Step 8: Graph Assembly
  └─► Python script:
        Merge all extracted nodes and edges
        Resolve cross-references (e.g., port names ↔ SDC targets)
        Assign unique IDs, validate graph consistency
        Export to NetworkX (primary) and optionally Neo4j

Step 9: Embedding Generation (for baselines)
  └─► Generate text embeddings for each node (using node attributes
      and context) for vector-RAG baseline comparison
```

### Key implementation details:

```python
# Pseudo-code for Yosys-based extraction
import json, subprocess

def extract_yosys_graph(verilog_files, top_module):
    # Run Yosys
    script = f"""
    read_verilog -sv {' '.join(verilog_files)}
    hierarchy -top {top_module}
    proc
    flatten  # optional: for flat netlist
    write_json design.json
    """
    subprocess.run(['yosys', '-p', script], check=True)

    with open('design.json') as f:
        design = json.load(f)

    graph = nx.DiGraph()

    for mod_name, mod_data in design['modules'].items():
        graph.add_node(mod_name, type='Module', attributes=mod_data.get('attributes', {}))

        for port_name, port_data in mod_data.get('ports', {}).items():
            port_id = f"{mod_name}.{port_name}"
            graph.add_node(port_id, type='Port',
                          direction=port_data['direction'],
                          bits=port_data['bits'])
            graph.add_edge(mod_name, port_id, relation='has_port')

        for cell_name, cell_data in mod_data.get('cells', {}).items():
            cell_type = cell_data['type']
            if cell_type in design['modules']:  # it's a module instantiation
                graph.add_edge(mod_name, cell_type, relation='instantiates',
                              instance_name=cell_name)

            # Identify registers (DFF cells)
            if 'DFF' in cell_type or 'dff' in cell_type:
                reg_id = f"{mod_name}.{cell_name}"
                graph.add_node(reg_id, type='Register')
                graph.add_edge(mod_name, reg_id, relation='contains_register')
                # Link clock
                if 'CLK' in cell_data.get('connections', {}):
                    clk_bits = cell_data['connections']['CLK']
                    # resolve clock signal...

    return graph
```

```python
# SDC constraint parser (simplified)
import re

def parse_sdc(sdc_file):
    constraints = []
    with open(sdc_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('create_clock'):
                match = re.match(
                    r'create_clock\s+-period\s+([\d.]+)\s+.*-name\s+(\w+).*\[get_ports\s+(\w+)\]',
                    line)
                if match:
                    constraints.append({
                        'type': 'create_clock',
                        'period': float(match.group(1)),
                        'name': match.group(2),
                        'port': match.group(3)
                    })
            elif line.startswith('set_input_delay'):
                # similar parsing...
                pass
    return constraints
```

---

## 7. Artifact Availability Analysis

### Realistically obtainable from public open-source projects:

| Artifact | Availability | Source | Notes |
|----------|-------------|--------|-------|
| RTL (Verilog/SV) | **High** | Repository source | Core artifact, always available |
| Module hierarchy | **High** | Yosys elaboration | Automatically derivable from RTL |
| Port/Net/Register | **High** | Yosys/Verible parsing | Automatically derivable from RTL |
| Parameters | **High** | RTL source parsing | Directly parseable |
| Git commits | **High** | Git history | Always available for any Git repo |
| Pull Requests | **Medium-High** | GitHub API | Available for GitHub-hosted projects |
| SDC constraints | **Medium** | Some repos include constraints | OpenTitan has constraints; many smaller projects do not |
| SVA assertions | **Medium** | In-source assertions | OpenTitan, Ibex have assertions; many projects lack them |
| Testbenches | **Medium-High** | Repository test directories | Most serious projects have testbenches |
| FSMs | **Medium** | Derivable from RTL | Can be automatically detected with Yosys `fsm_detect` |
| Clock domains | **Medium** | Derivable from RTL + constraints | Can be inferred from Yosys or constraints |
| CDC crossings | **Low-Medium** | Requires CDC analysis tool | Can be partially inferred; formal CDC tools are commercial |

### Must be synthetically generated:

| Artifact | Why Synthetic | Generation Strategy |
|----------|--------------|-------------------|
| **Specification/Requirements** | Open-source HW projects rarely have formal specs | Generate from module documentation, README, and LLM-based summarization of module purpose. Create structured requirement documents per module. |
| **Timing reports (STA)** | Requires commercial STA tools or specific FPGA targets | Run Yosys synthesis with timing-annotated cells, or generate synthetic timing paths based on register-to-register paths in the elaborated design. |
| **Timing violations** | Occur only under specific constraints/targets | Intentionally create constraint violations by tightening clock periods or adding false constraints. |
| **CDC reports** | Formal CDC tools are commercial | Infer CDC crossings from clock-domain analysis in Yosys; generate synthetic CDC reports. |
| **Requirement-to-Module traceability** | Rarely exists formally | Create manually for evaluation subset; use LLM-assisted linking for full dataset with human validation. |

### Recommended open-source repositories (20-50 target):

**Tier 1: Large, well-documented (rich artifact availability)**
1. OpenTitan (lowRISC) — full SoC, SDC, SVA, tests, docs
2. Ibex (lowRISC) — RISC-V core, well-documented
3. PULP Platform cores (CV32E40P, CVA6) — RISC-V, verification suites
4. Rocket Chip (Chisel→Verilog) — RISC-V SoC generator
5. OpenPiton — multi-core research processor
6. NVDLA — deep learning accelerator

**Tier 2: Medium, good RTL structure**
7. PicoRV32 — compact RISC-V
8. VexRiscv (SpinalHDL→Verilog) — RISC-V
9. SERV — minimal RISC-V
10. DarkRISCV — RISC-V
11. mor1kx — OpenRISC
12. OpenSPARC T1/T2 — SPARC (older but large)
13. Amber — ARM-compatible
14. ZipCPU — custom CPU
15. BIRISCV — RISC-V

**Tier 3: Peripheral/IP blocks**
16. UART16550 (OpenCores)
17. I2C controller (OpenCores)
18. SPI Master (OpenCores)
19. Wishbone interconnect
20. AXI interconnect modules
21-30. Various OpenCores IP blocks (Ethernet MAC, USB, SDRAM controller, etc.)

**Tier 4: Additional variety**
31-50. Curated from GitHub trending Verilog/SystemVerilog repos, selected for having >500 LOC, test infrastructure, and meaningful commit history.

---

## 8. Benchmark Design for Cross-Artifact Reasoning

### Benchmark: KEDA-Bench

#### Structure:

```
KEDA-Bench/
├── designs/                    # Source repositories (or references)
│   ├── opentitan/
│   ├── ibex/
│   └── ...
├── knowledge_graphs/           # Pre-constructed KGs per design
│   ├── opentitan.graphml
│   └── ...
├── changes/                    # Controlled design changes
│   ├── change_001/
│   │   ├── description.json    # Change metadata
│   │   ├── patch.diff          # The actual change
│   │   └── ground_truth.json   # Known impacted artifacts
│   └── ...
├── questions/                  # Engineering questions
│   ├── question_001.json
│   │   ├── question_text
│   │   ├── answer
│   │   ├── evidence_nodes      # KG nodes needed to answer
│   │   ├── evidence_edges      # KG edges needed to answer
│   │   ├── difficulty          # easy/medium/hard
│   │   └── hops_required       # graph hops to answer
│   └── ...
└── splits/                     # Train/val/test splits by repository
    ├── train_repos.txt
    ├── val_repos.txt
    └── test_repos.txt
```

#### Benchmark dimensions:

1. **Change-Impact Queries (200-500 instances)**
   - Input: a diff/patch + the design's KG
   - Output: set of affected artifacts (modules, constraints, tests, assertions, timing paths, requirements)
   - Stratified by: change type, design size, impact breadth, number of hops

2. **Engineering Questions (300-500 instances)**
   - Input: natural language question + design context
   - Output: answer text + evidence set
   - Categories:
     - Structural queries ("Which modules instantiate X?") — 1-hop, easy
     - Dependency queries ("Which modules depend on clock Y?") — 2-hop, medium
     - Cross-artifact queries ("Which requirements are affected by this change?") — 3+ hops, hard
     - Risk assessment queries ("Why is module X risky to modify?") — multi-hop, hard

3. **Risk Propagation Scenarios (100-200 instances)**
   - Input: a design change + risk metric
   - Output: ranked list of affected components with risk scores
   - Ground truth: manually validated + tool-confirmed impact sets

#### Difficulty levels:
- **Easy (1-hop):** Direct relationships. E.g., "What ports does module X have?"
- **Medium (2-hop):** One level of indirection. E.g., "What clocks drive registers in submodules of X?"
- **Hard (3+ hops):** Multi-artifact traversal. E.g., "Which requirements might be violated if we change the width of port P on module X?"

---

## 9. Controlled Design Change Generation

### Change categories and generation methodology:

#### A. Parameter Modifications
```verilog
// Original
module fifo #(parameter DEPTH = 16, WIDTH = 8) (...);
// Modified
module fifo #(parameter DEPTH = 32, WIDTH = 8) (...);
```
**Generation:** For each parameterized module, systematically modify each parameter (double, halve, change to boundary values). Record the parameter, old value, new value.

#### B. Interface Width Changes
```verilog
// Original
input [7:0] data_in,
// Modified
input [15:0] data_in,
```
**Generation:** For each module port, generate width-change variants. This forces all connected modules to adapt.

#### C. Clock Changes
```verilog
// Original: module uses clk_core
always @(posedge clk_core)
// Modified: module uses clk_slow
always @(posedge clk_slow)
```
**Generation:** Swap clock assignments for registers, changing clock domains. This should trigger CDC and constraint impacts.

#### D. Reset Changes
```verilog
// Original: synchronous reset
always @(posedge clk) if (rst) q <= 0;
// Modified: asynchronous reset
always @(posedge clk or posedge rst) if (rst) q <= 0;
```
**Generation:** Toggle reset polarity, synchronous/asynchronous, or remove/add reset.

#### E. Module Dependency Changes
```verilog
// Original: instantiates module_a
module_a u_a (.clk(clk), .data(data));
// Modified: instantiates module_b (different interface)
module_b u_b (.clk(clk), .din(data), .valid(1'b1));
```
**Generation:** Swap instantiated modules with alternatives (e.g., different FIFO implementations).

#### F. Constraint Changes
```tcl
# Original
create_clock -period 10.0 -name clk_core [get_ports clk_core]
# Modified
create_clock -period 5.0 -name clk_core [get_ports clk_core]
```
**Generation:** Modify clock periods, add/remove false paths, change I/O delays.

#### G. Hierarchy Modifications
```verilog
// Original: flat instantiation
module_a u_a (...);
module_b u_b (...);
// Modified: wrapped in sub-hierarchy
wrapper u_wrap (
  .clk(clk), ...
);
// where wrapper contains module_a and module_b
```
**Generation:** Add/remove hierarchy levels, move modules between hierarchy levels.

### Change generation script:

```python
class ChangeGenerator:
    def __init__(self, design_graph, rtl_files):
        self.graph = design_graph
        self.rtl = rtl_files

    def generate_parameter_changes(self):
        changes = []
        for node, data in self.graph.nodes(data=True):
            if data['type'] == 'Parameter':
                original = data['default_value']
                for new_val in self.parameter_variants(original):
                    changes.append(ParameterChange(
                        module=data['module'],
                        parameter=node,
                        old_value=original,
                        new_value=new_val
                    ))
        return changes

    def generate_width_changes(self):
        changes = []
        for node, data in self.graph.nodes(data=True):
            if data['type'] == 'Port' and data.get('width', 1) > 1:
                original_width = data['width']
                for new_width in [original_width * 2, original_width // 2,
                                  original_width + 1]:
                    if new_width > 0:
                        changes.append(WidthChange(
                            port=node,
                            old_width=original_width,
                            new_width=new_width
                        ))
        return changes

    # ... similar for other change types
```

---

## 10. Ground-Truth Generation

### Multi-layer ground truth approach:

#### Layer 1: Structural ground truth (automated)
For each change, compute the ground truth by running the actual tools:

```python
def compute_structural_impact(change, design_graph):
    """Determine which modules are structurally affected."""
    modified_modules = change.get_modified_modules()
    affected = set(modified_modules)

    # BFS/DFS from modified modules
    for module in modified_modules:
        # Upstream: modules that instantiate the changed module
        affected |= nx.ancestors(design_graph, module)
        # Downstream: modules instantiated by the changed module
        affected |= nx.descendants(design_graph, module)

    return affected
```

#### Layer 2: Functional ground truth (tool-based validation)
```
For each change:
1. Apply the patch to a clean checkout
2. Run Yosys synthesis on both original and modified designs
3. Compare elaborated netlists → identify structural differences
4. Run linting (Verilator --lint-only) → identify new warnings/errors
5. Run testbenches → identify failing tests
6. If SDC exists: compare constraint coverage
```

#### Layer 3: Cross-artifact ground truth (manual + heuristic)
```python
def compute_cross_artifact_impact(change, full_graph):
    impact = {
        'modules': set(),      # directly and transitively affected modules
        'constraints': set(),  # SDC constraints on affected modules/clocks
        'assertions': set(),   # assertions verifying affected modules
        'tests': set(),        # tests covering affected modules
        'clocks': set(),       # clocks driving affected registers
        'timing_paths': set(), # timing paths through affected modules
        'requirements': set(), # requirements implemented by affected modules
    }

    # Start from structurally affected modules
    impact['modules'] = compute_structural_impact(change, full_graph)

    # Expand to constraints
    for module in impact['modules']:
        for _, target, data in full_graph.out_edges(module, data=True):
            if full_graph.nodes[target]['type'] == 'Port':
                for _, constraint, edata in full_graph.out_edges(target, data=True):
                    if edata.get('relation') == 'constrained_by':
                        impact['constraints'].add(constraint)

    # Expand to assertions
    for node, data in full_graph.nodes(data=True):
        if data['type'] == 'Assertion':
            for _, target, edata in full_graph.out_edges(node, data=True):
                if edata.get('relation') == 'verifies' and target in impact['modules']:
                    impact['assertions'].add(node)

    # ... similar expansion for tests, clocks, timing_paths, requirements

    return impact
```

#### Layer 4: Human validation (for evaluation subset)
- Select 50-100 changes across designs
- Have 2-3 hardware engineers annotate the full impact set
- Compute inter-annotator agreement (Cohen's κ)
- Use consensus annotations as gold standard for evaluation

#### Important: Ground truth completeness
- Structural impact (Layer 1) provides a lower bound on truly affected modules
- Tool-based validation (Layer 2) confirms functional impact
- Cross-artifact expansion (Layer 3) provides the full cross-artifact impact but may over-estimate
- Human validation (Layer 4) provides calibrated ground truth for a subset

For the full benchmark, use Layer 3 (heuristic expansion) as the primary ground truth, validated by Layer 4 on a subset. Report both "strict" (human-validated) and "heuristic" ground truth results.

---

## 11. Experimental Design

### Experiment A: Change-Impact Analysis

**Goal:** Evaluate whether the KG improves identification of affected artifacts when a design change is made.

**Setup:**
- 200-500 controlled design changes across 20+ repositories
- Stratified by change type (parameter/width/clock/reset/dependency/constraint/hierarchy)
- For each change, compute ground-truth impact set

**Methods compared:**

| Method | Description |
|--------|-------------|
| **Lexical** | grep/ripgrep for changed identifiers in all design files |
| **File-level** | All files changed in the same commit + files importing the changed module |
| **Embedding** | Vector similarity search: embed the change description, retrieve most similar design artifact descriptions |
| **LLM-only** | Provide the LLM with the diff and a flat list of all modules/artifacts; ask it to predict impact |
| **Graph traversal** | BFS/DFS from modified nodes in the KG (no LLM) |
| **KG + LLM (KEDA)** | Retrieve relevant subgraph from KG, provide to LLM for impact analysis |

**Metrics:**
- **Precision:** fraction of predicted-impacted artifacts that are truly impacted
- **Recall:** fraction of truly-impacted artifacts that are predicted
- **F1:** harmonic mean
- **Recall@K (K=5,10,20):** fraction of true impacts found in top K predictions
- **MRR:** mean reciprocal rank of first true positive
- **NDCG@K:** if predictions are ranked by confidence

**Analysis:**
- Report metrics separately per artifact type (modules, constraints, assertions, tests, clocks, timing paths, requirements)
- Report per change type (parameter/width/clock etc.)
- Report per hop distance (1-hop impacts vs. 2-hop vs. 3+ hop)
- Statistical significance: paired bootstrap test or Wilcoxon signed-rank test between methods

---

### Experiment B: Engineering Question Answering

**Goal:** Evaluate KG-grounded QA versus alternative retrieval strategies for EDA engineering questions.

**Setup:**
- 300-500 engineering questions across designs
- Each question has: gold answer, evidence nodes/edges, difficulty rating, hops required
- Questions authored by researchers with hardware background, validated by a second annotator

**Methods compared:**

| Method | Description |
|--------|-------------|
| **LLM-only** | Provide full design file contents (or truncated) to LLM |
| **Vector RAG** | Embed design artifacts, retrieve top-K by similarity to question, provide to LLM |
| **KG retrieval** | Convert question to graph query, retrieve subgraph, return structured answer (no LLM) |
| **KG + LLM (KEDA-QA)** | Convert question to graph query (LLM-assisted), retrieve subgraph, LLM synthesizes answer |
| **KG + LLM + Vector** | Combine KG subgraph and vector-retrieved context for LLM |

**Metrics:**
- **Answer correctness:** exact match or LLM-as-judge with rubric (correct/partially correct/incorrect)
- **Evidence recall:** fraction of required evidence nodes/edges present in retrieved context
- **Evidence precision:** fraction of retrieved context that is relevant
- **Hallucination rate:** fraction of answers containing fabricated facts (judged by comparison to KG ground truth)
- **Human evaluation (subset):** 3-point scale for explanation quality on 50-100 questions

**Analysis:**
- Stratify by difficulty level (easy/medium/hard) and by hop count
- Hypothesis: KG methods should dominate on multi-hop questions while Vector RAG may suffice for single-hop
- Report latency (graph query time + LLM inference time) per method

---

### Experiment C: Risk Propagation

**Goal:** Evaluate whether multi-hop graph-based risk propagation identifies indirect impacts missed by simpler methods.

**Setup:**
- 100-200 design change scenarios
- For each: compute ground-truth set of ALL affected artifacts (including indirect, multi-hop)
- Separate the impact set into:
  - **Direct (1-hop):** artifacts immediately connected to changed module
  - **Indirect (2+ hops):** artifacts connected through intermediate nodes

**Methods compared:**

| Method | Description |
|--------|-------------|
| **File-level** | Only files directly changed |
| **1-hop graph** | Only directly connected artifacts |
| **Lexical search** | grep for related identifiers |
| **Full graph traversal** | BFS to depth D |
| **Weighted risk propagation (KEDA-Risk)** | Personalized PageRank or weighted BFS with decay |
| **LLM-only** | Ask LLM to predict indirect impacts |

**Key metric:** Recall on indirect (2+ hop) impacts
- This is the critical metric: can the system find impacts that are NOT directly connected to the change?

**Analysis:**
- Plot recall vs. graph depth (hops 1, 2, 3, 4, 5+)
- Show specific examples of indirect impacts found by graph traversal but missed by baselines
- Compute "indirect impact discovery rate" = fraction of 2+ hop impacts found

---

## 12. Risk Propagation Model

### Weighted Graph Risk Propagation

Let $G = (V, E, w)$ be the EDA knowledge graph with nodes $V$, directed edges $E$, and edge weights $w: E \rightarrow [0, 1]$.

Given a set of changed nodes $S \subset V$ (e.g., modified modules), we compute a risk score $r(v)$ for every node $v \in V$.

#### Model 1: Weighted BFS with Exponential Decay

$$r(v) = \max_{s \in S} \max_{P \in \text{paths}(s, v)} \prod_{e \in P} w(e) \cdot \alpha^{|P|}$$

where:
- $\text{paths}(s, v)$ = all simple paths from $s$ to $v$ up to maximum depth $D$
- $|P|$ = length (number of edges) of path $P$
- $\alpha \in (0, 1)$ = decay factor per hop (e.g., $\alpha = 0.7$)
- $w(e)$ = edge weight representing coupling strength

The risk score is the maximum product of edge weights along any path, attenuated by distance.

**Practical computation (modified Dijkstra in log-space):**

```python
import heapq
import math

def risk_propagation_bfs(graph, changed_nodes, alpha=0.7, max_depth=5):
    """Compute risk scores via weighted BFS with exponential decay."""
    risk = {v: 0.0 for v in graph.nodes()}
    for s in changed_nodes:
        risk[s] = 1.0

    # Priority queue: (-log_risk, node, depth)
    pq = [(-0.0, s, 0) for s in changed_nodes]  # risk=1.0 → -log=0
    visited = {}  # node → best risk seen

    while pq:
        neg_log_risk, node, depth = heapq.heappop(pq)
        current_risk = math.exp(-(-neg_log_risk))  # recover risk from log

        if node in visited and visited[node] >= current_risk:
            continue
        visited[node] = current_risk
        risk[node] = max(risk[node], current_risk)

        if depth >= max_depth:
            continue

        for _, neighbor, edge_data in graph.out_edges(node, data=True):
            w = edge_data.get('weight', 1.0)
            new_risk = current_risk * w * alpha
            if new_risk > risk.get(neighbor, 0):
                heapq.heappush(pq, (-math.log(max(new_risk, 1e-10)), neighbor, depth + 1))

    return risk
```

#### Model 2: Personalized PageRank

$$r = (1 - \beta) \cdot (I - \beta \cdot \hat{A})^{-1} \cdot p$$

where:
- $\hat{A}$ = column-normalized weighted adjacency matrix
- $\beta$ = damping factor (e.g., 0.85)
- $p$ = personalization vector ($p_v = 1/|S|$ if $v \in S$, else 0)

This provides a principled way to compute influence scores from changed nodes. NetworkX provides `nx.pagerank(G, personalization=p, alpha=beta)`.

#### Model 3: Heat Diffusion

$$r(t) = e^{-t \cdot L} \cdot r(0)$$

where $L$ is the graph Laplacian and $r(0)$ is the initial risk vector (1 for changed nodes, 0 otherwise). This models risk as "heat" diffusing through the graph.

**Recommendation:** Use Model 1 (Weighted BFS) as primary—it is the most interpretable and controllable. Use Model 2 (PPR) as a comparison to show robustness. Model 3 is included for completeness but may be unnecessarily complex.

---

## 13. Edge Weight Definitions

### Principled edge weight assignment:

| Edge Type | Weight Definition | Rationale |
|-----------|------------------|-----------|
| `instantiates` | $w = 1.0$ | Direct structural coupling; change always propagates |
| `has_port` / `has_net` | $w = 1.0$ | Intrinsic to module; always affected |
| `connected_to` | $w = 0.9$ | Signal connections propagate changes with high probability |
| `drives` / `driven_by` | $w = 0.9$ | Data dependency |
| `clocked_by` | $w = 0.8$ | Clock changes affect all registers, but register logic may be independent |
| `constrained_by` | $w = 0.7$ | Constraint may or may not be affected depending on change type |
| `implemented_by` | $w = 0.5$ | Requirement-module link is often imprecise |
| `verifies` | $w = 0.7$ | Assertion likely needs re-checking |
| `covers` | $w = 0.6$ | Test may need re-running |
| `modifies` | $w = 1.0$ | Direct change; always relevant |
| `depends_on` | $w = 0.8$ | Dependency propagates changes |
| `crosses_domain` | $w = 0.9$ | CDC changes are high-risk |
| `affects` (violation) | $w = 0.8$ | Violation directly relevant |

### Data-driven weight refinement (optional):
If sufficient training data exists, learn edge weights by optimizing:

$$\hat{w} = \arg\min_w \sum_{(c, I_c) \in \text{train}} \mathcal{L}(\text{rank}(r_w), I_c)$$

where $I_c$ is the ground-truth impact set for change $c$, $r_w$ is the risk vector computed with weights $w$, and $\mathcal{L}$ is a ranking loss (e.g., pairwise or listwise).

### Change-type-dependent weights:
Different change types may warrant different edge weights:

| Change Type | Modified Edges |
|------------|----------------|
| Clock change | Boost `clocked_by` to 1.0, `constrained_by` to 0.9 |
| Width change | Boost `connected_to` to 1.0, `drives`/`driven_by` to 1.0 |
| Parameter change | Lower `connected_to` to 0.7 (may be localized) |

---

## 14. Ablation Studies

### Systematic ablations to isolate contributions:

| Ablation | What is Removed | Hypothesis |
|----------|----------------|------------|
| **A1: No constraints** | Remove all Constraint nodes and edges | Recall on timing-related impacts drops significantly |
| **A2: No requirements** | Remove all Requirement nodes and edges | Recall on requirement-level impacts drops; module-level may be unaffected |
| **A3: No verification** | Remove Assertion and Test nodes | Cannot identify affected tests/assertions; module-level impact unchanged |
| **A4: No Git history** | Remove Commit/PR nodes and edges | Cannot link changes to historical patterns; impact analysis unaffected for structural queries |
| **A5: Depth limit** | Limit graph traversal to depth 1, 2, 3, 4, 5 | Performance degrades as depth decreases; identify optimal depth |
| **A6: No LLM** | Use only graph traversal, no LLM reasoning | QA performance drops (cannot generate NL answers); impact analysis may be comparable |
| **A7: No KG (vector only)** | Replace KG with vector embeddings of all artifacts | Multi-hop reasoning degrades; single-hop may be comparable |
| **A8: No edge weights** | Set all edge weights to 1.0 | Risk propagation becomes less precise; more false positives |
| **A9: Flat graph** | Remove hierarchy (no Instance nodes, no child_of) | Cannot distinguish instances; over-estimates impact |
| **A10: No cross-artifact edges** | Keep only intra-RTL edges | Reduces to standard RTL dependency graph; cross-artifact reasoning impossible |

### Ablation analysis:
- Report full metric set for each ablation
- Compute contribution of each artifact type: $\Delta F1 = F1_{\text{full}} - F1_{\text{ablated}}$
- Rank artifact types by contribution
- Identify which artifact types provide diminishing returns

---

## 15. Statistical Significance Tests

### Recommended tests:

**1. Paired Bootstrap Test (primary)**
For each metric comparison (method A vs. method B):
```python
def paired_bootstrap_test(scores_a, scores_b, n_bootstrap=10000):
    """Test whether method A significantly outperforms method B."""
    observed_diff = np.mean(scores_a) - np.mean(scores_b)
    n = len(scores_a)
    count_ge = 0
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        boot_diff = np.mean(scores_a[indices]) - np.mean(scores_b[indices])
        if boot_diff <= 0:
            count_ge += 1
    p_value = count_ge / n_bootstrap
    return observed_diff, p_value
```
Report p-values and note significance at α = 0.05 with Bonferroni correction for multiple comparisons.

**2. Wilcoxon Signed-Rank Test**
Non-parametric alternative, appropriate when score distributions are non-normal.

**3. McNemar's Test**
For binary outcomes (e.g., "was the correct impact artifact found: yes/no"). Appropriate for comparing two methods on the same instances.

**4. Confidence Intervals**
Report 95% bootstrap confidence intervals for all metrics:
```python
def bootstrap_ci(scores, n_bootstrap=10000, alpha=0.05):
    means = []
    for _ in range(n_bootstrap):
        boot = np.random.choice(scores, size=len(scores), replace=True)
        means.append(np.mean(boot))
    return np.percentile(means, [100*alpha/2, 100*(1-alpha/2)])
```

**5. Effect Size**
Report Cohen's d for primary comparisons to quantify practical significance beyond statistical significance.

### Multiple comparisons correction:
With 5-6 methods and multiple metrics, apply Holm-Bonferroni correction to control family-wise error rate.

---

## 16. Data Leakage and Benchmark Design Problems

### Identified risks and mitigations:

| Risk | Description | Mitigation |
|------|-------------|-----------|
| **Train/test contamination** | If LLM has seen the open-source repos during pre-training, it may "know" the answers | (1) Use the KG as primary evaluation artifact, not raw code recall. (2) Create novel changes that never existed in the repo history. (3) Report LLM-only baseline to quantify this effect. |
| **Ground truth circularity** | If the KG is used both to generate ground truth AND as the retrieval source | Compute ground truth using independent methods (tool-based synthesis diffs, manual annotation). Never use the same graph traversal for ground truth and prediction. |
| **Question leakage** | Questions authored with knowledge of the KG structure | Have question authors write questions from specification documents, not from the KG. Validate that questions are answerable from the original design artifacts. |
| **Overfitting to graph structure** | Methods may exploit graph-specific patterns that don't generalize | Split by repository (not by change within the same repo). Use leave-one-repo-out evaluation. |
| **Synthetic artifact bias** | Synthetically generated requirements/timing reports may be unrealistically easy to link | Report results separately on "real" (extracted) vs. "synthetic" artifacts. Clearly disclose which artifacts are synthetic. |
| **LLM-as-judge bias** | Using an LLM to evaluate LLM outputs | Use majority vote of 3 LLM-as-judge calls. Validate LLM-as-judge against human annotations on a subset (compute agreement κ). |
| **Annotation bias** | Human annotators may not identify all impacts | Use multiple annotators, compute inter-annotator agreement, use union of annotations as recall ceiling. |

### Critical design principle:
**The ground truth for a design change must be computed WITHOUT using the KG.** The KG is the system being evaluated, not the oracle. Ground truth comes from:
1. Structural analysis (synthesis tool diff)
2. Functional analysis (test execution)
3. Manual expert annotation

---

## 17. Dataset Splitting Strategy

### Split by repository (not by change or question):

```
Repositories (20-50 total)
├── Train repos (60%): ~12-30 repos
│   Used for: weight tuning, prompt development, pipeline debugging
├── Validation repos (20%): ~4-10 repos
│   Used for: hyperparameter selection (decay α, depth D, edge weights)
└── Test repos (20%): ~4-10 repos
    Used for: final evaluation, all reported results
```

### Stratified splitting:
Ensure each split contains:
- At least one large SoC-scale design (e.g., OpenTitan)
- At least one medium CPU core (e.g., Ibex, PicoRV32)
- Multiple small IP blocks
- Designs with and without SDC constraints
- Designs with and without SVA assertions

### Cross-validation (if dataset is small):
Use leave-one-repo-out cross-validation: train on N-1 repos, test on 1, rotate.

### Why NOT split by change within a repo:
- Changes within the same repo share the same graph structure
- A model could learn repo-specific patterns (e.g., naming conventions, module organization) rather than general cross-artifact reasoning
- Repository-level splits ensure the model must generalize to unseen design structures

### Additional precaution:
- Do not tune any hyperparameters on test repos
- Report results on a held-out test set that was not used for any development decisions
- Clearly state the split in the paper

---

## 18. GNNs vs. Simpler Graph Algorithms

### Position: Start simple, add complexity only if needed.

**Arguments against GNNs (for this work):**
1. **Interpretability:** Weighted BFS and PPR produce interpretable risk scores with clear path-based explanations. GNNs are black boxes.
2. **Training data:** GNNs require labeled training data. Our benchmark may have 200-500 labeled changes—sufficient for evaluation but marginal for training a GNN.
3. **Generalization:** GNNs trained on one design's graph may not generalize to unseen designs with different schemas. Simple graph algorithms are graph-agnostic.
4. **Engineering complexity:** GNN implementation adds significant engineering overhead (node feature engineering, GNN architecture selection, training pipeline) without guaranteed benefit.
5. **Baselines first:** The paper's contribution is the KG and cross-artifact reasoning framework. Adding a GNN is an orthogonal contribution that could dilute the message.

**When GNNs WOULD be justified:**
- If simple graph algorithms (BFS, PPR) perform poorly and there is evidence that local neighborhood aggregation would help
- If the task requires learning complex non-linear relationships between node features and impact
- If you can demonstrate GNN improvement on validation set and confirm on test set
- As a follow-up paper, not in the initial submission

**Recommendation for this paper:**
- Use weighted BFS (Model 1) and PPR (Model 2) as primary graph methods
- Include a GNN baseline (e.g., GCN, GAT, or GraphSAGE with a binary classification head: "is this node impacted?") as an **experimental comparison**, not as the proposed method
- Report whether GNN outperforms simpler methods
- If GNN wins, discuss why and propose it as future work
- If GNN loses or ties, argue for interpretability advantage of simpler methods

---

## 19. Example Knowledge Graph

### Small RTL subsystem: UART with FIFO

```
                              ┌─────────────┐
                              │ Requirement  │
                              │ REQ-001:     │
                              │ "UART shall  │
                              │ support 8N1" │
                              └──────┬───────┘
                                     │ implemented_by
                                     ▼
┌──────────┐   modifies    ┌─────────────────┐   instantiates   ┌──────────────┐
│ PR #42   │──────────────►│   uart_top       │────────────────►│  uart_rx      │
│ "Fix     │               │                 │                  │              │
│ baud gen"│               │                 │──instantiates──►│  uart_tx      │
└──────────┘               │                 │                  └──────┬───────┘
                           │                 │──instantiates──►│  baud_gen     │
                           └────────┬────────┘                  └──────┬───────┘
                                    │                                   │
                              has_port                          contains_register
                                    │                                   │
                                    ▼                                   ▼
                           ┌──────────────┐                    ┌──────────────┐
                           │ Port: clk    │                    │ Reg: counter  │
                           │ (input, 1b)  │                    │ (16-bit)     │
                           └──────┬───────┘                    └──────┬───────┘
                                  │                                    │
                          constrained_by                          clocked_by
                                  │                                    │
                                  ▼                                    ▼
                           ┌──────────────┐                    ┌──────────────┐
                           │ Constraint:  │                    │ Clock:       │
                           │ create_clock │                    │ clk_core     │
                           │ -period 20.0 │                    │ (50 MHz)     │
                           └──────┬───────┘                    └──────────────┘
                                  │
                            applies_to
                                  │
                                  ▼
                           ┌──────────────┐
                           │ TimingPath:  │
                           │ clk→counter  │
                           │ slack: 2.1ns │
                           └──────────────┘

Additional edges:
- Assertion "A_baud_rate" ──verifies──► baud_gen
- Assertion "A_baud_rate" ──verifies──► REQ-001
- Test "test_uart_loopback" ──covers──► uart_top
- Test "test_baud_gen" ──covers──► baud_gen
- uart_rx ──depends_on──► baud_gen
- uart_tx ──depends_on──► baud_gen
```

### As node/edge tables:

**Nodes:**
| ID | Type | Key Attributes |
|----|------|---------------|
| uart_top | Module | file: uart_top.v, LOC: 150 |
| uart_rx | Module | file: uart_rx.v, LOC: 85 |
| uart_tx | Module | file: uart_tx.v, LOC: 72 |
| baud_gen | Module | file: baud_gen.v, LOC: 40 |
| uart_top.clk | Port | direction: input, width: 1 |
| uart_top.rx_data | Port | direction: output, width: 8 |
| baud_gen.counter | Register | width: 16, reset: 0 |
| clk_core | Clock | frequency: 50MHz |
| C1 | Constraint | create_clock, period: 20.0ns |
| TP1 | TimingPath | startpoint: clk, endpoint: counter, slack: 2.1ns |
| REQ-001 | Requirement | "UART shall support 8N1 at configurable baud rate" |
| A_baud | Assertion | assert property (baud_tick period matches config) |
| T_loopback | Test | test_uart_loopback.v |
| T_baud | Test | test_baud_gen.v |
| PR42 | PullRequest | "Fix baud rate generator overflow" |

**Edges:**
| Source | Relation | Target |
|--------|----------|--------|
| uart_top | instantiates | uart_rx |
| uart_top | instantiates | uart_tx |
| uart_top | instantiates | baud_gen |
| uart_top | has_port | uart_top.clk |
| uart_top | has_port | uart_top.rx_data |
| baud_gen | contains_register | baud_gen.counter |
| baud_gen.counter | clocked_by | clk_core |
| uart_top.clk | constrained_by | C1 |
| C1 | applies_to | TP1 |
| REQ-001 | implemented_by | uart_top |
| REQ-001 | implemented_by | baud_gen |
| A_baud | verifies | baud_gen |
| A_baud | verifies | REQ-001 |
| T_loopback | covers | uart_top |
| T_baud | covers | baud_gen |
| uart_rx | depends_on | baud_gen |
| uart_tx | depends_on | baud_gen |
| PR42 | modifies | baud_gen |

### Trace example:
**Query:** "What is the risk impact of PR #42?"

**Trace:**
```
PR42 ──modifies──► baud_gen (risk: 1.0)
  baud_gen ◄──depends_on── uart_rx (risk: 0.8 × 0.7 = 0.56)
  baud_gen ◄──depends_on── uart_tx (risk: 0.56)
  baud_gen ◄──covers── T_baud (risk: 0.6 × 0.7 = 0.42)
  baud_gen ──contains_register──► counter (risk: 0.7)
    counter ──clocked_by──► clk_core (risk: 0.7 × 0.8 × 0.7 = 0.392)
  baud_gen ◄──verifies── A_baud (risk: 0.7 × 0.7 = 0.343)
    A_baud ──verifies──► REQ-001 (risk: 0.343 × 0.7 × 0.7 = 0.168)
  uart_top ──instantiates──► baud_gen → uart_top affected (risk: 0.7)
    uart_top.clk ──constrained_by──► C1 (risk: 0.7 × 0.7 × 0.7 = 0.343)
      C1 ──applies_to──► TP1 (risk: 0.343 × 0.7 × 0.7 = 0.168)
```

**Result:** PR42 indirectly impacts Timing Path TP1 through 4 hops: PR42 → baud_gen → (uart_top via instantiation) → clk port → constraint C1 → TP1.

---

## 20. Example Graph Queries

### Cypher queries (Neo4j):

**Q1: Which modules depend on clk_core?**
```cypher
MATCH (r:Register)-[:clocked_by]->(c:Clock {name: 'clk_core'}),
      (m:Module)-[:contains_register]->(r)
RETURN DISTINCT m.name AS module_name
```

**Q2: Which timing constraints affect paths through fifo_ctrl?**
```cypher
MATCH (m:Module {name: 'fifo_ctrl'})-[:has_port]->(p:Port),
      (p)-[:constrained_by]->(c:Constraint),
      (c)-[:applies_to]->(tp:TimingPath)
RETURN c.constraint_type, c.value, tp.startpoint, tp.endpoint, tp.slack
```

**Q3: Which tests should be rerun if uart_rx changes?**
```cypher
MATCH (m:Module {name: 'uart_rx'})<-[:covers]-(t:Test)
RETURN t.name
UNION
MATCH (m:Module {name: 'uart_rx'})<-[:depends_on*1..3]-(m2:Module)<-[:covers]-(t:Test)
RETURN t.name
```

**Q4: Which requirements are implemented by modules modified in PR #42?**
```cypher
MATCH (pr:PullRequest {pr_id: 42})-[:modifies]->(m:Module),
      (req:Requirement)-[:implemented_by]->(m)
RETURN req.req_id, req.text, m.name
```

**Q5: What downstream components are affected if data_in width changes on uart_rx?**
```cypher
MATCH (p:Port {name: 'data_in'})<-[:has_port]-(m:Module {name: 'uart_rx'}),
      (m)<-[:depends_on*1..5]-(downstream:Module)
RETURN downstream.name, length(shortestPath((m)-[:depends_on*]-(downstream))) AS distance
ORDER BY distance
```

**Q6: Which requirements may be affected by timing violation V1?**
```cypher
MATCH (v:Violation {id: 'V1'})-[:affects]->(tp:TimingPath),
      (tp)<-[:applies_to]-(c:Constraint)-[:applies_to]->(p:Port),
      (p)<-[:has_port]-(m:Module)<-[:implemented_by]-(req:Requirement)
RETURN DISTINCT req.req_id, req.text, m.name
```

**Q7: Why is modifying baud_gen considered risky?**
```cypher
// Count dependencies, assertions, tests, constraints
MATCH (m:Module {name: 'baud_gen'})
OPTIONAL MATCH (m)<-[:depends_on*1..3]-(dep:Module)
OPTIONAL MATCH (m)<-[:verifies]-(a:Assertion)
OPTIONAL MATCH (m)<-[:covers]-(t:Test)
OPTIONAL MATCH (m)-[:contains_register]->(r:Register)-[:clocked_by]->(c:Clock)
OPTIONAL MATCH (m)<-[:implemented_by]-(req:Requirement)
RETURN m.name,
       count(DISTINCT dep) AS dependent_modules,
       count(DISTINCT a) AS assertions,
       count(DISTINCT t) AS tests,
       count(DISTINCT c) AS clocks,
       count(DISTINCT req) AS requirements
```

### NetworkX equivalents (Python):

```python
# Q1: Modules depending on clk_core
def modules_clocked_by(G, clock_name):
    clock_node = clock_name
    registers = [src for src, tgt, d in G.in_edges(clock_node, data=True)
                 if d.get('relation') == 'clocked_by']
    modules = set()
    for reg in registers:
        for src, tgt, d in G.in_edges(reg, data=True):
            if d.get('relation') == 'contains_register':
                modules.add(src)
    return modules

# Q3: Tests to rerun
def tests_to_rerun(G, module_name, max_hops=3):
    # Find all modules that depend on the changed module (reverse direction)
    affected_modules = {module_name}
    frontier = {module_name}
    for _ in range(max_hops):
        next_frontier = set()
        for m in frontier:
            for src, tgt, d in G.in_edges(m, data=True):
                if d.get('relation') == 'depends_on' and src not in affected_modules:
                    next_frontier.add(src)
                    affected_modules.add(src)
        frontier = next_frontier

    # Find tests covering any affected module
    tests = set()
    for m in affected_modules:
        for src, tgt, d in G.in_edges(m, data=True):
            if d.get('relation') == 'covers':
                tests.add(src)
    return tests
```

---

## 21. Baselines

### Proposed baselines (10):

| # | Baseline | Description |
|---|----------|-------------|
| 1 | **Lexical Search** | grep/ripgrep for changed identifiers across all files. Count keyword matches as impact predictions. |
| 2 | **File-Level Dependency** | Track import/include dependencies at the file level. Affected = files that import/include changed files. |
| 3 | **Module-Level Dependency Graph** | Standard RTL module hierarchy (instantiation graph only). BFS from changed module. No cross-artifact edges. |
| 4 | **BM25 Retrieval** | Index all artifact descriptions with BM25. Retrieve top-K by similarity to change description. |
| 5 | **Dense Embedding Retrieval (Vector RAG)** | Embed all artifacts with a sentence transformer. Retrieve top-K by cosine similarity to change description. Provide to LLM. |
| 6 | **LLM-Only (Zero-Shot)** | Provide the LLM with the change diff and a flat list of all artifacts. Ask it to predict impact. |
| 7 | **LLM-Only (Few-Shot)** | Same as above but with 3-5 examples of change → impact in the prompt. |
| 8 | **Code-Change-Impact (Software-SE)** | Apply software engineering change-impact tools (e.g., call-graph analysis adapted to HDL). |
| 9 | **KG without LLM (Graph-Only)** | Full KEDA knowledge graph, but use only graph traversal (no LLM). Return all nodes within K hops. |
| 10 | **Random Baseline** | Randomly select K artifacts as "affected". Provides a floor for metrics. |

### For QA specifically, add:
| # | Baseline | Description |
|---|----------|-------------|
| 11 | **Closed-Book LLM** | LLM answers the question with no context (tests memorization). |
| 12 | **Full-Context LLM** | Provide entire design source code in context (tests long-context). |

---

## 22. Scalability Analysis

### Graph size estimates:

| Design | RTL LOC | Est. Modules | Est. Nodes | Est. Edges | Graph Memory |
|--------|---------|-------------|------------|------------|-------------|
| PicoRV32 | ~3K | ~5 | ~200 | ~500 | <1 MB |
| Ibex | ~15K | ~30 | ~2,000 | ~8,000 | ~5 MB |
| SERV | ~1K | ~3 | ~100 | ~250 | <1 MB |
| VexRiscv (generated) | ~20K | ~50 | ~3,000 | ~12,000 | ~8 MB |
| OpenTitan | ~200K+ | ~500+ | ~50,000+ | ~200,000+ | ~200 MB |
| Rocket Chip (generated) | ~100K+ | ~200+ | ~20,000 | ~80,000 | ~60 MB |

### Estimation formula:
- Nodes ≈ (modules × 30) + (constraints × 1.5) + (tests × 1) + (commits × 1)
  - Each module contributes: 1 module + ~10 ports + ~5 nets + ~10 registers + ~3 parameters ≈ 30 nodes
- Edges ≈ Nodes × 4 (empirical ratio for hardware designs)

### Query cost analysis:

| Query Type | Algorithm | Complexity | Time (50K nodes) | Time (200K nodes) |
|-----------|-----------|-----------|------------------|-------------------|
| 1-hop neighbors | Direct lookup | O(degree) | <1 ms | <1 ms |
| BFS depth D | BFS | O(V + E) worst case | ~10 ms | ~50 ms |
| PPR | Power iteration | O(k × (V + E)) | ~100 ms | ~500 ms |
| Shortest path | Dijkstra | O((V + E) log V) | ~50 ms | ~200 ms |
| Subgraph extraction | BFS + filtering | O(V + E) | ~20 ms | ~100 ms |

**Conclusion:** Graph operations are NOT the bottleneck. Even for OpenTitan-scale designs, all graph queries complete in under 1 second. The LLM inference is the bottleneck (~1-10 seconds per query).

### Neo4j vs. NetworkX:
- For ≤50K nodes: NetworkX in-memory is sufficient and simpler
- For >50K nodes or concurrent access: Neo4j provides indexing benefits
- Recommendation: Use NetworkX for research prototype, note Neo4j as production path

---

## 23. Threats to Validity

### Internal Validity

1. **Ground truth quality:** Cross-artifact ground truth is partially heuristic-generated. May contain false positives (over-estimation of impact) or false negatives (missed impacts). *Mitigation:* Human validation on a subset; report inter-annotator agreement; clearly label automated vs. human-validated ground truth.

2. **Synthetic artifacts:** Requirements, timing reports, and some constraints are synthetically generated and may not reflect real-world complexity. *Mitigation:* Clearly disclose; evaluate separately on real vs. synthetic artifacts; use OpenTitan (which has real constraints) as a calibration point.

3. **LLM variability:** LLM outputs are stochastic. Different runs may produce different results. *Mitigation:* Set temperature=0 for deterministic outputs; report variance across 3 runs for stochastic baselines.

4. **Controlled change representativeness:** Controlled changes may not reflect the distribution of real engineering changes. *Mitigation:* Additionally evaluate on real historical changes from Git history where ground truth can be inferred.

### External Validity

5. **Generalization across designs:** Results on 20-50 open-source designs may not generalize to proprietary industrial designs. *Mitigation:* Use diverse designs (CPUs, peripherals, accelerators, SoCs); discuss limitations.

6. **Generalization across languages:** Focus on Verilog/SystemVerilog. May not apply to VHDL, Chisel, or SpinalHDL designs without parser adaptation. *Mitigation:* Note as limitation; argue that the ontology is language-agnostic.

7. **Scale:** Open-source designs are typically smaller than industrial ASICs/SoCs (millions of lines). *Mitigation:* Include OpenTitan as the largest available open-source SoC; discuss scalability projections.

### Construct Validity

8. **Metric appropriateness:** Precision/recall for impact analysis assumes a binary "affected/not-affected" distinction. In reality, impact is a spectrum. *Mitigation:* Also report ranked metrics (NDCG, MRR) and risk-score correlation.

9. **QA evaluation:** LLM-as-judge for answer correctness may be biased or inconsistent. *Mitigation:* Validate against human judgments; report agreement metrics.

### Conclusion Validity

10. **Multiple comparisons:** Many methods, metrics, and ablations increase false discovery risk. *Mitigation:* Apply Holm-Bonferroni correction; pre-register primary hypotheses; clearly separate exploratory from confirmatory analysis.

---

## 24. Potential Reviewer Concerns and Mitigations

| Reviewer Concern | Risk Level | Mitigation |
|-----------------|------------|-----------|
| **"This is just a knowledge graph with no novel algorithm"** | HIGH | Emphasize: (1) the cross-artifact ontology is novel for EDA, (2) the benchmark is a contribution, (3) the experimental comparison demonstrates concrete utility. Position the KG as an enabler, not the sole contribution. |
| **"The ontology is hand-designed — why not learn the schema?"** | MEDIUM | Argue that EDA has well-defined artifact types making a hand-designed schema appropriate and more interpretable. Discuss learned schemas as future work. |
| **"Synthetic requirements/timing are not realistic"** | HIGH | Be transparent about what is synthetic. Use OpenTitan's real constraints/assertions as the primary evaluation point. Show results separately on real vs. synthetic artifacts. |
| **"What does this do that a synthesis tool's design hierarchy doesn't already do?"** | HIGH | Key differentiator: synthesis tools build module graphs but do NOT connect to constraints, requirements, verification, or Git history. Show specific multi-artifact queries that no single tool can answer. |
| **"Why not use GNNs?"** | MEDIUM | Include GNN as a baseline. Show that simpler methods are competitive and more interpretable. Argue that the contribution is the framework, not a specific GNN architecture. |
| **"Scale is too small"** | MEDIUM | Include OpenTitan (~200K LOC). Show scalability analysis. Argue that the approach is evaluated on the largest available open-source SoC. |
| **"The LLM is doing all the work — the KG adds nothing"** | HIGH | Ablation A7 (no KG) directly tests this. The experiment MUST show that KG + LLM > LLM-only, especially on multi-hop queries. If it doesn't, the paper fails. |
| **"No user study or industrial validation"** | MEDIUM | Acknowledge as limitation. Propose as future work. Argue that the automated benchmark provides reproducible evaluation. If possible, include a small expert survey (3-5 hardware engineers). |
| **"Concurrent work in LLM-for-EDA will subsume this"** | MEDIUM | Position clearly: this is about *structured cross-artifact reasoning*, not about using LLMs for RTL generation or bug detection. The KG is the differentiator. |

### Paper strength requirements:
1. The KG + LLM MUST outperform LLM-only on multi-hop queries (otherwise the KG is not useful)
2. Cross-artifact reasoning MUST be shown to find impacts that single-artifact analysis misses (otherwise the cross-artifact aspect is not useful)
3. The benchmark MUST be open-sourced (otherwise the evaluation is not reproducible)

---

## 25. Suggested Paper Titles

1. **KEDA: Knowledge-Graph-Enhanced Cross-Artifact Reasoning for Electronic Design Automation**

2. **Beyond Module Boundaries: A Knowledge Graph for Multi-Artifact Reasoning in Hardware Design**

3. **GraphEDA: Cross-Artifact Knowledge Graphs for Change-Impact Analysis and Risk Propagation in RTL Design**

4. **Connecting the Dots: Knowledge-Graph-Grounded LLM Reasoning Across EDA Design Artifacts**

5. **KEDA-Bench: A Benchmark for Cross-Artifact Reasoning in Electronic Design Automation**

6. **From RTL to Requirements: A Unified Knowledge Graph for Hardware Design Reasoning**

7. **Cross-Artifact Change Impact Analysis in Hardware Design via Knowledge Graphs and Large Language Models**

8. **Semiconductor Design as a Knowledge Graph: Enabling Multi-Hop Reasoning Across RTL, Constraints, Verification, and Requirements**

**Recommendation:** Title 1 or 3 for a systems/methods paper; Title 5 if the benchmark is the primary contribution.

---

## 26. Draft Abstract (198 words)

> Semiconductor design produces interconnected artifacts—RTL source code, timing constraints, clock-domain structures, assertions, verification tests, specification requirements, and version-control history—that are semantically coupled but analyzed independently by existing EDA tools. Engineering questions frequently require reasoning across multiple artifact types: determining which tests, constraints, and requirements are affected when a module changes, or identifying indirect timing risks introduced by a pull request. We present KEDA, a framework that constructs a unified, multi-artifact knowledge graph from open-source hardware designs and uses graph-grounded retrieval to augment LLM reasoning for cross-artifact EDA tasks. Our ontology captures 18 node types and 22 relation types spanning RTL structure, timing constraints, verification, requirements, and Git history. We evaluate KEDA on KEDA-Bench, a benchmark of controlled design changes and engineering questions across 25 open-source Verilog/SystemVerilog projects. In change-impact analysis, KEDA improves recall of cross-artifact impacts by [X]% over LLM-only and [Y]% over vector retrieval baselines. For engineering question answering, graph-grounded retrieval reduces hallucination by [Z]% compared to standard RAG. We further demonstrate that weighted graph-based risk propagation identifies indirect downstream impacts that file-level and lexical approaches systematically miss. Code and benchmark are publicly available.

---

## 27. Paper Outline

### 1. Introduction (1.5 pages)
- Motivating example: PR modifies baud rate generator → which constraints, tests, requirements, timing paths are affected?
- Problem: EDA artifacts are fragmented; tools analyze them independently; LLMs lack structured cross-artifact reasoning
- Key insight: a unified knowledge graph enables multi-hop cross-artifact reasoning
- Contributions:
  1. KEDA ontology and automated construction pipeline
  2. KEDA-Bench benchmark for cross-artifact EDA reasoning
  3. Experimental evaluation showing KG-grounded reasoning outperforms baselines
  4. Risk propagation model for indirect impact identification

### 2. Related Work (1.5 pages)
- 2.1 Knowledge Graphs for EDA and Semiconductor Design
  - IC manufacturing KGs, PDK KGs, analog design KGs
  - Gap: no cross-artifact KGs for digital design
- 2.2 LLMs for EDA
  - RTL generation, bug detection, specification understanding
  - Gap: no structured KG grounding for cross-artifact reasoning
- 2.3 Graph-Based Hardware Analysis
  - Circuit GNNs, netlist graphs, synthesis optimization
  - Gap: operate on single artifact type (netlist)
- 2.4 RAG and GraphRAG
  - Vector RAG, Microsoft GraphRAG, knowledge-grounded QA
  - Gap: not applied to hardware domain with domain-specific schemas
- 2.5 Change Impact Analysis in Software Engineering
  - Call-graph-based, dependency-based, traceability
  - Gap: software-focused; hardware has additional artifact types (SDC, CDC, timing)

### 3. EDA Knowledge Graph (2.5 pages)
- 3.1 Ontology Design
  - Node types (Table 1)
  - Edge types (Table 2)
  - Design principles: minimal but complete, grounded in EDA practice
- 3.2 Automated Construction Pipeline
  - RTL extraction (Yosys, Verible)
  - Constraint extraction (SDC parser)
  - Verification extraction (SVA/test parser)
  - Git history extraction
  - Requirement linking (LLM-assisted)
- 3.3 Graph Statistics
  - Size, density, diameter for each evaluated design

### 4. Cross-Artifact Reasoning (2 pages)
- 4.1 Change-Impact Analysis
  - Graph traversal algorithm
  - Integration with LLM for explanation
- 4.2 Engineering Question Answering (GraphRAG)
  - Question → graph query translation
  - Subgraph retrieval
  - LLM-based answer synthesis
- 4.3 Risk Propagation
  - Weighted BFS model (Section 12)
  - Edge weight definitions
  - Risk score interpretation

### 5. KEDA-Bench (1.5 pages)
- 5.1 Design Selection (25 repos, diversity criteria)
- 5.2 Controlled Change Generation (7 change types, ~300 changes)
- 5.3 Engineering Question Dataset (~400 questions)
- 5.4 Ground Truth Generation (automated + human-validated)
- 5.5 Dataset Splits (by repository)

### 6. Experimental Setup (1 page)
- 6.1 Baselines (10 methods)
- 6.2 Metrics
- 6.3 LLM Configuration (model, temperature, prompts)
- 6.4 Implementation Details

### 7. Results (2.5 pages)
- 7.1 Experiment A: Change-Impact Analysis
  - Table: method × metric
  - Breakdown by change type
  - Breakdown by hop distance
- 7.2 Experiment B: Engineering QA
  - Table: method × metric
  - Breakdown by difficulty
  - Hallucination analysis
- 7.3 Experiment C: Risk Propagation
  - Indirect impact discovery rate
  - Case studies of multi-hop risks found
- 7.4 Ablation Studies
  - Table: ablation × metric
  - Contribution ranking of artifact types

### 8. Discussion (1 page)
- When does the KG help most? (multi-hop, cross-artifact)
- When is it unnecessary? (single-artifact, simple structural queries)
- GNN comparison results
- Limitations of synthetic artifacts
- Computational costs

### 9. Threats to Validity (0.5 pages)
- Internal, external, construct validity (Section 23)

### 10. Conclusion (0.5 pages)
- Summary of findings
- Open challenges
- Future work: industrial validation, GNN exploration, automated ontology refinement

**Total: ~15 pages** (suitable for a full conference paper at DAC, ICCAD, ASP-DAC, or similar)

---

## 28. Recommended Venues

### Tier 1 (Primary targets):

| Venue | Type | Relevance | Deadline (typical) |
|-------|------|-----------|-------------------|
| **DAC** (Design Automation Conference) | Conference | Core EDA venue; ML-for-EDA track | November |
| **ICCAD** (International Conference on CAD) | Conference | Core EDA venue; AI/ML track | May |
| **ASP-DAC** (Asia and South Pacific DAC) | Conference | EDA venue, slightly lower bar than DAC/ICCAD | July |
| **DATE** (Design, Automation & Test in Europe) | Conference | EDA + embedded systems | September |

### Tier 2 (Strong alternatives):

| Venue | Type | Relevance |
|-------|------|-----------|
| **MLCAD** (Machine Learning for CAD) | Workshop/Conference | Directly targeted at ML-for-EDA |
| **LAD** (LLM-Aided Design) workshop | Workshop | New workshop on LLMs for hardware design |
| **AAAI / IJCAI** | Conference | AI venue; applied AI track (if framed as KG + LLM contribution) |
| **EMNLP / ACL** | Conference | NLP venue; if framed as domain-specific GraphRAG (harder sell) |
| **ICSE / FSE / ASE** | Conference | Software engineering; if framed as cross-artifact traceability (requires framing adaptation) |

### Tier 3 (Journals):

| Venue | Type |
|-------|------|
| **IEEE TCAD** | Journal; extended version of DAC/ICCAD paper |
| **ACM TODAES** | Journal; EDA focused |
| **IEEE Design & Test** | Magazine; shorter, more practical |

### Recommendation:
- **First submission: ICCAD or DAC** (core EDA audience, ML/AI track)
- **If rejected: ASP-DAC or DATE** (slightly broader scope)
- **Alternative framing: MLCAD workshop** (targeted audience, lower bar, good for getting feedback)
- **Extended journal: IEEE TCAD** (after conference publication)

---

## 29. Critical Novelty Assessment

### Is there enough novelty for publication?

**Honest assessment: Yes, with caveats.**

**What IS novel:**
1. No prior work constructs a unified cross-artifact KG spanning RTL structure + constraints + verification + requirements + Git history for digital hardware design
2. No prior benchmark exists for cross-artifact EDA reasoning
3. GraphRAG has not been applied to EDA with a domain-specific hardware ontology
4. Cross-artifact change-impact analysis in hardware (spanning code → constraints → timing → requirements) is not addressed by any existing tool or research

**What is NOT novel (and should not be claimed as such):**
1. Knowledge graphs in general
2. Graph-based hardware analysis (netlist/module graphs)
3. LLMs for EDA (rapidly growing field)
4. RAG in general
5. Change-impact analysis in software engineering

**The novelty lies in the intersection and the rigorous evaluation, not in any single component.**

### Weaknesses to address honestly:

1. **Engineering contribution vs. research contribution:** Constructing the KG is primarily an engineering effort. The research contribution must come from the evaluation and the demonstrated utility. A paper that only describes the ontology and pipeline without strong experiments would likely be rejected.

2. **The "obvious next step" critique:** Combining KGs + LLMs + EDA may seem like an obvious combination. The paper must demonstrate that this combination produces measurably better results than simpler approaches, and that the KG's cross-artifact structure is essential (not just nice to have).

3. **Scalability concerns:** Demonstrating only on small open-source designs weakens the contribution. Including OpenTitan is essential to show it works at realistic scale.

---

## 30. Literature-Oriented Analysis

### Prior Work: What Already Exists

#### Knowledge Graphs for EDA / Semiconductor

- **Semiconductor manufacturing KGs** (Samsung, TSMC research): Focus on manufacturing process, yield, defect analysis. NOT about design artifacts. Proprietary, not reproducible.
- **PDK knowledge graphs** (scattered papers): Capture process design kit information, device models, design rules. Single-artifact, not cross-design-artifact.
- **Analog circuit KGs** (e.g., work on topology synthesis): Graph representations of analog circuit topologies for automated synthesis. Different domain (analog vs. digital), different schema.
- **IC supply chain KGs**: Focus on supply chain, sourcing, counterfeit detection. Irrelevant to design reasoning.

**Gap KEDA fills:** No prior KG spans digital design artifacts across RTL, constraints, verification, and requirements.

#### RTL Graph Representations

- **Yosys/ABC netlists:** Standard gate-level graph representations used in synthesis optimization. Single artifact (netlist), no cross-artifact links.
- **FIRRTL graph:** Chisel's intermediate representation is a graph. Single artifact.
- **Circuit GNNs** (e.g., DeepGate, GRANNITE, GAMORA): Use graph neural networks on netlist graphs for timing prediction, power estimation, functional reasoning. Single artifact (netlist), focus on prediction tasks, not cross-artifact reasoning.
- **Hierarchical RTL representations:** Module hierarchy graphs exist in every synthesis tool. But these are pure RTL structure with no cross-artifact edges.

**Gap KEDA fills:** Extends RTL graphs with edges to constraints, verification, requirements, and version history.

#### LLMs for EDA

- **RTL generation:** ChipGPT, RTLCoder, VeriGen, AutoChip, RTLLM — generate Verilog from specifications.
- **Verification:** LLMs for assertion generation, test generation, property checking.
- **Bug detection:** LLM-based RTL bug detection and repair.
- **EDA tool scripting:** ChatEDA, EDA Copilot — natural language interfaces to EDA tools.
- **Specification understanding:** LLMs parsing datasheets, generating design intent.

**Gap KEDA fills:** None of these use a structured cross-artifact KG for grounded reasoning. They operate on raw text/code without cross-artifact structure.

#### RAG / GraphRAG

- **Microsoft GraphRAG (2024):** Builds a knowledge graph from text corpus for RAG. Generic, not EDA-specific. The graph is derived from text, not from structured EDA artifacts.
- **Various domain-specific GraphRAG:** Medical, legal, scientific literature. Not applied to hardware engineering.

**Gap KEDA fills:** Domain-specific GraphRAG with an ontology designed for EDA artifacts, using structured extraction (not NLP-from-text).

#### Cross-Artifact Traceability (Software Engineering)

- **Requirements-to-code traceability:** Large body of work in software engineering (ICSE, FSE, RE conferences). Uses IR, NLP, or link prediction to connect requirements to source code.
- **Change-impact analysis in SE:** Call-graph-based, dependency-based, co-change-based. Well-studied for Java/C++.
- **Software knowledge graphs:** KGs for software repositories, linking code, issues, tests, documentation.

**Gap KEDA fills:** Hardware design has additional artifact types with no software analog: SDC constraints, clock domains, CDC crossings, timing paths, synthesis results. The schema and extraction pipeline are fundamentally different.

### Summary Table: Prior Work vs. KEDA

| Dimension | Prior Work Status | KEDA Contribution |
|-----------|------------------|-------------------|
| KG spanning RTL + constraints + verification + requirements + Git | **Does not exist** | **Core contribution** |
| Automated KG construction from open-source EDA tooling | Not for cross-artifact KG | Extraction pipeline using Yosys, Verible, parsers |
| Cross-artifact change-impact analysis in hardware | Not addressed | Experiments A and C |
| GraphRAG for EDA QA | Not attempted | Experiment B |
| Benchmark for cross-artifact EDA reasoning | **Does not exist** | KEDA-Bench |
| Risk propagation across hardware artifact types | Not studied | Risk model + evaluation |

---

## 31. Recommended Minimal Strong Version (First Paper)

### Scope: Focus on what can be built and evaluated in ~3-4 months

#### Title:
**"Cross-Artifact Change-Impact Analysis in RTL Design via Knowledge Graphs"**

#### Reduce scope to:

1. **5-10 open-source Verilog projects** (not 50)
   - OpenTitan (large, rich artifacts)
   - Ibex (medium, well-documented)
   - PicoRV32 (small, clean)
   - CV32E40P (medium, verification suite)
   - 3-6 additional peripheral IP blocks

2. **Simplified ontology (10 node types, 12 edge types)**
   - Module, Instance, Port, Register, Clock, Constraint, Assertion, Test, Commit, Requirement
   - Drop: Net, FSM, CDCCrossing, ClockDomain, TimingPath, Violation, PullRequest, Parameter
   - These can be added in the extended journal version

3. **Two experiments only:**
   - **Experiment A: Change-Impact Analysis** (primary, 150-200 changes)
   - **Experiment C: Risk Propagation** (secondary, 50-100 scenarios, showing multi-hop advantage)
   - Drop Experiment B (QA) — it requires a large question dataset and LLM-as-judge evaluation, which is expensive and introduces noise

4. **Simpler baselines (5):**
   - Lexical search
   - Module dependency graph (RTL-only)
   - Embedding retrieval
   - LLM-only
   - KEDA (full KG traversal)

5. **Graph algorithms only — no LLM in the loop for the primary method:**
   - The primary KEDA method is weighted graph traversal
   - LLM is used only as a baseline and possibly for requirement linking during construction
   - This avoids LLM cost, variability, and evaluation complexity
   - LLM-augmented reasoning becomes a natural follow-up paper

6. **Controlled changes only (no real PRs for primary evaluation):**
   - Generate 150-200 controlled changes across the 5-10 designs
   - Automated ground truth (structural + cross-artifact expansion)
   - Human validation on 30-50 changes for calibration

7. **Core metrics: Precision, Recall, F1, Recall@K**
   - Stratified by hop distance
   - Stratified by change type
   - Paired bootstrap significance tests

#### What this version demonstrates:
- The cross-artifact KG can be constructed automatically
- Cross-artifact change-impact analysis improves recall over single-artifact methods
- Multi-hop graph traversal finds indirect impacts that simpler methods miss
- The approach works on realistic open-source designs including OpenTitan

#### Implementation timeline (estimated work items, NOT time predictions):
1. Build extraction pipeline (Yosys + Verible + SDC parser + SVA parser + GitPython)
2. Construct KGs for 5-10 designs
3. Generate controlled changes and ground truth
4. Implement baselines
5. Run experiments
6. Write paper

#### Lines of code estimate:
- Extraction pipeline: ~2,000-3,000 LOC Python
- Change generation: ~500-800 LOC
- Ground truth computation: ~500-800 LOC
- Baselines: ~500-1,000 LOC
- Evaluation: ~500 LOC
- **Total: ~4,000-6,000 LOC** — very feasible

#### Publication target: **ICCAD 2027 or DAC 2027** (depending on completion timeline)

---

## Appendix A: Repository Selection Criteria

Select repositories that satisfy at least 3 of:
1. ≥1,000 LOC Verilog/SystemVerilog
2. ≥50 Git commits with meaningful messages
3. At least one SDC or constraint file
4. At least one testbench or verification file
5. Module hierarchy depth ≥ 3
6. Active maintenance (commits within last 2 years)
7. Permissive license (Apache, MIT, BSD)

## Appendix B: LLM Prompt Templates

### For Graph Query Generation (Experiment B):
```
You are an EDA knowledge graph query assistant. Given the following question
about a hardware design, generate a graph traversal query.

Available node types: Module, Instance, Port, Register, Clock, Constraint,
Assertion, Test, TimingPath, Requirement, Commit, PullRequest

Available edge types: instantiates, has_port, connected_to, clocked_by,
constrained_by, implemented_by, verifies, covers, modifies, depends_on, affects

Question: {question}
Design: {design_name}

Generate a structured query specifying:
1. Start node(s)
2. Edge traversal pattern
3. Target node type(s)
4. Filters (if any)
```

### For Impact Explanation (KEDA full pipeline):
```
You are an EDA impact analysis assistant. Given a design change and the
affected subgraph from a knowledge graph, explain the impact.

Change: {change_description}
Affected subgraph:
{subgraph_nodes_and_edges}

For each affected artifact, explain:
1. What is affected
2. How it is connected to the change (path through the graph)
3. The risk level (high/medium/low)
4. Recommended action
```

## Appendix C: Reproducibility Checklist

- [ ] All source code publicly available (GitHub)
- [ ] KEDA-Bench dataset publicly available
- [ ] Extraction pipeline runnable with open-source tools only
- [ ] All LLM prompts documented
- [ ] Random seeds fixed and reported
- [ ] Hardware/compute requirements documented
- [ ] Evaluation scripts included
- [ ] Ground truth generation scripts included
- [ ] Statistical test implementations included
- [ ] Requirements.txt / environment.yml provided
