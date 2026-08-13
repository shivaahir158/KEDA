"""
SVA (SystemVerilog Assertion) and testbench extractor for the KEDA knowledge graph.

Extracts:
- SVA assertions (assert property, assume property, cover property)
- Inline assertions within RTL modules
- Immediate assertions (assert, assume)
- Testbench files and test names
- Module instantiations in testbenches (to determine coverage)

Produces Assertion and Test nodes with edges to Modules and Requirements.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class SVAAssertion:
    """A single parsed SVA assertion."""
    assertion_id: str
    name: str | None
    assertion_type: str          # "assert", "assume", "cover"
    property_type: str           # "concurrent" (property), "immediate", "deferred"
    containing_module: str | None
    file_path: str
    line_number: int
    end_line: int | None
    property_text: str
    clock_signal: str | None = None
    disable_condition: str | None = None
    error_message: str | None = None


@dataclass
class TestbenchInfo:
    """A parsed testbench / test."""
    test_id: str
    name: str
    file_path: str
    modules_instantiated: list[str] = field(default_factory=list)
    is_testbench: bool = True


@dataclass
class SVAExtractionResult:
    """All assertions and tests extracted from design files."""
    assertions: list[SVAAssertion] = field(default_factory=list)
    tests: list[TestbenchInfo] = field(default_factory=list)
    files_scanned: list[str] = field(default_factory=list)


# Regex for named concurrent assertions:
#   label: assert/assume/cover property (...)
_NAMED_CONCURRENT_RE = re.compile(
    r'^\s*(\w+)\s*:\s*(assert|assume|cover)\s+property\s*\(',
    re.MULTILINE,
)

# Regex for unnamed concurrent assertions:
#   assert/assume/cover property (...)
_UNNAMED_CONCURRENT_RE = re.compile(
    r'^\s*(assert|assume|cover)\s+property\s*\(',
    re.MULTILINE,
)

# Regex for immediate assertions:
#   assert(...) or assert #0 (...)
_IMMEDIATE_RE = re.compile(
    r'^\s*(\w+\s*:\s*)?(assert|assume)\s*(?:#\d+\s*)?\(',
    re.MULTILINE,
)

# Regex for module declaration
_MODULE_RE = re.compile(
    r'^\s*module\s+(\w+)',
    re.MULTILINE,
)

# Regex for endmodule
_ENDMODULE_RE = re.compile(
    r'^\s*endmodule',
    re.MULTILINE,
)

# Regex for clock in disable iff / @(posedge ...)
_CLOCK_RE = re.compile(
    r'@\s*\(\s*(posedge|negedge)\s+(\w+)',
)

# Regex for disable iff
_DISABLE_RE = re.compile(
    r'disable\s+iff\s*\(\s*([^)]+)\s*\)',
)

# Regex for module instantiation (non-parameterized form)
# For parameterized forms, _find_instantiations does a second pass.
_INSTANCE_RE = re.compile(
    r'^\s*(\w+)\s+(\w+)\s*\(',
    re.MULTILINE,
)
# Regex for parameterized module instantiation: module_type #(
_PARAM_INST_START_RE = re.compile(
    r'^\s*(\w+)\s+#\s*\(',
    re.MULTILINE,
)

# Keywords that are NOT module instantiation names
_VERILOG_KEYWORDS = frozenset({
    "module", "endmodule", "input", "output", "inout", "wire", "reg", "logic",
    "assign", "always", "always_ff", "always_comb", "always_latch",
    "initial", "begin", "end", "if", "else", "case", "endcase",
    "for", "while", "do", "repeat", "forever", "generate", "endgenerate",
    "function", "endfunction", "task", "endtask", "parameter", "localparam",
    "integer", "real", "time", "genvar", "typedef", "struct", "enum",
    "interface", "endinterface", "class", "endclass", "package", "endpackage",
    "import", "export", "virtual", "static", "automatic", "const",
    "assert", "assume", "cover", "property", "endproperty",
    "sequence", "endsequence", "checker", "endchecker",
    "program", "endprogram", "clocking", "endclocking",
    "default", "disable", "iff", "posedge", "negedge",
    "supply0", "supply1", "tri", "triand", "trior", "tri0", "tri1",
    "pullup", "pulldown", "buf", "not", "and", "or", "nand", "nor",
    "xor", "xnor", "void", "bit", "byte", "shortint", "int", "longint",
    "string", "chandle", "event",
})

# Testbench naming heuristics
_TB_PATTERNS = re.compile(
    r'(^tb_|_tb$|^test_|_test$|^testbench|_testbench$|^bench_|_bench$|_sim$|^sim_)',
    re.IGNORECASE,
)


class SVAExtractor:
    """Extract SVA assertions and testbench information from Verilog/SystemVerilog files.

    Usage:
        extractor = SVAExtractor()
        result = extractor.extract(["src/uart_top.v", "tb/test_uart.sv"])
        extractor.add_to_graph(result, graph, design_name="uart")
    """

    def extract(
        self,
        files: list[str | Path],
        design_name: str = "design",
        known_modules: set[str] | None = None,
    ) -> SVAExtractionResult:
        """Scan files for SVA assertions and testbench information.

        Args:
            files: Verilog/SystemVerilog files to scan.
            design_name: Design name prefix for IDs.
            known_modules: Set of known module names (from RTL extraction)
                          to validate instantiation detection.
        """
        result = SVAExtractionResult()
        assertion_counter = 0

        for file_path in files:
            path = Path(file_path)
            if not path.exists():
                logger.warning("File not found: %s", path)
                continue

            result.files_scanned.append(str(path))

            try:
                text = path.read_text(errors="replace")
            except OSError as e:
                logger.warning("Could not read %s: %s", path, e)
                continue

            lines = text.splitlines()

            # Build a map of line_offset -> line_number
            # and determine which module each line belongs to
            module_regions = self._find_module_regions(text, lines)

            # Extract concurrent assertions
            for m in _NAMED_CONCURRENT_RE.finditer(text):
                assertion_counter += 1
                line_num = text[:m.start()].count('\n') + 1
                name = m.group(1)
                atype = m.group(2)
                prop_text, end_line = self._extract_property_text(lines, line_num - 1)
                containing = self._find_containing_module(line_num, module_regions)

                clock = None
                disable = None
                cm = _CLOCK_RE.search(prop_text)
                if cm:
                    clock = cm.group(2)
                dm = _DISABLE_RE.search(prop_text)
                if dm:
                    disable = dm.group(1).strip()

                error_msg = None
                em = re.search(r'\$error\s*\(\s*"([^"]*)"', prop_text)
                if em:
                    error_msg = em.group(1)

                result.assertions.append(SVAAssertion(
                    assertion_id=f"{design_name}::sva::{assertion_counter:04d}",
                    name=name,
                    assertion_type=atype,
                    property_type="concurrent",
                    containing_module=containing,
                    file_path=str(path),
                    line_number=line_num,
                    end_line=end_line,
                    property_text=prop_text,
                    clock_signal=clock,
                    disable_condition=disable,
                    error_message=error_msg,
                ))

            # Extract unnamed concurrent assertions
            for m in _UNNAMED_CONCURRENT_RE.finditer(text):
                # Skip if this position was already captured by a named assertion
                line_num = text[:m.start()].count('\n') + 1
                if any(a.line_number == line_num and a.file_path == str(path)
                       for a in result.assertions):
                    continue

                assertion_counter += 1
                atype = m.group(1)
                prop_text, end_line = self._extract_property_text(lines, line_num - 1)
                containing = self._find_containing_module(line_num, module_regions)

                clock = None
                cm = _CLOCK_RE.search(prop_text)
                if cm:
                    clock = cm.group(2)

                error_msg = None
                em = re.search(r'\$error\s*\(\s*"([^"]*)"', prop_text)
                if em:
                    error_msg = em.group(1)

                result.assertions.append(SVAAssertion(
                    assertion_id=f"{design_name}::sva::{assertion_counter:04d}",
                    name=None,
                    assertion_type=atype,
                    property_type="concurrent",
                    containing_module=containing,
                    file_path=str(path),
                    line_number=line_num,
                    end_line=end_line,
                    property_text=prop_text,
                    clock_signal=clock,
                    error_message=error_msg,
                ))

            # Detect testbenches and their instantiations
            file_stem = path.stem
            is_tb = bool(_TB_PATTERNS.search(file_stem))

            # Also check for `initial` blocks or `$finish` which suggest testbench
            if not is_tb:
                is_tb = bool(re.search(r'\binitial\b', text)) and bool(
                    re.search(r'\$finish|\$stop|\$fatal|\$display', text)
                )

            for mod_name, (start, end) in module_regions.items():
                if is_tb or bool(_TB_PATTERNS.search(mod_name)):
                    # This is a testbench module — find what it instantiates
                    mod_text = "\n".join(lines[start - 1:end])
                    instantiated = self._find_instantiations(
                        mod_text, known_modules or set()
                    )

                    result.tests.append(TestbenchInfo(
                        test_id=f"{design_name}::test::{mod_name}",
                        name=mod_name,
                        file_path=str(path),
                        modules_instantiated=instantiated,
                    ))

        return result

    def _find_module_regions(
        self, text: str, lines: list[str]
    ) -> dict[str, tuple[int, int]]:
        """Map module names to (start_line, end_line) in the file."""
        regions: dict[str, tuple[int, int]] = {}
        current_module: str | None = None
        current_start: int = 0

        for i, line in enumerate(lines, 1):
            m = _MODULE_RE.match(line)
            if m:
                current_module = m.group(1)
                current_start = i
            if _ENDMODULE_RE.match(line) and current_module:
                regions[current_module] = (current_start, i)
                current_module = None

        return regions

    def _find_containing_module(
        self, line_num: int, module_regions: dict[str, tuple[int, int]]
    ) -> str | None:
        """Find which module contains the given line number."""
        for mod_name, (start, end) in module_regions.items():
            if start <= line_num <= end:
                return mod_name
        return None

    def _extract_property_text(
        self, lines: list[str], start_idx: int
    ) -> tuple[str, int | None]:
        """Extract the full assertion text, handling multi-line assertions.

        Returns (text, end_line_number).
        """
        collected = []
        paren_depth = 0
        found_open = False
        end_line = start_idx + 1

        for i in range(start_idx, min(start_idx + 30, len(lines))):
            line = lines[i]
            collected.append(line.strip())
            for ch in line:
                if ch == '(':
                    paren_depth += 1
                    found_open = True
                elif ch == ')':
                    paren_depth -= 1

            # Check for semicolon after balanced parens
            if found_open and paren_depth <= 0 and ';' in line:
                end_line = i + 1
                break

        return " ".join(collected), end_line

    def _find_instantiations(
        self, text: str, known_modules: set[str]
    ) -> list[str]:
        """Find module instantiations in a block of text.

        Uses known_modules if available, otherwise relies on heuristics.
        """
        instantiated = []

        def _check_mod(mod_type: str):
            if mod_type in _VERILOG_KEYWORDS or mod_type.startswith("$"):
                return
            if known_modules:
                if mod_type in known_modules:
                    instantiated.append(mod_type)
            else:
                if not mod_type.startswith(("reg", "wire", "logic", "bit")):
                    instantiated.append(mod_type)

        # Non-parameterized: "module_type inst_name ("
        for m in _INSTANCE_RE.finditer(text):
            _check_mod(m.group(1))

        # Parameterized: "module_type #(" — scan for balanced parens then inst name
        for m in _PARAM_INST_START_RE.finditer(text):
            _check_mod(m.group(1))

        return list(set(instantiated))

    def add_to_graph(
        self,
        result: SVAExtractionResult,
        graph: nx.DiGraph,
        design_name: str = "design",
    ):
        """Add assertions and tests to an existing knowledge graph.

        Links assertions to their containing modules and clocks.
        Links tests to modules they instantiate (covers relation).
        """
        # Add assertion nodes
        for assertion in result.assertions:
            aid = assertion.assertion_id
            graph.add_node(aid, **{
                "type": "Assertion",
                "name": assertion.name,
                "assertion_type": assertion.assertion_type,
                "property_type": assertion.property_type,
                "file_path": assertion.file_path,
                "line_number": assertion.line_number,
                "end_line": assertion.end_line,
                "property_text": assertion.property_text,
                "clock_signal": assertion.clock_signal,
                "disable_condition": assertion.disable_condition,
                "error_message": assertion.error_message,
            })

            # Link to containing module
            if assertion.containing_module:
                mod_id = f"{design_name}::{assertion.containing_module}"
                if graph.has_node(mod_id):
                    graph.add_edge(aid, mod_id, relation="verifies")
                    graph.add_edge(mod_id, aid, relation="verified_by")
                else:
                    logger.debug(
                        "Assertion %s references module '%s' not in graph",
                        aid, assertion.containing_module,
                    )

            # Link to clock
            if assertion.clock_signal:
                clk_id = f"{design_name}::clk::{assertion.clock_signal}"
                if graph.has_node(clk_id):
                    graph.add_edge(aid, clk_id, relation="uses_clock")

        # Add test nodes
        for test in result.tests:
            tid = test.test_id
            graph.add_node(tid, **{
                "type": "Test",
                "name": test.name,
                "file_path": test.file_path,
                "is_testbench": test.is_testbench,
            })

            # Link to covered modules
            for mod_name in test.modules_instantiated:
                mod_id = f"{design_name}::{mod_name}"
                if graph.has_node(mod_id):
                    graph.add_edge(tid, mod_id, relation="covers")
                else:
                    logger.debug(
                        "Test %s instantiates module '%s' not in graph",
                        tid, mod_name,
                    )


def print_sva_summary(result: SVAExtractionResult):
    """Print a summary of extracted SVA assertions and tests."""
    print(f"  Files scanned: {len(result.files_scanned)}")
    print(f"  Assertions found: {len(result.assertions)}")
    print(f"  Tests found: {len(result.tests)}")

    type_counts: dict[str, int] = {}
    for a in result.assertions:
        type_counts[a.assertion_type] = type_counts.get(a.assertion_type, 0) + 1
    if type_counts:
        print("  Assertion types:")
        for t, c in sorted(type_counts.items()):
            print(f"    {t}: {c}")

    if result.tests:
        print("  Tests:")
        for t in result.tests:
            print(f"    {t.name} -> covers {t.modules_instantiated}")
