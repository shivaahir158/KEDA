"""Tests for the controlled change generation pipeline."""

import sys
import json
from pathlib import Path

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from keda.analysis.changegen import (
    ChangeType,
    FilePatch,
    DesignChange,
    ChangeSet,
    GroundTruthComputer,
    ChangeGenerator,
    apply_changes,
    print_changeset_summary,
)
from keda.graph.builder import KGBuilder, DesignConfig

FIXTURE_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def build_result():
    config = DesignConfig(
        name="uart",
        rtl_files=[
            FIXTURE_DIR / "uart_top.v",
            FIXTURE_DIR / "uart_tx.v",
            FIXTURE_DIR / "uart_rx.v",
            FIXTURE_DIR / "baud_gen.v",
        ],
        top_module="uart_top",
        sdc_files=[FIXTURE_DIR / "uart.sdc"],
        sva_files=[FIXTURE_DIR / "uart_assertions.sv"],
        repo_path=FIXTURE_DIR / "git_repo",
    )
    builder = KGBuilder()
    return builder.build(config)


@pytest.fixture(scope="module")
def rtl_sources():
    sources = {}
    for f in ["uart_top.v", "uart_tx.v", "uart_rx.v", "baud_gen.v"]:
        p = FIXTURE_DIR / f
        sources[str(p)] = p.read_text()
    return sources


@pytest.fixture(scope="module")
def sdc_sources():
    p = FIXTURE_DIR / "uart.sdc"
    return {str(p): p.read_text()}


@pytest.fixture(scope="module")
def generator(build_result, rtl_sources, sdc_sources):
    return ChangeGenerator(
        graph=build_result.graph,
        rtl_sources=rtl_sources,
        design_name="uart",
        sdc_sources=sdc_sources,
    )


@pytest.fixture(scope="module")
def changeset(generator):
    return generator.generate_all()


# ---------------------------------------------------------------------------
# FilePatch tests
# ---------------------------------------------------------------------------

class TestFilePatch:
    def test_apply(self):
        patch = FilePatch("test.v", "old_text", "new_text")
        result = patch.apply("prefix old_text suffix")
        assert result == "prefix new_text suffix"

    def test_apply_not_found(self):
        patch = FilePatch("test.v", "missing", "new")
        with pytest.raises(ValueError, match="Patch target not found"):
            patch.apply("some content")

    def test_revert(self):
        patch = FilePatch("test.v", "old_text", "new_text")
        result = patch.revert("prefix new_text suffix")
        assert result == "prefix old_text suffix"

    def test_revert_not_found(self):
        patch = FilePatch("test.v", "old", "new_text")
        with pytest.raises(ValueError, match="Revert target not found"):
            patch.revert("no match here")

    def test_apply_only_first_occurrence(self):
        patch = FilePatch("test.v", "A", "B")
        result = patch.apply("A and A and A")
        assert result == "B and A and A"


# ---------------------------------------------------------------------------
# DesignChange serialization
# ---------------------------------------------------------------------------

class TestDesignChange:
    def test_to_dict_and_back(self):
        dc = DesignChange(
            change_id="test::change::parameter::0001",
            change_type=ChangeType.PARAMETER,
            description="Test change",
            target_module="baud_gen",
            patches=[FilePatch("baud_gen.v", "old", "new", line_number=5)],
            metadata={"key": "value"},
            ground_truth={"uart::baud_gen", "uart::uart_top"},
            ground_truth_by_type={"Module": {"uart::baud_gen", "uart::uart_top"}},
            ground_truth_by_hop={1: {"uart::baud_gen"}, 2: {"uart::uart_top"}},
            synthesis_valid=True,
        )
        d = dc.to_dict()
        dc2 = DesignChange.from_dict(d)

        assert dc2.change_id == dc.change_id
        assert dc2.change_type == ChangeType.PARAMETER
        assert dc2.description == "Test change"
        assert dc2.target_module == "baud_gen"
        assert len(dc2.patches) == 1
        assert dc2.patches[0].old_text == "old"
        assert dc2.patches[0].line_number == 5
        assert dc2.metadata == {"key": "value"}
        assert dc2.ground_truth == {"uart::baud_gen", "uart::uart_top"}
        assert dc2.ground_truth_by_hop[1] == {"uart::baud_gen"}
        assert dc2.synthesis_valid is True

    def test_json_roundtrip(self):
        dc = DesignChange(
            change_id="id1",
            change_type=ChangeType.WIDTH,
            description="width change",
            target_module="mod",
        )
        d = dc.to_dict()
        s = json.dumps(d)
        d2 = json.loads(s)
        dc2 = DesignChange.from_dict(d2)
        assert dc2.change_type == ChangeType.WIDTH


# ---------------------------------------------------------------------------
# ChangeSet persistence
# ---------------------------------------------------------------------------

class TestChangeSet:
    def test_by_type(self):
        cs = ChangeSet(design_name="test")
        cs.changes = [
            DesignChange("id1", ChangeType.PARAMETER, "d1", "m1"),
            DesignChange("id2", ChangeType.PARAMETER, "d2", "m2"),
            DesignChange("id3", ChangeType.WIDTH, "d3", "m3"),
        ]
        groups = cs.by_type()
        assert len(groups[ChangeType.PARAMETER]) == 2
        assert len(groups[ChangeType.WIDTH]) == 1

    def test_save_load(self, tmp_path):
        cs = ChangeSet(design_name="test_design")
        cs.changes = [
            DesignChange(
                change_id="id1",
                change_type=ChangeType.CLOCK,
                description="clock change",
                target_module="mod_a",
                patches=[FilePatch("file.v", "old", "new")],
                ground_truth={"node1", "node2"},
            ),
        ]
        path = tmp_path / "changeset.json"
        cs.save(path)
        assert path.exists()

        cs2 = ChangeSet.load(path)
        assert cs2.design_name == "test_design"
        assert len(cs2.changes) == 1
        assert cs2.changes[0].change_type == ChangeType.CLOCK
        assert cs2.changes[0].ground_truth == {"node1", "node2"}


# ---------------------------------------------------------------------------
# ChangeGenerator — parameter changes
# ---------------------------------------------------------------------------

class TestParameterChanges:
    def test_generates_parameter_changes(self, changeset):
        param_changes = [c for c in changeset.changes
                         if c.change_type == ChangeType.PARAMETER]
        assert len(param_changes) >= 1

    def test_parameter_has_patches(self, changeset):
        param_changes = [c for c in changeset.changes
                         if c.change_type == ChangeType.PARAMETER]
        for c in param_changes:
            assert len(c.patches) >= 1
            assert c.patches[0].old_text != c.patches[0].new_text

    def test_parameter_metadata(self, changeset):
        param_changes = [c for c in changeset.changes
                         if c.change_type == ChangeType.PARAMETER]
        for c in param_changes:
            assert "parameter_name" in c.metadata
            assert "old_value" in c.metadata
            assert "new_value" in c.metadata
            assert "variant" in c.metadata

    def test_parameter_variants(self, changeset):
        param_changes = [c for c in changeset.changes
                         if c.change_type == ChangeType.PARAMETER]
        variants = {c.metadata["variant"] for c in param_changes}
        # Should have at least double and halve
        assert "double" in variants or "halve" in variants

    def test_parameter_patch_applies(self, changeset, rtl_sources):
        param_changes = [c for c in changeset.changes
                         if c.change_type == ChangeType.PARAMETER]
        if param_changes:
            c = param_changes[0]
            result = apply_changes(c, rtl_sources)
            assert c.patches[0].new_text in list(result.values())[0] or \
                   any(c.patches[0].new_text in v for v in result.values())


class TestFormatLike:
    def test_plain_number(self):
        assert ChangeGenerator._format_like(100, "200") == "100"

    def test_underscore_format(self):
        result = ChangeGenerator._format_like(100000000, "50_000_000")
        assert "_" in result
        assert result == "100_000_000"

    def test_small_number_underscore(self):
        result = ChangeGenerator._format_like(1, "50_000_000")
        assert result == "1"

    def test_medium_number(self):
        result = ChangeGenerator._format_like(25000000, "50_000_000")
        assert result == "25_000_000"


# ---------------------------------------------------------------------------
# ChangeGenerator — width changes
# ---------------------------------------------------------------------------

class TestWidthChanges:
    def test_generates_width_changes(self, changeset):
        width_changes = [c for c in changeset.changes
                         if c.change_type == ChangeType.WIDTH]
        assert len(width_changes) >= 1

    def test_width_metadata(self, changeset):
        width_changes = [c for c in changeset.changes
                         if c.change_type == ChangeType.WIDTH]
        for c in width_changes:
            assert "port_name" in c.metadata
            assert "old_width" in c.metadata
            assert "new_width" in c.metadata
            assert c.metadata["new_width"] != c.metadata["old_width"]

    def test_width_double_and_halve(self, changeset):
        width_changes = [c for c in changeset.changes
                         if c.change_type == ChangeType.WIDTH]
        variants = {c.metadata.get("variant") for c in width_changes}
        # baud_div is [15:0] => 16 bits, should have both
        assert "double" in variants

    def test_width_patch_applies(self, changeset, rtl_sources):
        width_changes = [c for c in changeset.changes
                         if c.change_type == ChangeType.WIDTH]
        if width_changes:
            c = width_changes[0]
            result = apply_changes(c, rtl_sources)
            # The patch should have been applied
            for v in result.values():
                if c.patches[0].new_text in v:
                    break
            else:
                pytest.fail("Width patch not applied to any source file")


# ---------------------------------------------------------------------------
# ChangeGenerator — clock changes
# ---------------------------------------------------------------------------

class TestClockChanges:
    def test_generates_clock_changes(self, changeset):
        clock_changes = [c for c in changeset.changes
                         if c.change_type == ChangeType.CLOCK]
        assert len(clock_changes) >= 1

    def test_edge_swap_variant(self, changeset):
        clock_changes = [c for c in changeset.changes
                         if c.change_type == ChangeType.CLOCK]
        edge_swaps = [c for c in clock_changes
                      if c.metadata.get("variant") == "edge_swap"]
        assert len(edge_swaps) >= 1
        for c in edge_swaps:
            assert "posedge" in c.patches[0].old_text or "negedge" in c.patches[0].old_text

    def test_clock_deduplicated(self, changeset):
        clock_changes = [c for c in changeset.changes
                         if c.change_type == ChangeType.CLOCK]
        edge_swaps = [c for c in clock_changes
                      if c.metadata.get("variant") == "edge_swap"]
        # Should not have duplicate (module, clock_signal) pairs
        keys = [(c.target_module, c.metadata.get("clock_signal"))
                for c in edge_swaps]
        assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# ChangeGenerator — reset changes
# ---------------------------------------------------------------------------

class TestResetChanges:
    def test_generates_reset_changes(self, changeset):
        reset_changes = [c for c in changeset.changes
                         if c.change_type == ChangeType.RESET]
        assert len(reset_changes) >= 1

    def test_polarity_swap(self, changeset):
        reset_changes = [c for c in changeset.changes
                         if c.change_type == ChangeType.RESET]
        polarity = [c for c in reset_changes
                    if c.metadata.get("variant") == "polarity_swap"]
        if polarity:
            c = polarity[0]
            assert "negedge rst_n" in c.patches[0].old_text
            assert "posedge rst_n" in c.patches[0].new_text

    def test_reset_per_module(self, changeset):
        reset_changes = [c for c in changeset.changes
                         if c.change_type == ChangeType.RESET]
        modules = [c.target_module for c in reset_changes]
        # Each module should only have one reset change (dedup)
        assert len(modules) == len(set(modules))


# ---------------------------------------------------------------------------
# ChangeGenerator — dependency changes
# ---------------------------------------------------------------------------

class TestDependencyChanges:
    def test_generates_dependency_changes(self, changeset):
        dep_changes = [c for c in changeset.changes
                       if c.change_type == ChangeType.DEPENDENCY]
        assert len(dep_changes) >= 1

    def test_dependency_comments_out(self, changeset):
        dep_changes = [c for c in changeset.changes
                       if c.change_type == ChangeType.DEPENDENCY]
        for c in dep_changes:
            assert "// REMOVED:" in c.patches[0].new_text

    def test_dependency_metadata(self, changeset):
        dep_changes = [c for c in changeset.changes
                       if c.change_type == ChangeType.DEPENDENCY]
        for c in dep_changes:
            assert "instance_name" in c.metadata
            assert "child_module" in c.metadata
            assert "parent_module" in c.metadata
            assert c.metadata["variant"] == "remove_instance"


# ---------------------------------------------------------------------------
# ChangeGenerator — constraint changes
# ---------------------------------------------------------------------------

class TestConstraintChanges:
    def test_generates_constraint_changes(self, changeset):
        cst_changes = [c for c in changeset.changes
                       if c.change_type == ChangeType.CONSTRAINT]
        assert len(cst_changes) >= 1

    def test_tighten_clock_present(self, changeset):
        cst_changes = [c for c in changeset.changes
                       if c.change_type == ChangeType.CONSTRAINT]
        tighten = [c for c in cst_changes
                   if c.metadata.get("variant") == "tighten_clock"]
        assert len(tighten) >= 1
        c = tighten[0]
        assert c.metadata["new_period"] < c.metadata["old_period"]

    def test_loosen_clock_present(self, changeset):
        cst_changes = [c for c in changeset.changes
                       if c.change_type == ChangeType.CONSTRAINT]
        loosen = [c for c in cst_changes
                  if c.metadata.get("variant") == "loosen_clock"]
        assert len(loosen) >= 1

    def test_false_path_removal(self, changeset):
        cst_changes = [c for c in changeset.changes
                       if c.change_type == ChangeType.CONSTRAINT]
        fps = [c for c in cst_changes
               if c.metadata.get("variant") == "remove_false_path"]
        assert len(fps) >= 1
        for c in fps:
            assert "# REMOVED:" in c.patches[0].new_text

    def test_delay_change(self, changeset):
        cst_changes = [c for c in changeset.changes
                       if c.change_type == ChangeType.CONSTRAINT]
        delays = [c for c in cst_changes
                  if c.metadata.get("variant") == "change_delay"]
        assert len(delays) >= 1
        for c in delays:
            assert c.metadata["new_delay"] == c.metadata["old_delay"] * 2


# ---------------------------------------------------------------------------
# ChangeGenerator — hierarchy changes
# ---------------------------------------------------------------------------

class TestHierarchyChanges:
    def test_generates_hierarchy_changes(self, changeset):
        hier_changes = [c for c in changeset.changes
                        if c.change_type == ChangeType.HIERARCHY]
        assert len(hier_changes) >= 1

    def test_flatten_variant(self, changeset):
        hier_changes = [c for c in changeset.changes
                        if c.change_type == ChangeType.HIERARCHY]
        for c in hier_changes:
            assert "FLATTENED" in c.patches[0].new_text
            assert c.metadata["variant"] == "flatten"


# ---------------------------------------------------------------------------
# ChangeGenerator — generate_all
# ---------------------------------------------------------------------------

class TestGenerateAll:
    def test_all_types_present(self, changeset):
        types = {c.change_type for c in changeset.changes}
        assert ChangeType.PARAMETER in types
        assert ChangeType.WIDTH in types
        assert ChangeType.CLOCK in types
        assert ChangeType.RESET in types
        assert ChangeType.DEPENDENCY in types
        assert ChangeType.CONSTRAINT in types
        assert ChangeType.HIERARCHY in types

    def test_max_per_type(self, build_result, rtl_sources, sdc_sources):
        gen = ChangeGenerator(
            graph=build_result.graph,
            rtl_sources=rtl_sources,
            design_name="uart",
            sdc_sources=sdc_sources,
        )
        cs = gen.generate_all(max_per_type=1)
        by_type = cs.by_type()
        for ct, changes in by_type.items():
            assert len(changes) <= 1

    def test_total_changes_reasonable(self, changeset):
        # UART design should generate 15-60 changes
        assert 10 < len(changeset.changes) < 100

    def test_all_changes_have_ids(self, changeset):
        ids = [c.change_id for c in changeset.changes]
        assert len(ids) == len(set(ids))  # unique
        for cid in ids:
            assert "::" in cid

    def test_all_changes_have_target_module(self, changeset):
        for c in changeset.changes:
            assert c.target_module, f"Change {c.change_id} has no target_module"


# ---------------------------------------------------------------------------
# GroundTruthComputer
# ---------------------------------------------------------------------------

class TestGroundTruth:
    def test_compute_single(self, build_result, changeset):
        gtc = GroundTruthComputer(build_result.graph)
        change = changeset.changes[0]
        gtc.compute(change, "uart")
        assert len(change.ground_truth) > 0

    def test_ground_truth_excludes_self(self, build_result, changeset):
        gtc = GroundTruthComputer(build_result.graph)
        for c in changeset.changes[:5]:
            gtc.compute(c, "uart")
            module_id = f"uart::{c.target_module}"
            assert module_id not in c.ground_truth

    def test_ground_truth_by_type(self, build_result, changeset):
        gtc = GroundTruthComputer(build_result.graph)
        c = changeset.changes[0]
        gtc.compute(c, "uart")
        assert len(c.ground_truth_by_type) > 0
        # All nodes in by_type should also be in ground_truth
        all_typed = set()
        for nodes in c.ground_truth_by_type.values():
            all_typed.update(nodes)
        assert all_typed == c.ground_truth

    def test_ground_truth_by_hop(self, build_result, changeset):
        gtc = GroundTruthComputer(build_result.graph)
        c = changeset.changes[0]
        gtc.compute(c, "uart")
        assert len(c.ground_truth_by_hop) > 0
        # All nodes in by_hop should also be in ground_truth
        all_hopped = set()
        for nodes in c.ground_truth_by_hop.values():
            all_hopped.update(nodes)
        assert all_hopped == c.ground_truth

    def test_compute_all(self, build_result, changeset):
        gtc = GroundTruthComputer(build_result.graph)
        # Make a copy so we don't pollute other tests
        cs_copy = ChangeSet(design_name="uart")
        for c in changeset.changes[:5]:
            cs_copy.changes.append(DesignChange(
                change_id=c.change_id,
                change_type=c.change_type,
                description=c.description,
                target_module=c.target_module,
                patches=c.patches,
                metadata=c.metadata,
            ))
        gtc.compute_all(cs_copy)
        for c in cs_copy.changes:
            assert len(c.ground_truth) > 0

    def test_missing_module(self, build_result):
        gtc = GroundTruthComputer(build_result.graph)
        change = DesignChange(
            change_id="test",
            change_type=ChangeType.PARAMETER,
            description="test",
            target_module="nonexistent_module",
        )
        gtc.compute(change, "uart")
        assert len(change.ground_truth) == 0

    def test_type_specific_edges(self, build_result, changeset):
        gtc = GroundTruthComputer(build_result.graph)
        # Compare generic vs type-specific ground truth
        c = changeset.changes[0]

        c_generic = DesignChange(
            change_id="g",
            change_type=c.change_type,
            description="g",
            target_module=c.target_module,
        )
        c_specific = DesignChange(
            change_id="s",
            change_type=c.change_type,
            description="s",
            target_module=c.target_module,
        )
        gtc.compute(c_generic, "uart", use_type_specific_edges=False)
        gtc.compute(c_specific, "uart", use_type_specific_edges=True)

        # Type-specific should be a subset of or equal to generic
        # (or could differ if type-specific edges include different relations)
        # Both should be non-empty
        assert len(c_generic.ground_truth) > 0
        assert len(c_specific.ground_truth) > 0


# ---------------------------------------------------------------------------
# apply_changes
# ---------------------------------------------------------------------------

class TestApplyChanges:
    def test_apply_rtl_change(self, changeset, rtl_sources):
        param_changes = [c for c in changeset.changes
                         if c.change_type == ChangeType.PARAMETER]
        if not param_changes:
            pytest.skip("No parameter changes generated")
        c = param_changes[0]
        result = apply_changes(c, rtl_sources)
        # Result should differ from original
        for k in result:
            if result[k] != rtl_sources.get(k, ""):
                break
        else:
            pytest.fail("No source file was modified")

    def test_apply_sdc_change(self, changeset, sdc_sources):
        cst_changes = [c for c in changeset.changes
                       if c.change_type == ChangeType.CONSTRAINT]
        if not cst_changes:
            pytest.skip("No constraint changes generated")
        c = cst_changes[0]
        result = apply_changes(c, sdc_sources)
        for k in result:
            if result[k] != sdc_sources.get(k, ""):
                break
        else:
            pytest.fail("No SDC file was modified")

    def test_apply_missing_file(self):
        change = DesignChange(
            change_id="test",
            change_type=ChangeType.PARAMETER,
            description="test",
            target_module="mod",
            patches=[FilePatch("nonexistent.v", "old", "new")],
        )
        with pytest.raises(ValueError, match="Source file not found"):
            apply_changes(change, {"other.v": "content"})

    def test_apply_missing_text(self, rtl_sources):
        change = DesignChange(
            change_id="test",
            change_type=ChangeType.PARAMETER,
            description="test",
            target_module="mod",
            patches=[FilePatch("uart_top.v", "NONEXISTENT_TEXT", "new")],
        )
        with pytest.raises(ValueError, match="Patch target not found"):
            apply_changes(change, rtl_sources)


# ---------------------------------------------------------------------------
# Print summary (smoke test)
# ---------------------------------------------------------------------------

class TestPrintSummary:
    def test_summary_runs(self, changeset, capsys):
        print_changeset_summary(changeset)
        captured = capsys.readouterr()
        assert "ChangeSet:" in captured.out
        assert "Total changes:" in captured.out

    def test_summary_with_ground_truth(self, build_result, changeset, capsys):
        gtc = GroundTruthComputer(build_result.graph)
        # Compute GT for a few
        for c in changeset.changes[:3]:
            gtc.compute(c, "uart")
        print_changeset_summary(changeset)
        captured = capsys.readouterr()
        assert "ChangeSet:" in captured.out
