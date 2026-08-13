"""Tests for the Git history extractor."""

import sys
from pathlib import Path

import networkx as nx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from keda.extractors.git_extractor import GitExtractor, print_git_summary

FIXTURE_DIR = Path(__file__).parent / "fixtures"
GIT_REPO = FIXTURE_DIR / "git_repo"


@pytest.fixture(scope="module")
def git_result():
    extractor = GitExtractor()
    return extractor.extract(
        repo_path=GIT_REPO,
        design_name="uart",
        max_commits=100,
    )


class TestGitExtraction:
    def test_repo_found(self, git_result):
        assert git_result.repo_path
        assert git_result.branch == "master"

    def test_commits_extracted(self, git_result):
        assert git_result.total_commits == 4
        assert len(git_result.commits) == 4

    def test_commit_metadata(self, git_result):
        for c in git_result.commits:
            assert c.sha
            assert len(c.sha) == 40
            assert c.short_sha == c.sha[:8]
            assert c.author
            assert c.date
            assert c.summary

    def test_commit_messages(self, git_result):
        summaries = {c.summary for c in git_result.commits}
        assert "Initial commit: add UART design" in summaries
        assert "Fix: change default baud rate to 9600" in summaries
        assert "Feature: widen data path to 16 bits" in summaries
        assert "Constraint: tighten clock period to 10ns" in summaries

    def test_files_changed(self, git_result):
        # The baud rate fix should only change baud_gen.v
        baud_commit = [c for c in git_result.commits
                       if "baud rate" in c.summary.lower()][0]
        assert "baud_gen.v" in baud_commit.files_changed

        # The width change should change uart_tx.v and uart_rx.v
        width_commit = [c for c in git_result.commits
                        if "widen" in c.summary.lower()][0]
        assert "uart_tx.v" in width_commit.files_changed
        assert "uart_rx.v" in width_commit.files_changed

    def test_modules_inferred(self, git_result):
        baud_commit = [c for c in git_result.commits
                       if "baud rate" in c.summary.lower()][0]
        assert "baud_gen" in baud_commit.modules_modified

    def test_change_type_classification(self, git_result):
        baud_commit = [c for c in git_result.commits
                       if "baud rate" in c.summary.lower()][0]
        assert baud_commit.change_type == "fix"

        width_commit = [c for c in git_result.commits
                        if "widen" in c.summary.lower()][0]
        assert width_commit.change_type == "feature"

        sdc_commit = [c for c in git_result.commits
                      if "clock period" in c.summary.lower()][0]
        assert sdc_commit.change_type == "constraint"

    def test_file_histories(self, git_result):
        assert len(git_result.file_histories) > 0
        # baud_gen.v should have 2 commits (initial + fix)
        baud_fh = git_result.file_histories.get("baud_gen.v")
        assert baud_fh is not None
        assert baud_fh.commit_count == 2

    def test_file_history_authors(self, git_result):
        for fh in git_result.file_histories.values():
            assert len(fh.authors) >= 1

    def test_diff_stats(self, git_result):
        # At least some commits should have nonzero insertions
        has_stats = any(
            c.insertions > 0 or c.deletions > 0
            for c in git_result.commits
        )
        assert has_stats


class TestGitGraphIntegration:
    def test_add_to_graph_with_modules(self, git_result):
        G = nx.DiGraph()
        G.add_node("uart::baud_gen", type="Module", name="baud_gen")
        G.add_node("uart::uart_tx", type="Module", name="uart_tx")
        G.add_node("uart::uart_rx", type="Module", name="uart_rx")
        G.add_node("uart::uart_top", type="Module", name="uart_top")

        extractor = GitExtractor()
        extractor.add_to_graph(git_result, G, design_name="uart")

        # Commit nodes should be added
        commit_nodes = [
            n for n, d in G.nodes(data=True) if d.get("type") == "Commit"
        ]
        assert len(commit_nodes) == 4

        # modifies edges
        modifies_edges = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("relation") == "modifies"
        ]
        assert len(modifies_edges) >= 3

        # modified_by edges (reverse)
        modified_by = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("relation") == "modified_by"
        ]
        assert len(modified_by) >= 3

    def test_module_change_frequency(self, git_result):
        G = nx.DiGraph()
        G.add_node("uart::baud_gen", type="Module", name="baud_gen")
        G.add_node("uart::uart_tx", type="Module", name="uart_tx")

        extractor = GitExtractor()
        extractor.add_to_graph(git_result, G, design_name="uart")

        # baud_gen modified in 2 commits (initial + fix)
        assert G.nodes["uart::baud_gen"].get("change_frequency") == 2
        # uart_tx modified in 2 commits (initial + width)
        assert G.nodes["uart::uart_tx"].get("change_frequency") == 2

    def test_module_file_map(self):
        G = nx.DiGraph()
        G.add_node("uart::baud_gen", type="Module", name="baud_gen",
                    src_file="baud_gen.v")
        G.add_node("uart::uart_top", type="Module", name="uart_top",
                    src_file="uart_top.v")

        extractor = GitExtractor()
        mapping = extractor.get_module_file_map(G)
        assert mapping["baud_gen"] == "baud_gen.v"
        assert mapping["uart_top"] == "uart_top.v"


class TestPrintSummary:
    def test_summary(self, git_result, capsys):
        print_git_summary(git_result)
        captured = capsys.readouterr()
        assert "Total commits processed:" in captured.out
        assert "Commit types:" in captured.out
        assert "Most changed files:" in captured.out
