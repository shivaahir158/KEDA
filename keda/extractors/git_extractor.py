"""
Git history extractor for the KEDA knowledge graph.

Extracts:
- Commits (with metadata, changed files, and linked modules)
- File change history
- Per-module change frequency and recency
- Optionally: pull requests via GitHub API (not implemented here)

Produces Commit nodes and edges linking them to Modules.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

try:
    import git as gitpython
    from git import Repo, Commit as GitCommit
except ImportError:
    raise ImportError("GitPython is required: pip install GitPython")

logger = logging.getLogger(__name__)


@dataclass
class CommitInfo:
    """A single parsed Git commit."""
    commit_id: str          # KEDA graph node ID
    sha: str
    short_sha: str
    author: str
    author_email: str
    date: datetime
    message: str
    summary: str            # first line
    files_changed: list[str] = field(default_factory=list)
    insertions: int = 0
    deletions: int = 0
    modules_modified: list[str] = field(default_factory=list)
    change_type: str | None = None  # inferred: fix, feature, refactor, etc.


@dataclass
class FileHistory:
    """Change history for a single file."""
    file_path: str
    commit_count: int = 0
    last_modified: datetime | None = None
    authors: set[str] = field(default_factory=set)
    commits: list[str] = field(default_factory=list)  # list of SHAs


@dataclass
class GitExtractionResult:
    """All data extracted from Git history."""
    commits: list[CommitInfo] = field(default_factory=list)
    file_histories: dict[str, FileHistory] = field(default_factory=dict)
    repo_path: str = ""
    total_commits: int = 0
    branch: str = ""


# Patterns for classifying commit messages
_CHANGE_TYPE_PATTERNS = [
    (r'\bfix(?:es|ed)?\b|\bbug\b|\bpatch\b|\bhotfix\b', "fix"),
    (r'\bfeat(?:ure)?\b|\badd(?:ed|s)?\b|\bnew\b|\bimplement', "feature"),
    (r'\brefactor\b|\bclean\b|\brestructur', "refactor"),
    (r'\btest\b|\bverif', "test"),
    (r'\bdoc\b|\bcomment\b|\breadme\b', "docs"),
    (r'\bconstraint\b|\bsdc\b|\btiming\b', "constraint"),
    (r'\bsynth\b|\bsynthes', "synthesis"),
    (r'\brevert\b', "revert"),
    (r'\bmerge\b', "merge"),
]

# File extensions that are RTL source
_RTL_EXTENSIONS = frozenset({".v", ".sv", ".svh", ".vh", ".vhd", ".vhdl"})
_CONSTRAINT_EXTENSIONS = frozenset({".sdc", ".xdc", ".tcl"})
_TEST_PATTERNS = re.compile(r'(^tb_|_tb\.|^test_|_test\.|testbench|_sim\.)', re.IGNORECASE)


class GitExtractor:
    """Extract version history from a Git repository.

    Usage:
        extractor = GitExtractor()
        result = extractor.extract(
            repo_path="/path/to/repo",
            design_name="uart",
            max_commits=500,
        )
        extractor.add_to_graph(result, graph, design_name="uart")
    """

    def extract(
        self,
        repo_path: str | Path,
        design_name: str = "design",
        max_commits: int = 500,
        branch: str | None = None,
        file_filter: str | None = None,
        module_file_map: dict[str, str] | None = None,
    ) -> GitExtractionResult:
        """Extract commit history and file change data.

        Args:
            repo_path: Path to the Git repository.
            design_name: Design name prefix for node IDs.
            max_commits: Maximum number of commits to process.
            branch: Branch to analyze (default: current HEAD).
            file_filter: Only include commits touching files matching this
                        glob pattern (e.g., "*.v").
            module_file_map: Optional mapping from module names to file paths,
                           from RTL extraction, to link commits to modules.
        """
        repo_path = Path(repo_path).resolve()
        repo = Repo(str(repo_path))

        if repo.bare:
            raise ValueError(f"Repository at {repo_path} is bare")

        result = GitExtractionResult(
            repo_path=str(repo_path),
            branch=branch or repo.active_branch.name,
        )

        # Build reverse map: file_path -> module_name(s)
        file_to_modules: dict[str, list[str]] = {}
        if module_file_map:
            for mod_name, fpath in module_file_map.items():
                # Normalize path relative to repo root
                rel = self._normalize_path(fpath, repo_path)
                file_to_modules.setdefault(rel, []).append(mod_name)

        # Iterate commits
        try:
            commits_iter = repo.iter_commits(
                branch or "HEAD",
                max_count=max_commits,
            )
        except Exception as e:
            logger.error("Failed to iterate commits: %s", e)
            return result

        for git_commit in commits_iter:
            result.total_commits += 1

            # Get changed files
            files_changed = self._get_changed_files(git_commit, repo)

            # Apply file filter
            if file_filter:
                import fnmatch
                files_changed = [
                    f for f in files_changed
                    if fnmatch.fnmatch(f, file_filter)
                ]

            # Classify commit
            change_type = self._classify_commit(git_commit.message)

            # Map files to modules
            modules_modified = []
            for fpath in files_changed:
                if fpath in file_to_modules:
                    modules_modified.extend(file_to_modules[fpath])
                else:
                    # Heuristic: try to infer module name from file stem
                    p = Path(fpath)
                    if p.suffix in _RTL_EXTENSIONS:
                        modules_modified.append(p.stem)

            modules_modified = list(set(modules_modified))

            # Compute insertions/deletions
            insertions, deletions = self._get_diff_stats(git_commit)

            commit_info = CommitInfo(
                commit_id=f"{design_name}::commit::{git_commit.hexsha[:8]}",
                sha=git_commit.hexsha,
                short_sha=git_commit.hexsha[:8],
                author=str(git_commit.author),
                author_email=str(git_commit.author.email) if git_commit.author.email else "",
                date=datetime.fromtimestamp(
                    git_commit.committed_date, tz=timezone.utc
                ),
                message=git_commit.message.strip(),
                summary=git_commit.summary.strip(),
                files_changed=files_changed,
                insertions=insertions,
                deletions=deletions,
                modules_modified=modules_modified,
                change_type=change_type,
            )
            result.commits.append(commit_info)

            # Update file histories
            for fpath in files_changed:
                if fpath not in result.file_histories:
                    result.file_histories[fpath] = FileHistory(file_path=fpath)
                fh = result.file_histories[fpath]
                fh.commit_count += 1
                fh.authors.add(str(git_commit.author))
                fh.commits.append(git_commit.hexsha)
                commit_dt = datetime.fromtimestamp(
                    git_commit.committed_date, tz=timezone.utc
                )
                if fh.last_modified is None or commit_dt > fh.last_modified:
                    fh.last_modified = commit_dt

        return result

    def _get_changed_files(self, commit: GitCommit, repo: Repo) -> list[str]:
        """Get the list of files changed in a commit."""
        try:
            if commit.parents:
                diffs = commit.diff(commit.parents[0])
            else:
                # Root commit
                diffs = commit.diff(gitpython.NULL_TREE)

            files = []
            for diff in diffs:
                if diff.a_path:
                    files.append(diff.a_path)
                if diff.b_path and diff.b_path != diff.a_path:
                    files.append(diff.b_path)
            return list(set(files))
        except Exception as e:
            logger.debug("Could not get diff for %s: %s", commit.hexsha[:8], e)
            # Fallback: use stats
            try:
                return list(commit.stats.files.keys())
            except Exception:
                return []

    def _get_diff_stats(self, commit: GitCommit) -> tuple[int, int]:
        """Get total insertions and deletions for a commit."""
        try:
            stats = commit.stats.total
            return stats.get("insertions", 0), stats.get("deletions", 0)
        except Exception:
            return 0, 0

    def _classify_commit(self, message: str) -> str | None:
        """Classify a commit message into a change type."""
        msg_lower = message.lower()
        for pattern, ctype in _CHANGE_TYPE_PATTERNS:
            if re.search(pattern, msg_lower):
                return ctype
        return None

    def _normalize_path(self, fpath: str, repo_root: Path) -> str:
        """Normalize a file path to be relative to repo root."""
        p = Path(fpath)
        if p.is_absolute():
            try:
                return str(p.relative_to(repo_root))
            except ValueError:
                return str(p)
        return str(p)

    def add_to_graph(
        self,
        result: GitExtractionResult,
        graph: nx.DiGraph,
        design_name: str = "design",
    ):
        """Add commits and file history to an existing knowledge graph.

        Links commits to modules they modify.
        Adds change_frequency and last_modified to module nodes.
        """
        # Collect all known module names from the graph
        module_nodes: dict[str, str] = {}  # module_name -> node_id
        for node_id, data in graph.nodes(data=True):
            if data.get("type") == "Module":
                module_nodes[data.get("name", "")] = node_id

        # Add commit nodes
        for commit in result.commits:
            cid = commit.commit_id
            graph.add_node(cid, **{
                "type": "Commit",
                "sha": commit.sha,
                "short_sha": commit.short_sha,
                "author": commit.author,
                "author_email": commit.author_email,
                "date": commit.date.isoformat(),
                "summary": commit.summary,
                "message": commit.message,
                "files_changed": commit.files_changed,
                "insertions": commit.insertions,
                "deletions": commit.deletions,
                "change_type": commit.change_type,
            })

            # Link to modules
            for mod_name in commit.modules_modified:
                if mod_name in module_nodes:
                    mod_id = module_nodes[mod_name]
                    graph.add_edge(cid, mod_id,
                                   relation="modifies",
                                   files=[f for f in commit.files_changed
                                         if mod_name in f])
                    graph.add_edge(mod_id, cid,
                                   relation="modified_by")

        # Annotate module nodes with change frequency data
        module_change_counts: dict[str, int] = {}
        module_last_modified: dict[str, datetime] = {}
        module_authors: dict[str, set[str]] = {}

        for commit in result.commits:
            for mod_name in commit.modules_modified:
                if mod_name in module_nodes:
                    module_change_counts[mod_name] = (
                        module_change_counts.get(mod_name, 0) + 1
                    )
                    if (mod_name not in module_last_modified or
                            commit.date > module_last_modified[mod_name]):
                        module_last_modified[mod_name] = commit.date
                    module_authors.setdefault(mod_name, set()).add(commit.author)

        for mod_name, node_id in module_nodes.items():
            if mod_name in module_change_counts:
                graph.nodes[node_id]["change_frequency"] = module_change_counts[mod_name]
                graph.nodes[node_id]["last_modified"] = (
                    module_last_modified[mod_name].isoformat()
                )
                graph.nodes[node_id]["num_authors"] = len(module_authors.get(mod_name, set()))

    def get_module_file_map(self, graph: nx.DiGraph) -> dict[str, str]:
        """Build a module_name -> src_file mapping from an existing graph.

        Useful for passing to extract() to improve file-to-module linking.
        """
        mapping = {}
        for _, data in graph.nodes(data=True):
            if data.get("type") == "Module" and data.get("src_file"):
                mapping[data["name"]] = data["src_file"]
        return mapping


def print_git_summary(result: GitExtractionResult):
    """Print a summary of extracted Git history."""
    print(f"  Repository: {result.repo_path}")
    print(f"  Branch: {result.branch}")
    print(f"  Total commits processed: {result.total_commits}")
    print(f"  Files tracked: {len(result.file_histories)}")

    if result.commits:
        # Change type distribution
        type_counts: dict[str, int] = {}
        for c in result.commits:
            t = c.change_type or "other"
            type_counts[t] = type_counts.get(t, 0) + 1
        print("  Commit types:")
        for t, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            print(f"    {t}: {count}")

        # Most-changed files
        sorted_files = sorted(
            result.file_histories.values(),
            key=lambda f: f.commit_count,
            reverse=True,
        )
        print("  Most changed files:")
        for fh in sorted_files[:5]:
            print(f"    {fh.file_path}: {fh.commit_count} commits, "
                  f"{len(fh.authors)} authors")
