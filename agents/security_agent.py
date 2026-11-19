"""
agents/security_agent.py
-------------------------
Security Agent:
  - Scans the files this change affects
  - Runs Semgrep, Gitleaks, and Bandit (for Python)
  - Normalizes findings into Finding model
  - Deduplicates overlapping findings across tools
  - Differential scanning: drops findings that already exist on the base branch,
    so a PR is only blamed for problems it introduces (like differential testing)
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from core.ablation import ablation_active
from core.logging import get_logger
from core.state import Finding, ReviewState
from tools.bandit_runner import run_bandit
from tools.deduplicator import deduplicate_findings
from tools.git_ops import cleanup_worktree, create_worktree, review_workspace
from tools.gitleaks_runner import run_gitleaks
from tools.project_detector import detect_project
from tools.semgrep_runner import run_semgrep

logger = get_logger(__name__)


def run_security_agent(state: ReviewState) -> dict[str, Any]:
    """
    Execute static security analysis tools (Semgrep, Gitleaks, Bandit) on the files this
    change affects, and keep only the findings that are new compared with the base branch.
    """
    logger.info("stage_start", stage="security_agent")

    cs = state.get("changeset")
    if not cs:
        logger.warning("security_agent_missing_changeset")
        return {"security_findings": []}

    affected_files = state.get("affected_files", [])

    # Scan the merged code (base + head), not whatever the user's folder has checked out.
    with review_workspace(cs.repo_path, cs.base_ref, cs.head_ref) as workspace:
        merged_dir = Path(workspace).resolve()
        target_files = _target_files(merged_dir, affected_files)
        findings = _scan(merged_dir, target_files)
        if not findings or ablation_active("no_diff_scan"):
            return {"security_findings": findings}
        merged_sigs = [_signature(f, merged_dir) for f in findings]

    base_counts = _base_signatures(cs.repo_path, cs.base_ref, target_files)
    new_findings: list[Finding] = []
    for finding, sig in zip(findings, merged_sigs, strict=True):
        if base_counts[sig] > 0:
            base_counts[sig] -= 1  # one occurrence on base explains one occurrence here
        else:
            new_findings.append(finding)

    logger.info(
        "security_agent_differential",
        merged=len(findings),
        pre_existing=len(findings) - len(new_findings),
        new=len(new_findings),
    )
    return {"security_findings": new_findings}


def _target_files(repo_dir: Path, affected_files: list[str]) -> list[str]:
    """Affected files that exist in repo_dir, relative to it."""
    target_files: list[str] = []
    for f in affected_files:
        p = Path(f)
        full = p if p.is_absolute() else (repo_dir / p)
        if full.is_file():
            try:
                target_files.append(str(full.relative_to(repo_dir)))
            except ValueError:
                target_files.append(str(full))
    return target_files


def _scan(repo_dir: Path, target_files: list[str]) -> list[Finding]:
    """Run all scanners over target_files inside repo_dir and deduplicate across tools."""
    if not target_files:
        logger.info("security_agent_no_target_files_to_scan")
        return []

    all_findings: list[Finding] = []
    scanners_run: list[str] = []

    # 1. Semgrep (Code Vulnerabilities)
    try:
        semgrep_findings = run_semgrep(repo_dir, target_files)
        all_findings.extend(semgrep_findings)
        scanners_run.append("semgrep")
    except Exception as exc:
        logger.warning("semgrep_run_failed", error=str(exc))

    # 2. Gitleaks (Hardcoded Secrets)
    try:
        gitleaks_findings = run_gitleaks(repo_dir, target_files)
        all_findings.extend(gitleaks_findings)
        scanners_run.append("gitleaks")
    except Exception as exc:
        logger.warning("gitleaks_run_failed", error=str(exc))

    # 3. Bandit (Python Vulnerabilities)
    try:
        project_info = detect_project(repo_dir)
        is_python = (
            getattr(project_info, "primary_language", "") == "python"
            or getattr(project_info, "language", "") == "python"
            or any(f.endswith(".py") for f in target_files)
        )
    except Exception:
        is_python = any(f.endswith(".py") for f in target_files)

    if is_python:
        try:
            bandit_findings = run_bandit(repo_dir, target_files)
            all_findings.extend(bandit_findings)
            scanners_run.append("bandit")
        except Exception as exc:
            logger.warning("bandit_run_failed", error=str(exc))

    # Deduplicate overlapping findings across tools
    deduped = deduplicate_findings(all_findings)

    logger.info(
        "security_agent_complete",
        scanners=scanners_run,
        raw_count=len(all_findings),
        deduped_count=len(deduped),
    )
    return deduped


def _base_signatures(repo_path: Path | str, base_ref: str, target_files: list[str]) -> Counter[tuple[str, str, str]]:
    """Signatures of the findings the same files already have on the base branch."""
    base_dir: Path | None = None
    try:
        base_dir = create_worktree(repo_path, base_ref)
        existing = [f for f in target_files if (base_dir / f).is_file()]
        return Counter(_signature(f, base_dir) for f in _scan(base_dir, existing))
    except Exception as exc:
        # Without a baseline every finding counts as new, which errs on the side of reporting.
        logger.warning("security_baseline_scan_failed", error=str(exc)[:300])
        return Counter()
    finally:
        if base_dir:
            cleanup_worktree(repo_path, base_dir)


def _signature(finding: Finding, root: Path) -> tuple[str, str, str]:
    """
    Identify a finding independently of its line number: file, rule, and the text of the
    flagged line. Lines move when code is added above them; the flagged code does not.
    """
    rel = (finding.file or "").replace("\\", "/")
    try:
        p = Path(finding.file or "")
        if p.is_absolute():
            rel = p.resolve().relative_to(root).as_posix()
    except ValueError:
        pass
    rel = rel.lstrip("./")
    rule = finding.rule_id or finding.title or finding.source_tool
    text = ""
    if finding.line:
        try:
            lines = (root / rel).read_text(encoding="utf-8", errors="replace").splitlines()
            if 0 < finding.line <= len(lines):
                text = lines[finding.line - 1]
        except OSError:
            pass
    if not text and finding.code_snippet:
        # Scanner snippets often carry their own line numbers ("12 x = eval(y)"); drop them.
        text = re.sub(r"^\s*\d+\s+", "", finding.code_snippet.strip().splitlines()[0])
    return rel, rule, " ".join(text.split())
