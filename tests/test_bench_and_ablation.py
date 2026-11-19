"""Ablation switches and the BugsInPy benchmark helpers (no network, no AI)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from agents.merge_sim_agent import run_merge_simulation_agent
from agents.triage_agent import run_triage_agent
from core.ablation import ablation_active, active_ablations
from core.state import ChangeSet, Finding, Patch, RootCause, Severity
from eval.bugsinpy import (
    Bug,
    build_branches,
    near,
    parse_info,
    parse_test_commands,
    patch_lines,
    same_file,
)
from eval.run_bench import Run, render_report, score_bug_scenario

FIX_PATCH = """\
diff --git a/pkg/mod.py b/pkg/mod.py
index 1111111..2222222 100644
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -1,4 +1,4 @@
 def f(x):
-    return x + 2
+    return x + 1


@@ -10,2 +10,3 @@ def g():
     a = 1
+    a += 1
     return a
"""


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=str(cwd), text=True, capture_output=True, check=True,
    ).stdout.strip()


# ── Ablation switches ────────────────────────────────────────────────────────


def test_ablations_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGAGI_ABLATIONS", raising=False)
    assert active_ablations() == set()
    assert not ablation_active("no_triage")


def test_ablations_parse_and_reject_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGAGI_ABLATIONS", " no_triage , no_feedback ")
    assert active_ablations() == {"no_triage", "no_feedback"}
    assert ablation_active("no_feedback")
    monkeypatch.setenv("AGAGI_ABLATIONS", "no_such_thing")
    with pytest.raises(ValueError):
        active_ablations()


def test_no_triage_keeps_every_finding_without_ai(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGAGI_ABLATIONS", "no_triage")
    findings = [
        Finding(type="security", severity=Severity.LOW, file="a.py", line=1, explanation="e1", source_tool="bandit"),
        Finding(type="security", severity=Severity.HIGH, file="b.py", line=2, explanation="e2", source_tool="semgrep"),
    ]
    state: dict[str, Any] = {
        "changeset": ChangeSet(repo_path=tmp_path, base_ref="main", head_ref="HEAD"),
        "security_findings": findings,
        "dependency_findings": [],
    }
    with patch("agents.triage_agent.chat_structured", side_effect=AssertionError("LLM must not be called")):
        out = run_triage_agent(state)  # type: ignore[arg-type]
    assert len(out["findings"]) == 2
    assert all(not f.is_false_positive for f in out["findings"])


@pytest.fixture()
def repo_with_failing_main(tmp_path: Path) -> Path:
    repo = tmp_path / "r"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)
    (repo / "test_old.py").write_text("def test_always_broken():\n    assert False\n", encoding="utf-8")
    (repo / "test_ok.py").write_text("def test_fine():\n    assert True\n", encoding="utf-8")
    _git(["add", "."], repo)
    _git(["commit", "-q", "-m", "base"], repo)
    _git(["checkout", "-q", "-b", "feature"], repo)
    (repo / "notes.py").write_text("X = 1\n", encoding="utf-8")
    _git(["add", "."], repo)
    _git(["commit", "-q", "-m", "feature"], repo)
    _git(["checkout", "-q", "main"], repo)
    return repo


@pytest.mark.parametrize("ablate, regressions, pre_existing", [("", 0, 1), ("no_differential", 1, 0)])
def test_no_differential_blames_pre_existing_failures(
    repo_with_failing_main: Path, monkeypatch: pytest.MonkeyPatch, ablate: str, regressions: int, pre_existing: int
) -> None:
    monkeypatch.setenv("AGAGI_ABLATIONS", ablate)
    monkeypatch.delenv("AGAGI_TEST_COMMAND", raising=False)
    monkeypatch.delenv("AGAGI_TARGET_PYTHON", raising=False)
    monkeypatch.setattr("agents.merge_sim_agent.is_docker_available", lambda: False)
    out = run_merge_simulation_agent(
        {"changeset": ChangeSet(repo_path=repo_with_failing_main, base_ref="main", head_ref="feature")}  # type: ignore[arg-type]
    )
    assert len(out["regressions"]) == regressions
    assert len(out["pre_existing_failures"]) == pre_existing


# ── LLM rate-limit handling ──────────────────────────────────────────────────


def _rate_limit(msg: str) -> Exception:
    import litellm

    return litellm.RateLimitError(message=msg, llm_provider="groq", model="m")


def test_retry_waits_as_long_as_the_provider_asks() -> None:
    from tenacity import RetryCallState

    from core.llm import _retry_wait

    def state(exc: Exception, attempt: int = 1) -> RetryCallState:
        rs = RetryCallState(retry_object=None, fn=None, args=(), kwargs={})  # type: ignore[arg-type]
        rs.attempt_number = attempt
        rs.set_exception((type(exc), exc, None))
        return rs

    assert _retry_wait(state(_rate_limit("Please try again in 8.43s. Need more tokens?"))) == pytest.approx(9.43)
    assert _retry_wait(state(_rate_limit("Please try again in 1m30.5s"))) == 90.0  # capped
    assert _retry_wait(state(_rate_limit('"retryDelay": "12s"'))) == pytest.approx(13.0)
    assert _retry_wait(state(RuntimeError("boom"), attempt=3)) == 8.0


def test_no_retry_for_oversized_requests_or_daily_quota() -> None:
    from core.llm import _should_retry

    assert _should_retry(_rate_limit("Rate limit reached ... on tokens per minute (TPM)"))
    assert not _should_retry(_rate_limit("Request too large for model on tokens per minute"))
    assert not _should_retry(_rate_limit("quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"))
    assert not _should_retry(ValueError("not an API error"))


# ── BugsInPy parsing and scoring ─────────────────────────────────────────────


def test_parse_info_and_test_commands() -> None:
    info = parse_info('python_version="3.7.0"\nbuggy_commit_id="abc"\ntest_file="t/a.py;t/b.py"\n')
    assert info == {"python_version": "3.7.0", "buggy_commit_id": "abc", "test_file": "t/a.py;t/b.py"}
    cmds = parse_test_commands("#!/bin/bash\n\npython -m pytest t/a.py::T::x\n# note\n")
    assert cmds == ["python -m pytest t/a.py::T::x"]


def test_patch_lines_old_and_new_sides() -> None:
    old = patch_lines(FIX_PATCH, "old")
    new = patch_lines(FIX_PATCH, "new")
    assert old == {"pkg/mod.py": {2, 10, 11}}  # replaced line 2; insertion between old 10 and 11
    assert new == {"pkg/mod.py": {2, 11}}  # replacement line 2; inserted line 11


def test_same_file_and_near() -> None:
    assert same_file("C:/tmp/wt/pkg/mod.py", "pkg/mod.py")
    assert same_file("./pkg/mod.py", "pkg/mod.py")
    assert not same_file("pkg/other.py", "pkg/mod.py")
    assert near(5, {2}) and not near(6, {2}) and not near(None, {2})


def test_score_bug_scenario_file_line_and_patch_location() -> None:
    gt = patch_lines(FIX_PATCH, "old")
    good_patch = "--- a/pkg/mod.py\n+++ b/pkg/mod.py\n@@ -2,1 +2,1 @@\n-    return x + 2\n+    return x + 1\n"
    state = {
        "root_causes": [RootCause(root_cause_file="pkg/mod.py", root_cause_line=3, explanation="off by one")],
        "patches": [Patch(unified_diff=good_patch, rationale="r", target_files=["pkg/mod.py"])],
    }
    run = Run(config="full", bug="p-1", scenario="bug", repeat=0)
    score_bug_scenario(run, state, gt)
    assert run.rc_file_ok and run.rc_line_ok and run.patch_file_ok and run.patch_at_fix_location

    wrong = {"root_causes": [RootCause(root_cause_file="pkg/mod.py", root_cause_line=40, explanation="x")], "patches": []}
    run2 = Run(config="full", bug="p-1", scenario="bug", repeat=0)
    score_bug_scenario(run2, wrong, gt)
    assert run2.rc_file_ok and not run2.rc_line_ok
    assert run2.patches == 0 and run2.patch_at_fix_location is False


def test_render_report_lists_each_config() -> None:
    rows = [
        {**Run("full", "p-1", "bug", 0, verdict="needs_changes", flagged=True, regressions=1, rc_file_ok=True,
               rc_line_ok=True, patches=1, patch_at_fix_location=True, verified=True, attempts=1).__dict__},
        {**Run("full", "p-1", "fixpr", 0, verdict="safe").__dict__},
        {**Run("no_triage", "p-1", "fixpr", 0, verdict="needs_changes", flagged=True, reported_issues=2).__dict__},
    ]
    report = render_report(rows)
    assert "| full | 100% (1/1) |" in report
    assert "| no_triage | 100% (1/1) | 2.0 |" in report


def test_build_branches_reverses_and_reapplies_fix(tmp_path: Path) -> None:
    repo = tmp_path / "proj"
    (repo / "pkg").mkdir(parents=True)
    _git(["init", "-q", "-b", "main"], repo)
    body = ["def f(x):", "    return x + 1", "", "", "", "", "", "", "def g():", "    a = 1", "    a += 1", "    return a"]
    (repo / "pkg" / "mod.py").write_text("\n".join(body) + "\n", encoding="utf-8")
    _git(["add", "."], repo)
    _git(["commit", "-q", "-m", "fixed"], repo)
    fixed = _git(["rev-parse", "HEAD"], repo)
    patch_file = tmp_path / "bug_patch.txt"
    patch_file.write_text(FIX_PATCH, encoding="utf-8", newline="\n")
    bug = Bug("proj", "1", "", "deadbeef", fixed, "3.11", ["t.py"], [], str(patch_file))

    build_branches(repo, bug)

    show = lambda ref: _git(["show", f"{ref}:pkg/mod.py"], repo)  # noqa: E731
    assert "return x + 2" in show(bug.branches["buggy"]) and "a += 1" not in show(bug.branches["buggy"])
    assert show(bug.branches["fixpr"]) == show(bug.branches["fixed"])
    build_branches(repo, bug)  # idempotent
