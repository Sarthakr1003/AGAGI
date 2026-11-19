"""Differential scanning (only new findings are reported) and search/replace patch generation."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from agents.fix_agent import (
    CodeEdit,
    GeneratedPatchOutput,
    _code_window,
    apply_edits,
    generate_patch_for_root_cause,
    make_unified_diff,
)
from agents.security_agent import run_security_agent
from core.state import ChangeSet, Finding, RootCause, Severity


def _git(args: list[str], cwd: Path) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.autocrlf=false", *args],
        cwd=str(cwd), text=True, capture_output=True, check=True,
    ).stdout


# ── Differential scanning ────────────────────────────────────────────────────


def _fake_eval_scanner(repo_dir: Path, files: list[str]) -> list[Finding]:
    """Flags every line containing eval( - a stand-in for Bandit B307."""
    out = []
    for rel in files:
        for no, line in enumerate((repo_dir / rel).read_text(encoding="utf-8").splitlines(), start=1):
            if "eval(" in line:
                out.append(Finding(type="security", severity=Severity.MEDIUM, file=rel, line=no,
                                   rule_id="B307", explanation="eval", source_tool="bandit"))
    return out


@pytest.fixture()
def eval_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "r"
    repo.mkdir()
    _git(["init", "-q", "-b", "main"], repo)
    (repo / "app.py").write_text("def old(x):\n    return eval(x)\n", encoding="utf-8")
    _git(["add", "."], repo)
    _git(["commit", "-q", "-m", "base"], repo)
    _git(["checkout", "-q", "-b", "feature"], repo)
    # New code above the old eval moves it down; one new eval is added.
    (repo / "app.py").write_text(
        "import os\n\n\ndef new(y):\n    return eval(y)\n\n\ndef old(x):\n    return eval(x)\n", encoding="utf-8"
    )
    _git(["commit", "-q", "-am", "feature"], repo)
    _git(["checkout", "-q", "main"], repo)
    return repo


def _scan(repo: Path, monkeypatch: pytest.MonkeyPatch, ablations: str = "") -> list[Finding]:
    monkeypatch.setenv("AGAGI_ABLATIONS", ablations)
    state: dict[str, Any] = {
        "changeset": ChangeSet(repo_path=repo, base_ref="main", head_ref="feature"),
        "affected_files": ["app.py"],
    }
    with (
        patch("agents.security_agent.run_bandit", side_effect=_fake_eval_scanner),
        patch("agents.security_agent.run_semgrep", return_value=[]),
        patch("agents.security_agent.run_gitleaks", return_value=[]),
    ):
        return run_security_agent(state)["security_findings"]  # type: ignore[arg-type]


def test_only_findings_new_to_the_pr_are_reported(eval_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    findings = _scan(eval_repo, monkeypatch)
    assert [(f.file, f.line) for f in findings] == [("app.py", 5)]  # the moved eval on line 9 is old


def test_no_diff_scan_ablation_reports_everything(eval_repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    findings = _scan(eval_repo, monkeypatch, "no_diff_scan")
    assert sorted(f.line for f in findings) == [5, 9]


# ── Search/replace edits ─────────────────────────────────────────────────────

SRC = "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n"


def test_apply_edits_exact_numbered_and_whitespace_tolerant() -> None:
    new, err = apply_edits(SRC, [CodeEdit(search="    return a - b", replace="    return a - b - 0")])
    assert not err and "a - b - 0" in new
    new, err = apply_edits(SRC, [CodeEdit(search="   2 |     return a + b", replace="    return b + a")])
    assert not err and "return b + a" in new
    new, err = apply_edits(SRC, [CodeEdit(search="def sub(a, b):   \n    return a - b  ", replace="def sub(a, b):\n    return b")])
    assert not err and new.endswith("def sub(a, b):\n    return b\n")


@pytest.mark.parametrize(
    "edit, message",
    [
        (CodeEdit(search="return a", replace="x"), "matches 2 places"),
        (CodeEdit(search="    return a * b", replace="x"), "was not found"),
        (CodeEdit(search="    return a + b", replace="    return a + b"), "do not change"),
        (CodeEdit(search="  ", replace="x"), "empty"),
    ],
)
def test_apply_edits_rejects_unusable_edits(edit: CodeEdit, message: str) -> None:
    new, err = apply_edits(SRC, [edit])
    assert message in err and new == SRC


def test_unified_diff_applies_to_file_without_final_newline(tmp_path: Path) -> None:
    _git(["init", "-q"], tmp_path)
    (tmp_path / "m.py").write_bytes(b"x = 1\ny = 2")
    _git(["add", "."], tmp_path)
    _git(["commit", "-q", "-m", "i"], tmp_path)
    diff = make_unified_diff("m.py", "x = 1\ny = 2", "x = 1\ny = 3")
    (tmp_path / "p.diff").write_text(diff, encoding="utf-8", newline="\n")
    _git(["apply", "--check", "p.diff"], tmp_path)


def test_code_window_centres_on_root_cause_in_long_files() -> None:
    text = "\n".join(f"line{i}" for i in range(1, 1001))
    window = _code_window(text, 500)
    assert window.startswith("(lines 420-580 of 1000)")
    assert " 500 | line500" in window and "line1\n" not in window
    assert _code_window("a\nb", None).startswith("(lines 1-2 of 2)")


def test_patch_for_crlf_file_applies(tmp_path: Path) -> None:
    _git(["init", "-q"], tmp_path)
    (tmp_path / "calc.py").write_bytes(b"def add(a, b):\r\n    return a - b\r\n")
    _git(["add", "."], tmp_path)
    _git(["commit", "-q", "-m", "i"], tmp_path)
    out = GeneratedPatchOutput(edits=[CodeEdit(search="    return a - b", replace="    return a + b")], rationale="r")
    rc = RootCause(root_cause_file="calc.py", root_cause_line=2, explanation="wrong operator")
    with patch("agents.fix_agent.chat_structured", return_value=out):
        p = generate_patch_for_root_cause(rc, tmp_path)
    assert p is not None
    (tmp_path / "p.diff").write_bytes(p.unified_diff.encode("utf-8"))
    _git(["apply", "p.diff"], tmp_path)
    assert (tmp_path / "calc.py").read_bytes() == b"def add(a, b):\r\n    return a + b\r\n"
