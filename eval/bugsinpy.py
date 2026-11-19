"""
eval/bugsinpy.py
----------------
Turns BugsInPy (real Python bugs with their developer fixes and failing tests) into
AGAGI review scenarios.

BugsInPy: https://github.com/soarsmu/BugsInPy
Each bug lists a buggy commit, a fixed commit, the test file(s) that expose the bug,
the commands that run the failing tests (run_test.sh) and the developer fix (bug_patch.txt).

Every bug gives two scenarios on three local branches:

  agai-bench/<bug>/fixed  the fixed commit (source fixed, tests that expose the bug)
  agai-bench/<bug>/buggy  fixed + the developer fix reversed  (the bug put back)
  agai-bench/<bug>/fixpr  buggy + the developer fix applied again

  "bug" scenario:    base=fixed, head=buggy  -> a PR that introduces a real bug; expected NEEDS_CHANGES
  "fixpr" scenario:  base=buggy, head=fixpr  -> the developer's own fix as a PR; expected SAFE

A bug is used only if it reproduces here: its failing tests pass on 'fixed' and fail on 'buggy'.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

BUGSINPY_URL = "https://github.com/soarsmu/BugsInPy.git"
DEFAULT_ROOT = Path("data") / "bench"
# Pure-Python projects whose tests run without services or heavy native dependencies.
DEFAULT_PROJECTS = ("youtube-dl", "tqdm", "PySnooper", "thefuck", "httpie", "cookiecutter")
BRANCH_PREFIX = "agai-bench"
_GIT_IDENTITY = ["-c", "user.name=agai-bench", "-c", "user.email=agai-bench@localhost"]
LINE_TOLERANCE = 3


@dataclass
class Bug:
    project: str
    bug_id: str
    github_url: str
    buggy_commit: str
    fixed_commit: str
    python_version: str
    test_files: list[str]
    test_commands: list[str]
    patch_path: str

    @property
    def key(self) -> str:
        return f"{self.project}-{self.bug_id}"

    @property
    def branches(self) -> dict[str, str]:
        base = f"{BRANCH_PREFIX}/{self.key}"
        return {"fixed": f"{base}/fixed", "buggy": f"{base}/buggy", "fixpr": f"{base}/fixpr"}

    def patch_text(self) -> str:
        # A checkout with core.autocrlf=true turns the patches into CRLF; the sources are LF.
        return Path(self.patch_path).read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")


@dataclass
class PreparedBug:
    bug: Bug
    repo: str
    python: str
    reproduced: bool
    detail: str = ""
    gt_files: list[str] = field(default_factory=list)


# ── Parsing ──────────────────────────────────────────────────────────────────


def parse_info(text: str) -> dict[str, str]:
    """Parse BugsInPy's key="value" info files."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = re.match(r'\s*([A-Za-z_]+)\s*=\s*"?(.*?)"?\s*$', line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def parse_test_commands(run_test_sh: str) -> list[str]:
    """Commands from run_test.sh, without comments, shebang or blank lines."""
    cmds = []
    for line in run_test_sh.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            cmds.append(line)
    return cmds


def load_bugs(bugsinpy_dir: Path, projects: tuple[str, ...] | list[str], max_per_project: int | None = None) -> list[Bug]:
    bugs: list[Bug] = []
    for project in projects:
        pdir = bugsinpy_dir / "projects" / project
        if not pdir.is_dir():
            print(f"  ! project '{project}' not found in BugsInPy, skipping")
            continue
        pinfo = parse_info((pdir / "project.info").read_text(encoding="utf-8"))
        bug_dirs = sorted((pdir / "bugs").iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 10**9)
        if max_per_project:
            bug_dirs = bug_dirs[:max_per_project]
        for bdir in bug_dirs:
            info_file, patch, run_test = bdir / "bug.info", bdir / "bug_patch.txt", bdir / "run_test.sh"
            if not (info_file.is_file() and patch.is_file() and run_test.is_file()):
                continue
            info = parse_info(info_file.read_text(encoding="utf-8"))
            bugs.append(
                Bug(
                    project=project,
                    bug_id=bdir.name,
                    github_url=pinfo.get("github_url", ""),
                    buggy_commit=info.get("buggy_commit_id", ""),
                    fixed_commit=info.get("fixed_commit_id", ""),
                    python_version=info.get("python_version", ""),
                    test_files=[t for t in info.get("test_file", "").split(";") if t.strip()],
                    test_commands=parse_test_commands(run_test.read_text(encoding="utf-8", errors="replace")),
                    patch_path=str(patch.resolve()),
                )
            )
    return bugs


def patch_lines(patch_text: str, side: str = "old") -> dict[str, set[int]]:
    """
    Lines a unified diff touches, per file.

    side="old": line numbers in the pre-patch file (removed lines, plus the two lines
    around each pure insertion). side="new": the same in the post-patch file.
    """
    if side not in ("old", "new"):
        raise ValueError("side must be 'old' or 'new'")
    result: dict[str, set[int]] = {}
    current: str | None = None
    old_no = new_no = 0
    removed: list[int] = []  # the current block of consecutive -/+ lines
    added: list[int] = []
    block_old = block_new = 0  # where the block starts on each side

    def flush() -> None:
        if current is not None and (removed or added):
            own, pos = (removed, block_old) if side == "old" else (added, block_new)
            # A pure insertion (or deletion) has no lines on this side: use the lines around it.
            result.setdefault(current, set()).update(own or {max(pos - 1, 1), pos})
        removed.clear()
        added.clear()

    for line in patch_text.splitlines():
        if line.startswith("--- ") or line.startswith("+++ "):
            flush()
            if line.startswith("+++ "):
                path = line[4:].strip()
                current = None if path == "/dev/null" else re.sub(r"^b/", "", path)
            elif side == "old" and line[4:].strip() != "/dev/null":
                current = re.sub(r"^a/", "", line[4:].strip())
            continue
        m = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
        if m:
            flush()
            old_no, new_no = int(m.group(1)), int(m.group(2))
            continue
        if current is None or not line or line.startswith("\\"):
            continue
        tag = line[0]
        if tag in "-+" and not (removed or added):
            block_old, block_new = old_no, new_no
        if tag == "-":
            removed.append(old_no)
            old_no += 1
        elif tag == "+":
            added.append(new_no)
            new_no += 1
        else:
            flush()
            old_no += 1
            new_no += 1
    flush()
    return {f: lines for f, lines in result.items() if lines}


def same_file(candidate: str, truth: str) -> bool:
    c = candidate.replace("\\", "/").strip().lstrip("./")
    t = truth.replace("\\", "/").strip().lstrip("./")
    return bool(c) and (c == t or c.endswith("/" + t) or t.endswith("/" + c))


def near(line: int | None, lines: set[int], tol: int = LINE_TOLERANCE) -> bool:
    return line is not None and any(abs(line - x) <= tol for x in lines)


# ── Git / environment ────────────────────────────────────────────────────────


def _git(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(["git", *args], cwd=str(cwd), text=True, capture_output=True, check=False)
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()[:400]}")
    return proc


def ensure_bugsinpy(root: Path) -> Path:
    root = root.resolve()
    target = root / "BugsInPy"
    if not (target / "projects").is_dir():
        root.mkdir(parents=True, exist_ok=True)
        print(f"  cloning BugsInPy into {target} ...")
        _git(["clone", "-c", "core.autocrlf=false", "--depth", "1", BUGSINPY_URL, str(target)], cwd=root)
    return target


def ensure_project_repo(root: Path, bug: Bug) -> Path:
    repo = root.resolve() / "repos" / bug.project
    if not (repo / ".git").exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        print(f"  cloning {bug.github_url} ...")
        _git(
            ["clone", "-c", "core.autocrlf=false", "--filter=blob:none", "--no-checkout", bug.github_url, str(repo)],
            cwd=repo.parent,
        )
    return repo


def ensure_venv(root: Path, bug: Bug, bugsinpy_dir: Path) -> str:
    """A virtual environment per project, outside the repo so the indexer never scans it."""
    venv = root.resolve() / "venvs" / bug.project
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    marker = venv / ".agai_ready"
    if not marker.exists():
        print(f"  creating test environment for {bug.project} ...")
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        subprocess.run([str(py), "-m", "pip", "install", "-q", "pytest"], check=False)
        req = bugsinpy_dir / "projects" / bug.project / "bugs" / bug.bug_id / "requirements.txt"
        if req.is_file():
            # Old pins often fail on new Pythons; install what we can, line by line.
            for line in req.read_text(encoding="utf-8", errors="replace").splitlines():
                pkg = line.strip()
                if pkg and not pkg.startswith("#"):
                    subprocess.run([str(py), "-m", "pip", "install", "-q", pkg], check=False, capture_output=True)
        # The pinned test runners (pytest 5, py 1.8) do not import on current Pythons; replace them last.
        subprocess.run([str(py), "-m", "pip", "uninstall", "-y", "-q", "py"], check=False, capture_output=True)
        subprocess.run([str(py), "-m", "pip", "install", "-q", "-U", "pytest"], check=False)
        marker.write_text("ok", encoding="utf-8")
    return str(py)


def _branch_exists(repo: Path, name: str) -> bool:
    return _git(["rev-parse", "--verify", "--quiet", f"refs/heads/{name}"], cwd=repo, check=False).returncode == 0


def build_branches(repo: Path, bug: Bug) -> None:
    """Create the fixed / buggy / fixpr branches (idempotent)."""
    br = bug.branches
    if all(_branch_exists(repo, b) for b in br.values()):
        return
    _git(["fetch", "--quiet", "origin", bug.fixed_commit], cwd=repo, check=False)
    _git(["branch", "-f", br["fixed"], bug.fixed_commit], cwd=repo)
    tmp = Path(tempfile.mkdtemp(prefix="agai_bench_"))
    wt = tmp / "wt"
    patch_file = tmp / "fix.patch"
    patch_file.write_text(bug.patch_text(), encoding="utf-8", newline="\n")
    try:
        _git(["worktree", "add", "--detach", str(wt), bug.fixed_commit], cwd=repo)
        _git(["apply", "-R", "--whitespace=nowarn", str(patch_file)], cwd=wt)
        _git([*_GIT_IDENTITY, "commit", "-a", "-q", "-m", f"Reintroduce {bug.key} (developer fix reversed)"], cwd=wt)
        _git(["branch", "-f", br["buggy"], "HEAD"], cwd=wt)
        _git(["apply", "--whitespace=nowarn", str(patch_file)], cwd=wt)
        _git([*_GIT_IDENTITY, "commit", "-a", "-q", "-m", f"Fix {bug.key} (developer fix)"], cwd=wt)
        _git(["branch", "-f", br["fixpr"], "HEAD"], cwd=wt)
    finally:
        _git(["worktree", "remove", "--force", str(wt)], cwd=repo, check=False)
        _git(["worktree", "prune"], cwd=repo, check=False)
        shutil.rmtree(tmp, ignore_errors=True)


def _run_target_tests(repo: Path, ref: str, bug: Bug, python: str, timeout: int) -> tuple[bool, str]:
    """Run the bug's own failing-test commands at ref. Returns (all passed, output tail)."""
    wt = Path(tempfile.mkdtemp(prefix="agai_repro_"))
    try:
        _git(["worktree", "add", "--detach", str(wt), ref], cwd=repo)
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{wt}{os.pathsep}{env.get('PYTHONPATH', '')}"
        ok, tail = True, ""
        for cmd in bug.test_commands:
            if cmd.startswith("python3 ") or cmd.startswith("python "):
                cmd = f'"{python}" ' + cmd.split(" ", 1)[1]
            elif cmd.startswith("pytest"):
                cmd = f'"{python}" -m ' + cmd
            try:
                proc = subprocess.run(
                    cmd, shell=True, cwd=str(wt), env=env, text=True, capture_output=True, timeout=timeout, check=False
                )
                tail = (proc.stdout + proc.stderr)[-600:]
                ok = ok and proc.returncode == 0
            except subprocess.TimeoutExpired:
                ok, tail = False, f"timed out after {timeout}s"
        return ok, tail
    finally:
        _git(["worktree", "remove", "--force", str(wt)], cwd=repo, check=False)
        _git(["worktree", "prune"], cwd=repo, check=False)
        shutil.rmtree(wt, ignore_errors=True)


def prepare_bug(root: Path, bugsinpy_dir: Path, bug: Bug, timeout: int = 300) -> PreparedBug:
    repo = ensure_project_repo(root, bug)
    python = ensure_venv(root, bug, bugsinpy_dir)
    gt_files = sorted(patch_lines(bug.patch_text(), "old"))
    try:
        build_branches(repo, bug)
    except Exception as exc:
        return PreparedBug(bug, str(repo), python, False, f"branch build failed: {exc}", gt_files)
    fixed_ok, fixed_out = _run_target_tests(repo, bug.branches["fixed"], bug, python, timeout)
    if not fixed_ok:
        return PreparedBug(bug, str(repo), python, False, "failing test does not pass on the fixed version: " + fixed_out, gt_files)
    buggy_ok, _ = _run_target_tests(repo, bug.branches["buggy"], bug, python, timeout)
    if buggy_ok:
        return PreparedBug(bug, str(repo), python, False, "failing test also passes on the buggy version", gt_files)
    return PreparedBug(bug, str(repo), python, True, "reproduced", gt_files)


def save_manifest(path: Path, prepared: list[PreparedBug]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{**asdict(p), "bug": asdict(p.bug)} for p in prepared]
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> list[PreparedBug]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [PreparedBug(**{**r, "bug": Bug(**r["bug"])}) for r in rows]
