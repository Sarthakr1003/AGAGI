"""
eval/run_bench.py
-----------------
Real-bug benchmark (BugsInPy) and ablation study for AGAGI.

  python -m eval.run_bench prepare --projects youtube-dl tqdm --max-per-project 15
      Downloads BugsInPy and the project repos into data/bench, builds the scenario
      branches, and keeps only bugs that reproduce (failing test passes on the fix,
      fails on the bug). Writes data/bench/manifest.json.

  python -m eval.run_bench run --model groq/openai/gpt-oss-20b --configs full no_triage --limit 30
      Reviews every reproduced bug as two PRs ("bug" and "fixpr") under each config,
      approving every proposed patch. --model pins one model for every agent (default:
      PRIMARY_MODEL from .env). Results are appended to
      eval/results/bench/<model>/results.jsonl; finished runs are skipped, so it can resume.

  python -m eval.run_bench report [--model ...]
      Writes eval/results/bench/<model>/bench_report.md (and prints it).

Configs: "full" (everything on) or any ablation name from core/ablation.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from core.ablation import ABLATIONS
from eval.bugsinpy import (
    DEFAULT_PROJECTS,
    DEFAULT_ROOT,
    PreparedBug,
    ensure_bugsinpy,
    load_bugs,
    load_manifest,
    near,
    patch_lines,
    prepare_bug,
    same_file,
    save_manifest,
)

RESULTS_ROOT = Path(__file__).parent / "results" / "bench"


def results_dir(model: str) -> Path:
    """One folder per model, so results from different models never mix."""
    return RESULTS_ROOT / re.sub(r"[^A-Za-z0-9.-]+", "_", model)
CONFIGS = ("full", *ABLATIONS)


@dataclass
class Run:
    config: str
    bug: str
    scenario: str  # "bug" or "fixpr"
    repeat: int
    model: str = ""
    verdict: str = ""
    flagged: bool = False  # verdict is anything but SAFE
    regressions: int = 0
    pre_existing: int = 0
    reported_issues: int = 0
    review_comments: int = 0
    rc_files: list[str] = field(default_factory=list)
    rc_file_ok: bool | None = None
    rc_line_ok: bool | None = None
    rc_precision: float | None = None  # share of root causes that name a file the developer fixed
    patches: int = 0
    patch_file_ok: bool | None = None
    patch_at_fix_location: bool | None = None
    verified: bool | None = None
    attempts: int = 0
    escalated: bool = False
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_failures: int = 0
    seconds: float = 0.0
    error: str = ""

    @property
    def key(self) -> str:
        return f"{self.config}|{self.bug}|{self.scenario}|{self.repeat}"


# ── Scoring ──────────────────────────────────────────────────────────────────


def _dump(item: Any) -> dict[str, Any]:
    return item.model_dump() if hasattr(item, "model_dump") else dict(item)


def score_bug_scenario(run: Run, state: dict[str, Any], gt_old_lines: dict[str, set[int]]) -> None:
    """Score a PR that reintroduces a real bug. Locations are in the buggy (head) files."""
    root_causes = [_dump(rc) for rc in state.get("root_causes") or []]
    patches = [_dump(p) for p in state.get("patches") or []]
    run.rc_files = sorted({str(rc.get("root_cause_file") or "") for rc in root_causes} - {""})

    def hits(path: str, line: int | None) -> tuple[bool, bool]:
        for gt_file, lines in gt_old_lines.items():
            if same_file(path, gt_file):
                return True, near(line, lines)
        return False, False

    rc_hits = [hits(str(rc.get("root_cause_file") or ""), rc.get("root_cause_line")) for rc in root_causes]
    run.rc_file_ok = any(f for f, _ in rc_hits)
    run.rc_line_ok = any(ln for _, ln in rc_hits)
    run.rc_precision = round(sum(f for f, _ in rc_hits) / len(rc_hits), 3) if rc_hits else None

    run.patches = len(patches)
    if patches:
        run.patch_file_ok = any(
            any(same_file(t, g) for t in p.get("target_files") or [] for g in gt_old_lines) for p in patches
        )
        loc = False
        for p in patches:
            for pfile, plines in patch_lines(p.get("unified_diff") or "", "old").items():
                for gfile, glines in gt_old_lines.items():
                    if same_file(pfile, gfile) and any(near(x, glines) for x in plines):
                        loc = True
        run.patch_at_fix_location = loc
    else:
        run.patch_file_ok = False
        run.patch_at_fix_location = False


def score_common(run: Run, state: dict[str, Any]) -> None:
    run.verdict = str(getattr(state.get("verdict"), "value", state.get("verdict")) or "").lower()
    run.flagged = run.verdict not in ("", "safe")
    run.regressions = len(state.get("regressions") or [])
    run.pre_existing = len(state.get("pre_existing_failures") or [])
    findings = [f for f in state.get("findings") or [] if not getattr(f, "is_false_positive", False)]
    run.review_comments = len(state.get("review_findings") or [])
    run.reported_issues = len(findings) + run.review_comments
    history = list(state.get("verification_history") or [])
    run.attempts = len(history)
    run.escalated = bool(state.get("escalated"))
    if history:
        run.verified = bool(history[-1].get("verified"))


# ── Running ──────────────────────────────────────────────────────────────────


def _set_env(config: str, prepared: PreparedBug, timeout: int) -> None:
    os.environ["AGAGI_ABLATIONS"] = "" if config == "full" else config
    os.environ["AGAGI_TARGET_PYTHON"] = prepared.python
    os.environ["AGAGI_TEST_COMMAND"] = "pytest -q -p no:cacheprovider " + " ".join(prepared.bug.test_files)
    os.environ["AGAGI_TEST_TIMEOUT"] = str(timeout)


def run_one(prepared: PreparedBug, config: str, scenario: str, repeat: int, timeout: int) -> Run:
    # Imported here so 'prepare' and 'report' work without loading the whole pipeline.
    from core.llm import rate_limiter
    from core.orchestrator import start_review
    from core.state import ChangeSet
    from eval.run_eval import _approve_everything, _count_llm_failures

    bug = prepared.bug
    run = Run(config=config, bug=bug.key, scenario=scenario, repeat=repeat)
    base, head = (
        (bug.branches["fixed"], bug.branches["buggy"]) if scenario == "bug" else (bug.branches["buggy"], bug.branches["fixpr"])
    )
    repo = Path(prepared.repo)
    start = time.time()
    try:
        _set_env(config, prepared, timeout)
        # The indexer reads the working tree, so it must show the base branch.
        subprocess.run(["git", "checkout", "-q", "-f", base], cwd=str(repo), check=True, capture_output=True)
        rate_limiter.reset_stats()
        with _count_llm_failures() as failures:
            try:
                snap = _approve_everything(start_review(ChangeSet(repo_path=repo, base_ref=base, head_ref=head)))
            finally:
                run.llm_failures = failures["n"]
        state = snap.get("state") or {}
        score_common(run, state)
        if scenario == "bug":
            score_bug_scenario(run, state, patch_lines(bug.patch_text(), "old"))
        stats = rate_limiter.get_stats()
        run.llm_calls = stats["total_calls"]
        run.prompt_tokens = stats["total_prompt_tokens"]
        run.completion_tokens = stats["total_completion_tokens"]
    except Exception as exc:  # one bad run must not stop the benchmark
        run.error = f"{type(exc).__name__}: {exc}"[:400]
    run.seconds = round(time.time() - start, 1)
    return run


def _done_keys(rdir: Path) -> set[str]:
    results = rdir / "results.jsonl"
    if not results.is_file():
        return set()
    keys = set()
    for line in results.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            # Runs that hit LLM failures used rule-based fallbacks; run them again.
            if not r.get("error") and not r.get("llm_failures"):
                keys.add(f"{r['config']}|{r['bug']}|{r['scenario']}|{r['repeat']}")
    return keys


# ── Report ───────────────────────────────────────────────────────────────────


def _pct(rows: list[dict[str, Any]], key: str) -> str:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return "n/a" if not vals else f"{round(100 * sum(bool(v) for v in vals) / len(vals))}% ({sum(bool(v) for v in vals)}/{len(vals)})"


def _mean(rows: list[dict[str, Any]], key: str, digits: int = 1) -> str:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return "n/a" if not vals else str(round(sum(vals) / len(vals), digits))


def render_report(rows: list[dict[str, Any]], model: str = "") -> str:
    # A run whose LLM calls failed used rule-based fallbacks, so it does not measure AGAGI as configured.
    errors = [r for r in rows if r.get("error")]
    fallback = [r for r in rows if not r.get("error") and r.get("llm_failures")]
    ok_rows = [r for r in rows if not r.get("error") and not r.get("llm_failures")]
    by_cfg: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in ok_rows:
        by_cfg[r["config"]].append(r)
    configs = [c for c in CONFIGS if c in by_cfg]
    bugs = sorted({r["bug"] for r in ok_rows})
    repeats = sorted({r["repeat"] for r in ok_rows})

    lines = [
        "# AGAGI BugsInPy Benchmark",
        "",
        *([f"Model: `{model}`", ""] if model else []),
        f"Bugs: {len(bugs)} | Repeats: {len(repeats)} | Scored runs: {len(ok_rows)} "
        f"(excluded: {len(fallback)} with failed LLM calls, {len(errors)} errored) | Configs: {', '.join(configs)}",
        "",
        "## Bug-introducing PRs (expected: flagged)",
        "",
        "| Config | Flagged | Regression found | RC file | RC line (within 3) | RC precision | Patch proposed "
        "| Patch at fix location | Verified | Mean attempts | Escalated | Mean tokens | Mean time (s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in configs:
        b = [r for r in by_cfg[c] if r["scenario"] == "bug"]
        for r in b:
            r["_proposed"] = r["patches"] > 0
            r["_tokens"] = r["prompt_tokens"] + r["completion_tokens"]
            r["_reg"] = r["regressions"] > 0
        lines.append(
            f"| {c} | {_pct(b, 'flagged')} | {_pct(b, '_reg')} | {_pct(b, 'rc_file_ok')} | {_pct(b, 'rc_line_ok')} "
            f"| {_mean(b, 'rc_precision', 2)} | {_pct(b, '_proposed')} | {_pct(b, 'patch_at_fix_location')} | {_pct(b, 'verified')} "
            f"| {_mean(b, 'attempts')} | {_pct(b, 'escalated')} | {_mean(b, '_tokens', 0)} | {_mean(b, 'seconds')} |"
        )
    lines += [
        "",
        "RC file / RC line = at least one root cause names a file (line) the developer's fix changed. "
        "RC precision = share of all reported root causes that name such a file. "
        "Verified = the approved patch passes the bug's test files and adds no new scanner findings "
        "(a plausible patch; it may still differ from the developer's fix).",
        "",
        "## Developer-fix PRs (expected: SAFE)",
        "",
        "| Config | False alarm (not SAFE) | Mean reported issues | Mean review comments | Spurious regressions | Mean time (s) |",
        "|---|---|---|---|---|---|",
    ]
    for c in configs:
        f = [r for r in by_cfg[c] if r["scenario"] == "fixpr"]
        for r in f:
            r["_spurious"] = r["regressions"] > 0
        lines.append(
            f"| {c} | {_pct(f, 'flagged')} | {_mean(f, 'reported_issues')} | {_mean(f, 'review_comments')} "
            f"| {_pct(f, '_spurious')} | {_mean(f, 'seconds')} |"
        )
    if fallback:
        lines += ["", "## Excluded: LLM calls failed (re-run with `run` once quota is available)", ""]
        lines += [f"- {r['config']} / {r['bug']} / {r['scenario']}: {r['llm_failures']} failed call(s)" for r in fallback]
    if errors:
        lines += ["", "## Errored runs", ""] + [f"- {r['config']} / {r['bug']} / {r['scenario']}: {r['error']}" for r in errors]
    return "\n".join(lines) + "\n"


def write_report(rdir: Path) -> str:
    results = rdir / "results.jsonl"
    rows = []
    if results.is_file():
        rows = [json.loads(x) for x in results.read_text(encoding="utf-8").splitlines() if x.strip()]
    # Keep only the latest attempt of each run key.
    latest: dict[str, dict[str, Any]] = {}
    for r in rows:
        latest[f"{r['config']}|{r['bug']}|{r['scenario']}|{r['repeat']}"] = r
    report = render_report(list(latest.values()), model=rdir.name)
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "bench_report.md").write_text(report, encoding="utf-8")
    return report


# ── CLI ──────────────────────────────────────────────────────────────────────


def cmd_prepare(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()  # git runs in other folders, so relative paths would break
    bugsinpy_dir = ensure_bugsinpy(root)
    bugs = load_bugs(bugsinpy_dir, args.projects, args.max_per_project)
    print(f"{len(bugs)} candidate bugs")
    prepared: list[PreparedBug] = []
    for i, bug in enumerate(bugs, 1):
        print(f"[{i}/{len(bugs)}] {bug.key} ...", flush=True)
        try:
            p = prepare_bug(root, bugsinpy_dir, bug, timeout=args.timeout)
        except Exception as exc:
            p = PreparedBug(bug, "", "", False, f"{type(exc).__name__}: {exc}"[:300])
        print(f"      {'OK ' if p.reproduced else 'skip'} {p.detail[:160]}", flush=True)
        prepared.append(p)
    save_manifest(root / "manifest.json", prepared)
    n = sum(p.reproduced for p in prepared)
    print(f"\n{n}/{len(prepared)} bugs reproduce. Manifest: {root / 'manifest.json'}")


def _use_model(model: str | None) -> str:
    """Pin primary and fast model for the whole run. Settings are already loaded (core imports them)."""
    from core.config import settings

    if model:
        settings.primary_model = model
        settings.fast_model = model
    return settings.primary_model


def cmd_run(args: argparse.Namespace) -> None:
    for c in args.configs:
        if c not in CONFIGS:
            raise SystemExit(f"Unknown config '{c}'. Choose from: {', '.join(CONFIGS)}")
    if args.no_cache:
        os.environ["AGAGI_LLM_CACHE"] = "off"
    model = _use_model(args.model)
    if not args.allow_no_llm:
        from core.llm import smoke_test_providers

        if not any(i.get("status") == "ok" and i.get("model") == model for i in smoke_test_providers().values()):
            raise SystemExit(
                f"The model {model} does not answer, so runs would use other models or rule-based fallbacks. "
                "Fix the key (python -m tools.check_llm), choose another --model, or pass --allow-no-llm."
            )
    rdir = results_dir(model)
    print(f"Model: {model}  Results: {rdir}")
    prepared = [p for p in load_manifest(Path(args.root) / "manifest.json") if p.reproduced]
    if args.bugs:
        prepared = [p for p in prepared if p.bug.key in args.bugs]
    if args.limit:
        prepared = prepared[: args.limit]
    done = _done_keys(rdir)
    rdir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (p, c, s, rep)
        for rep in range(args.repeats)
        for c in args.configs
        for p in prepared
        for s in args.scenarios
    ]
    print(f"{len(jobs)} runs planned ({len(done)} already done)")
    for i, (p, c, s, rep) in enumerate(jobs, 1):
        key = f"{c}|{p.bug.key}|{s}|{rep}"
        if key in done:
            continue
        print(f"[{i}/{len(jobs)}] {key}", flush=True)
        run = run_one(p, c, s, rep, args.timeout)
        run.model = model
        print(
            f"      verdict={run.verdict or 'ERROR'} rc_file={run.rc_file_ok} verified={run.verified} "
            f"tokens={run.prompt_tokens + run.completion_tokens} llm_failures={run.llm_failures} "
            f"{run.seconds}s {run.error}",
            flush=True,
        )
        with (rdir / "results.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(run)) + "\n")
    print("\n" + write_report(rdir))


def main() -> None:
    parser = argparse.ArgumentParser(description="AGAGI BugsInPy benchmark and ablations")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="Where datasets, repos and venvs live")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="Download BugsInPy, build branches, keep reproducible bugs")
    p.add_argument("--projects", nargs="+", default=list(DEFAULT_PROJECTS))
    p.add_argument("--max-per-project", type=int, default=15)
    p.add_argument("--timeout", type=int, default=300)
    p.set_defaults(func=cmd_prepare)

    r = sub.add_parser("run", help="Review each bug under each config")
    r.add_argument("--configs", nargs="+", default=["full"], help=f"From: {', '.join(CONFIGS)}")
    r.add_argument("--scenarios", nargs="+", default=["bug", "fixpr"], choices=["bug", "fixpr"])
    r.add_argument("--repeats", type=int, default=1)
    r.add_argument("--limit", type=int, default=None, help="Use only the first N reproduced bugs")
    r.add_argument("--bugs", nargs="*", default=None, help="Only these bug keys, e.g. tqdm-3")
    r.add_argument("--timeout", type=int, default=300, help="Seconds per test run")
    r.add_argument("--no-cache", action="store_true", help="Fresh LLM answers (needed for meaningful repeats)")
    r.add_argument("--allow-no-llm", action="store_true", help="Run even if no LLM provider works")
    r.add_argument("--model", default=None, help="LiteLLM model for every agent, e.g. groq/openai/gpt-oss-120b")
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="Rebuild the report from results.jsonl")
    rep.add_argument("--model", default=None, help="Which model's results (default: PRIMARY_MODEL)")
    rep.set_defaults(func=lambda a: print(write_report(results_dir(_use_model(a.model)))))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
