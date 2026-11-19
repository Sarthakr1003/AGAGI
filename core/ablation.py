"""
core/ablation.py
----------------
Switches that turn off one AGAGI component at a time, for ablation experiments.

Set AGAGI_ABLATIONS to a comma-separated list of names, for example
  AGAGI_ABLATIONS=no_triage,no_feedback
Normal use leaves it unset, and every component runs.

  no_differential    - skip the base-branch test run; every failing test counts as a regression
  no_diff_scan       - keep scanner findings that already exist on the base branch
  no_triage          - keep every scanner finding as real without asking the LLM
  no_feedback        - on a retry, RCA and the fix agent do not see why the last patch failed
  no_impact_context  - RCA does not get the changed functions from the impact graph
"""

from __future__ import annotations

import os

ABLATIONS: dict[str, str] = {
    "no_differential": "No base-branch test run (all merged-tree failures are regressions)",
    "no_diff_scan": "No base-branch scan (pre-existing scanner findings are reported)",
    "no_triage": "No LLM triage (all scanner findings kept)",
    "no_feedback": "No verification feedback to RCA/fix on retries",
    "no_impact_context": "No impact-graph context in the RCA prompt",
}


def active_ablations() -> set[str]:
    raw = os.environ.get("AGAGI_ABLATIONS", "")
    names = {n.strip() for n in raw.split(",") if n.strip()}
    unknown = names - set(ABLATIONS)
    if unknown:
        raise ValueError(f"Unknown ablation(s): {', '.join(sorted(unknown))}. Known: {', '.join(ABLATIONS)}")
    return names


def ablation_active(name: str) -> bool:
    if name not in ABLATIONS:
        raise ValueError(f"Unknown ablation: {name}")
    return name in active_ablations()
