# EvoHarness original (webui_design.md): read-only data shaping for the
# console. Single source of truth: the run directory written by SearchLoop.
"""Run-directory readers for the evoweb console API."""

from __future__ import annotations

import json
import time
from pathlib import Path

from evoharness.core import MetricLog, PopulationStore


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _best_from_store(run_dir: Path) -> float | None:
    store = PopulationStore.open_readonly(run_dir / "run.db")
    best = store.best()
    store.close()
    return best.fitness if best else None


def list_runs(root: Path) -> list[dict]:
    runs = []
    for run_db in sorted(root.glob("*/run.db")):
        run_dir = run_db.parent
        ckpt = _read_json(run_dir / "checkpoint.json")
        manifest = _read_json(run_dir / "manifest.json")
        report = ckpt.get("run_report", {})
        if report.get("best_fitness") is None:  # pre-fix checkpoints
            report["best_fitness"] = _best_from_store(run_dir)
        mtime = checkpoint_mtime(run_dir)
        runs.append(
            {
                "name": run_dir.name,
                # A "running" run whose checkpoint stopped moving is likely a
                # dead process; the UI downgrades its live indicator.
                "heartbeat_age_s": round(time.time() - mtime, 1) if mtime else None,
                "recipe": manifest.get("recipe", "?"),
                "generation": ckpt.get("generation", 0),
                "target_generations": manifest.get("search", {}).get(
                    "num_generations"
                ),
                "stopped_reason": report.get("stopped_reason", "running"),
                "best_fitness": report.get("best_fitness"),
            }
        )
    return runs


def _candidate_row(c) -> dict:
    return {
        "id": c.id,
        "generation": c.generation,
        "parent_id": c.parent_id,
        "island": c.island_idx,
        "operator": c.operator,
        "fitness": c.fitness if c.report else None,
        "passed": c.passed,
        "title": c.change_title,
        "summary": c.change_summary,
        "model": c.model_name,
        "signature": c.behavior_signature,
        "duplicate": c.behavior_duplicate,
        "in_archive": c.in_archive,
        "fault": c.report.fault if c.report else None,
        "metadata": c.metadata or {},
    }


def _eval_series(run_dir: Path) -> dict[str, list]:
    """All task-emitted eval/* metric series, task-agnostic: whatever the
    grader put into visible/hidden metrics charts automatically."""
    path = run_dir / "metrics.jsonl"
    out: dict[str, list] = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        for key, value in d.get("metrics", {}).items():
            if key.startswith("eval/") and isinstance(value, (int, float)):
                out.setdefault(key[5:], []).append([d.get("step", 0), value])
    return out


def run_detail(run_dir: Path) -> dict:
    store = PopulationStore.open_readonly(run_dir / "run.db")
    candidates = store.all_candidates()
    store.close()
    metrics = MetricLog(run_dir / "metrics.jsonl")
    ckpt = _read_json(run_dir / "checkpoint.json")
    manifest = _read_json(run_dir / "manifest.json")
    report = ckpt.get("run_report", {})
    if report.get("best_fitness") is None:
        passed = [c for c in candidates if c.passed]
        if passed:
            report["best_fitness"] = max(c.fitness for c in passed)
    return {
        "name": run_dir.name,
        "recipe": manifest.get("recipe", "?"),
        "recipe_description": manifest.get("recipe_description", ""),
        "generation": ckpt.get("generation", 0),
        "target_generations": manifest.get("search", {}).get("num_generations"),
        "report": report,
        "candidates": [_candidate_row(c) for c in candidates],
        "series": {
            "best_fitness": metrics.series("sys/best_fitness"),
            "fitness": metrics.series("sys/fitness"),
        },
        "eval_series": _eval_series(run_dir),
        "manifest": manifest,      # task-details panel: configs, brief sha
        "distinct_signatures": len(
            {c.behavior_signature for c in candidates if c.behavior_signature}
        ),
    }


def candidate_detail(run_dir: Path, cand_id: str) -> dict | None:
    store = PopulationStore.open_readonly(run_dir / "run.db")
    cand = store.get(cand_id)
    store.close()
    if cand is None:
        return None
    row = _candidate_row(cand)
    # Display the program text, not the serialized genome (a git workspace
    # serializes to a JSON blob nobody wants to read in the UI).
    row["code"] = cand.workspace.main_text()
    row["report"] = cand.report.to_json() if cand.report else None
    return row


def checkpoint_mtime(run_dir: Path) -> float:
    path = run_dir / "checkpoint.json"
    return path.stat().st_mtime if path.exists() else 0.0


# ---- research copilot (docs/research_copilot_design.md §3) -----------------

def insights(run_dir: Path) -> dict:
    """Findings feed + proposal cards. Files are written by the copilot agent
    (or hand-authored golden samples); the console only reads and decides."""
    base = run_dir / "insights"
    findings = []
    fpath = base / "findings.jsonl"
    if fpath.exists():
        for line in fpath.read_text().splitlines():
            try:
                findings.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    proposals = []
    pdir = base / "proposals"
    if pdir.exists():
        for pf in sorted(pdir.glob("*.json")):
            try:
                proposals.append(json.loads(pf.read_text()))
            except json.JSONDecodeError:
                continue
    return {"findings": findings, "proposals": proposals}


def proposal_patch(run_dir: Path, pid: str) -> str | None:
    path = run_dir / "insights" / "proposals" / f"{pid}.patch"
    return path.read_text() if path.exists() else None


def decide_proposal(
    run_dir: Path, pid: str, action: str, reason: str | None = None
) -> dict | None:
    """State transition draft -> accepted/rejected, with the design-doc §6
    guardrails enforced MECHANICALLY (not just in the UI):
    - rejection requires a written reason (feeds the agent's memory);
    - an L2 proposal without version bumps cannot be accepted."""
    path = run_dir / "insights" / "proposals" / f"{pid}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    if d.get("status") != "draft":
        raise ValueError(f"proposal {pid} already {d.get('status')}")
    if action == "reject":
        if not (reason or "").strip():
            raise ValueError("rejection requires a reason (it feeds agent memory)")
        d["status"] = "rejected"
        d["rejection_reason"] = reason.strip()
    elif action == "accept":
        if d.get("level") == "L2" and not d.get("version_bumps"):
            raise ValueError(
                "L2 proposal without version bumps cannot be accepted "
                "(task_version discipline, design doc §2)"
            )
        d["status"] = "accepted"
    else:
        raise ValueError(f"unknown action {action!r}")
    path.write_text(json.dumps(d, ensure_ascii=False, indent=2))
    return d
