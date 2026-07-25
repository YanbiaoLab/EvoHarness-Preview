"""Liveness audit for a run directory.

Motivation (2026-07-24 postmortem): the novelty gate shipped unmounted for
months, and the bug inside it (judging main_text instead of the whole
workspace) stayed invisible precisely because nothing ever executed it.
Unit tests could not catch either: they asserted the mechanism was CORRECT,
never that it was RUNNING.

So this audits liveness per mechanism — did it actually fire in this run —
and flags mechanisms that are silently inert. Run it on any run dir:

    python scripts/audit.py results/e5s_s0
"""

import json
import os
import sqlite3
import sys
import time
from collections import Counter

OK, WARN, DEAD = "ok  ", "WARN", "DEAD"


def _load(path, default=None):
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return default


def _jsonl(path):
    try:
        with open(path) as handle:
            return [json.loads(x) for x in handle if x.strip()]
    except OSError:
        return []


def audit(run_dir):
    findings = []

    def report(status, mechanism, detail):
        findings.append((status, mechanism, detail))

    ckpt = _load(f"{run_dir}/checkpoint.json", {})
    rr = ckpt.get("run_report", {})
    manifest = _load(f"{run_dir}/experiment_manifest.json", {})
    arm = manifest.get("actual", {})
    rows = []
    if os.path.exists(f"{run_dir}/run.db"):
        con = sqlite3.connect(f"{run_dir}/run.db")
        rows = list(con.execute(
            "SELECT id, generation, island_idx, operator, in_archive, "
            "embedding IS NOT NULL, behavior_signature, inspiration_ids, "
            "json_extract(report,'$.fitness') FROM candidates"))
        con.close()

    n = len(rows)
    # Liveness in the wall-clock sense. A hung provider connection leaves
    # the process alive and silent: status output still says "running" and
    # every other check still passes, because the last checkpoint is
    # perfectly valid — it is just old. One live run sat like this for 2h04m.
    if rr.get("stopped_reason") == "running":
        newest = max(
            (os.path.getmtime(os.path.join(run_dir, f))
             for f in os.listdir(run_dir)
             if os.path.isfile(os.path.join(run_dir, f))),
            default=None,
        )
        if newest is not None:
            idle_min = (time.time() - newest) / 60
            report("STALL" if idle_min > 20 else OK, "progress",
                   f"last write {idle_min:.0f} min ago"
                   + (" — run appears hung, not slow" if idle_min > 20 else ""))

    version = manifest.get("code_version", "unknown")
    report("info", "run", f"{n} candidates, gen {ckpt.get('generation')}, "
           f"arm {arm.get('experience_mode')}, best {rr.get('best_fitness')}")
    report(WARN if version.endswith("+dirty") or version == "unknown" else OK,
           "code version", version[:12] + version[40:]
           + (" — uncommitted edits: this run is not reproducible"
              if version.endswith("+dirty") else ""))

    # Seeds never pass the gate, never enter the buffer and cannot fill a
    # reflection batch, so any verdict before the first offspring is noise.
    offspring = sum(1 for r in rows if r[1] > 0)
    started = offspring > 0

    # -- pre-evaluation rejection channel ------------------------------------
    embedded = sum(1 for r in rows if r[5])
    if not started:
        report("info", "novelty gate", "no offspring yet")
    elif embedded == 0 and n:
        report(DEAD, "novelty gate", "no candidate carries an embedding — "
               "gate not mounted, novelty rejections impossible")
    else:
        rej = rr.get("novelty_rejections", 0)
        rate = rej / max(1, rej + n)
        # Identity mode rejects only proposals that changed nothing, so a
        # low rate is healthy and zero is fine. A high rate means agent
        # sessions are being burned producing no-ops — or that a fuzzy
        # mode is rejecting legitimate small edits.
        report(WARN if rate > 0.15 else OK, "novelty gate",
               f"{embedded}/{n} embedded, {rej} rejections "
               f"({rate:.0%} of proposals)"
               + (" — high: sessions burned on no-ops, or the gate is"
                  " judging small edits as duplicates" if rate > 0.15 else ""))

    failed = rr.get("proposals_failed", 0)
    total_props = failed + n
    report(OK if failed / max(1, total_props) < 0.2 else WARN,
           "proposal success", f"{failed} failed of ~{total_props} proposals")

    # -- archive is supposed to be selective ---------------------------------
    archived = sum(1 for r in rows if r[4])
    if n and n <= 5:
        # Below the configured archive size everything legitimately fits.
        report("info", "archive", f"{archived}/{n} — too few to be selective")
    elif n and archived == n:
        report(WARN, "archive", f"{archived}/{n} — archive holds EVERYTHING, "
               "so 'archive inspiration' is just a random ancestor")
    elif n:
        report(OK, "archive", f"{archived}/{n} candidates retained")

    # -- behaviour signatures / duplicate detection --------------------------
    signed = sum(1 for r in rows if r[6])
    report(DEAD if signed == 0 and n else OK, "behavior signature",
           f"{signed}/{n} candidates signed"
           + (" — dedup filter in refresh_archive is a no-op" if not signed else ""))

    # -- experience buffer ----------------------------------------------------
    entries = _jsonl(f"{run_dir}/experience.jsonl")
    kinds = Counter(e.get("kind", "evaluated") for e in entries)
    report(OK if entries else (DEAD if started else "info"),
           "experience buffer", dict(kinds) or "no offspring yet")
    lessons = [e["lesson"] for e in entries if e.get("kind") == "lesson"]
    if arm.get("experience_mode", "").startswith("lessons"):
        if not lessons:
            pending = kinds.get("evaluated", 0)
            batch = arm.get("reflect_batch_size", 4)
            report(DEAD if pending >= batch else "info", "reflection",
                   f"no lesson rows yet ({pending} graded, batch {batch})")
        else:
            verdicts = Counter(x.get("verdict") for x in lessons)
            noise = verdicts.get("noise", 0) / len(lessons)
            report(WARN if noise > 0.5 else OK, "reflection",
                   f"{len(lessons)} lessons {dict(verdicts)} — "
                   f"{noise:.0%} noise"
                   + (" (evaluation too noisy to learn from)" if noise > 0.5 else ""))
            tags = Counter(t for x in lessons for t in x.get("tags", []))
            report(OK if len(tags) > 1 else WARN, "lesson tags",
                   f"{len(tags)} distinct: {dict(tags.most_common(6))}")

    for key, comp in (ckpt.get("components") or {}).items():
        if "RegressionSoftPenalty" in key:
            streaks = comp.get("streaks") or {}
            worst = max(streaks.values()) if streaks else 0
            report(WARN if worst >= 4 else OK, "soft gate",
                   f"{len(streaks)} parents on a regression streak, "
                   f"longest {worst}"
                   + (" — the loop is grinding on a dead parent"
                      if worst >= 4 else ""))
        if "Reflector" in key:
            pad = comp.get("scratchpad") or ""
            sections = sum(s in pad for s in
                           ("Successful", "Ineffective", "Unexplored"))
            report(OK if sections == 3 else WARN, "scratchpad",
                   f"v{comp.get('scratchpad_version')}, {len(pad)} chars, "
                   f"{sections}/3 sections, consolidated_at "
                   f"{comp.get('consolidated_at')}")
        if "Bandit" in key:
            report(OK, "operator bandit", comp.get("ema"))

    report(OK if any(e.get("redeemed_by") for e in entries) else "info",
           "redemption", f"{sum(1 for e in entries if e.get('redeemed_by'))} "
           "regressions redeemed by an improving child")

    # -- search dynamics ------------------------------------------------------
    ops = Counter(r[3] for r in rows if r[3] != "seed")
    report(OK if len(ops) > 1 else WARN, "operator spread", dict(ops))
    insp = [len(json.loads(r[7])) if r[7] else 0 for r in rows]
    report(OK if any(insp) else WARN, "inspirations",
           f"max {max(insp) if insp else 0} per candidate")
    islands = Counter(r[2] for r in rows)
    report(OK if len(islands) > 1 else WARN, "islands", dict(islands))

    # -- agent session economics ---------------------------------------------
    base = f"{run_dir}/agent_sessions"
    sessions, tools, compactions = [], Counter(), 0
    if os.path.isdir(base):
        for sid in os.listdir(base):
            summary = _load(f"{base}/{sid}/summary.json")
            if summary:
                sessions.append(summary)
            for event in _jsonl(f"{base}/{sid}/events.jsonl"):
                ev = event.get("event", {})
                if ev.get("kind") == "tool_call":
                    action = ev.get("data", {}).get("arguments", {}).get("action")
                    tools[f"{ev.get('tool_name')}:{action}" if action
                          else ev.get("tool_name")] += 1
                if ev.get("kind") == "context_compact":
                    compactions += 1
    if sessions:
        prompt_tokens = sum(s.get("prompt_tokens", 0) for s in sessions)
        turns = [s.get("turns", 0) for s in sessions]
        pinned = sum(1 for s in sessions
                     if s.get("termination") == "turn_limit")
        report(WARN if pinned else OK, "turn budget",
               f"turns {min(turns)}-{max(turns)}, {pinned} hit the limit")
        report(WARN if prompt_tokens / max(1, len(sessions)) > 150_000 else OK,
               "context cost",
               f"{prompt_tokens:,} prompt tokens over {len(sessions)} "
               f"sessions ({prompt_tokens // max(1, len(sessions)):,}/session)")
        report(OK if compactions else WARN, "context compaction",
               f"{compactions} compaction events")
        report("info", "tool usage", dict(tools.most_common(8)))

    # -- prompt injection liveness -------------------------------------------
    sections = Counter()
    if os.path.isdir(base):
        for sid in os.listdir(base):
            events = _jsonl(f"{base}/{sid}/events.jsonl")
            if not events:
                continue
            system = events[0].get("event", {}).get("data", {}).get("system", "")
            for label, marker in (
                ("lessons", "# Lessons from past mutations"),
                ("experience", "# Experience from this run"),
                ("rejected", "Recently ineffective or rejected"),
                ("direction", "# Direction hint"),
                ("directive", "# Mutation directive"),
                ("budget", "# Session budget"),
                ("failure", "# Parent failure analysis"),
            ):
                if marker in system:
                    sections[label] += 1
    report("info", "prompt sections", f"{dict(sections)} of {len(sessions)} sessions")

    # -- cost accounting ------------------------------------------------------
    report(WARN if not rr.get("total_llm_cost") else OK, "cost accounting",
           f"llm {rr.get('total_llm_cost')}, eval {rr.get('total_eval_cost')}"
           + (" — pricing unset, budget fuse inert" if not rr.get("total_llm_cost") else ""))

    return findings


if __name__ == "__main__":
    for target in sys.argv[1:] or ["results/e5s_s0"]:
        print(f"\n===== {target} =====")
        for status, mechanism, detail in audit(target):
            print(f"[{status}] {mechanism:<20} {detail}")
