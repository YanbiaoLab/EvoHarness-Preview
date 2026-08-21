"""Liveness audit for a run directory.

Motivation (2026-07-24 postmortem): the novelty gate shipped unmounted for
months, and the bug inside it (judging main_text instead of the whole
workspace) stayed invisible precisely because nothing ever executed it.
Unit tests could not catch either: they asserted the mechanism was CORRECT,
never that it was RUNNING.

So this audits liveness per mechanism — did it actually fire in this run —
and flags mechanisms that are silently inert. Run it on any run dir:

    python scripts/audit.py results/e5s_s0

Runs that belong to a research experiment can also be audited against the
research store, which is where routing and the human inbox leave their marks:

    python scripts/audit.py results/e5s_s0 --research research/
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


def _pending_requests(research_root, experiment_id):
    """(count, oldest created_at) of cards nobody has answered."""
    ledger = f"{research_root}/research.sqlite3"
    if not os.path.exists(ledger):
        return None
    con = sqlite3.connect(f"file:{ledger}?mode=ro", uri=True)
    try:
        row = con.execute(
            """
            SELECT COUNT(*), MIN(r.created_at)
            FROM decision_requests AS r
            LEFT JOIN research_decisions AS d ON d.request_id = r.request_id
            WHERE d.request_id IS NULL AND r.experiment_id = ?
            """,
            (experiment_id,),
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    finally:
        con.close()
    return int(row[0]), row[1]


def audit_research(run_dir, research_root, report):
    """Did this run reach the research layer at all, and is anyone answering?

    Same disease as the novelty gate: a router that is wired but never fires
    looks exactly like a router that was never wired. The run dir alone
    cannot tell them apart — the marks are in the research store.
    """
    ref = (_load(f"{run_dir}/manifest.json", {}) or {}).get("experiment_ref")
    if not ref:
        return                              # 不是实验的一部分,无路由可言
    experiment_id = ref.get("experiment_id", "")
    run_id = os.path.basename(str(run_dir).rstrip("/"))
    if not research_root:
        report(WARN, "research routing",
               f"run belongs to experiment {experiment_id} but no research "
               "root given — pass --research to audit it")
        return

    exp_dir = f"{research_root}/experiments/{experiment_id}"
    if not os.path.isdir(exp_dir):
        report(DEAD, "research routing",
               f"experiment {experiment_id} has no record in {research_root}")
        return

    routed = _jsonl(f"{exp_dir}/routing.jsonl")
    mine = [e for e in routed if e.get("run_id") == run_id]
    if not routed:
        report(DEAD, "research routing",
               "outcome recorded but nothing routed — router unmounted")
    elif not mine:
        report(DEAD, "research routing",
               f"{len(routed)} routed events, none from this run "
               f"({run_id}) — this run bypassed the router")
    else:
        report(OK, "research routing",
               f"{len(mine)} events this run: "
               f"{dict(Counter(e.get('disposition') for e in mine))}")

    # 断言评估留痕:评估过的 claim 是卡片和冲突通知的唯一来源,
    # 一条都没有意味着 route_assessment 从未被走到。
    assessments = _jsonl(f"{exp_dir}/assessments.jsonl")
    evidence = _jsonl(f"{run_dir}/evidence.jsonl")
    verdicts = sum(1 for e in evidence if e.get("objective_met") is not None)
    if verdicts and not assessments:
        report(DEAD, "research assessment",
               f"{verdicts} envelopes carry a verifier verdict but no claim "
               "was ever assessed")
    else:
        report(OK if assessments else "info", "research assessment",
               f"{len(assessments)} assessments recorded "
               f"({verdicts} verifier verdicts in this run)")

    pending = _pending_requests(research_root, experiment_id)
    if pending is None:
        report(DEAD, "research inbox", "no decision ledger — inbox unmounted")
        return
    count, oldest = pending
    stale_days = (time.time() - oldest) / 86400 if count and oldest else 0.0
    report(WARN if stale_days > 7 else ("info" if count else OK),
           "research inbox",
           f"{count} unanswered cards"
           + (f", oldest {stale_days:.0f} days" if count and oldest else "")
           + (" — decisions are the output; an unanswered queue means the "
              "loop stops at the human" if stale_days > 7 else ""))


def audit(run_dir, research_root=None):
    findings = []

    def report(status, mechanism, detail):
        findings.append((status, mechanism, detail))

    ckpt = _load(f"{run_dir}/checkpoint.json", {})
    rr = ckpt.get("run_report", {})
    # Two drivers, two filenames. `start_manifest` (the recipe path, and every
    # run started through evoharness.launch) writes manifest.json; the imo
    # driver writes experiment_manifest.json. Reading only the second meant
    # that on every mainline run this whole file was `{}` — so `code version`
    # read "unknown" forever and the manifest half of the evidence gate never
    # applied. An audit for silently-inert mechanisms, silently inert.
    manifest = (
        _load(f"{run_dir}/manifest.json")
        or _load(f"{run_dir}/experiment_manifest.json", {})
        or {}
    )
    arm = manifest.get("actual", {})
    rows = []
    if os.path.exists(f"{run_dir}/run.db"):
        con = sqlite3.connect(f"{run_dir}/run.db")
        rows = list(con.execute(
            "SELECT id, generation, island_idx, operator, in_archive, "
            "embedding IS NOT NULL, behavior_signature, inspiration_ids, "
            "json_extract(report,'$.fitness'), metadata FROM candidates"))
        con.close()

    n = len(rows)
    # Liveness in the wall-clock sense. A hung provider connection leaves
    # the process alive and silent: status output still says "running" and
    # every other check still passes, because the last checkpoint is
    # perfectly valid — it is just old. One live run sat like this for 2h04m.
    if rr.get("stopped_reason") == "running":
        # Walk the tree: the final evaluation phase writes only into
        # per-candidate subdirectories, so a root-only check reported a
        # two-hour stall on a run that was working the whole time.
        # Walk the tree AND the sibling log: the final evaluation phase
        # writes one file per candidate, minutes apart, so file mtimes alone
        # reported a stall on a run that was working the whole time. The log
        # is the only continuous liveness signal a hang actually silences.
        candidates = [
            os.path.getmtime(os.path.join(parent, f))
            for parent, _, files in os.walk(run_dir)
            for f in files
        ]
        log_path = str(run_dir).rstrip("/") + ".log"
        if os.path.isfile(log_path):
            candidates.append(os.path.getmtime(log_path))
        newest = max(candidates, default=None)
        if newest is not None:
            idle_min = (time.time() - newest) / 60
            report("STALL" if idle_min > 20 else OK, "progress",
                   f"last write {idle_min:.0f} min ago"
                   + (" — run appears hung, not slow" if idle_min > 20 else ""))

    # Same fact under two spellings: a `code_version` string from the imo
    # driver, a `code` object from `start_manifest`. Normalised to the string
    # so the dirty-tree warning fires on both.
    code = manifest.get("code") or {}
    version = manifest.get("code_version") or (
        f"{code['commit']}+dirty" if code.get("dirty")
        else code.get("commit", "unknown")
    )
    report("info", "run", f"{n} candidates, gen {ckpt.get('generation')}, "
           f"arm {arm.get('experience_mode')}, best {rr.get('best_fitness')}")
    report(WARN if version.endswith("+dirty") or version == "unknown" else OK,
           "code version", version[:12] + version[40:]
           + (" — uncommitted edits: this run is not reproducible"
              if version.endswith("+dirty") else ""))

    # Evidence production (research layer): every verdict and every infra
    # drop must leave an envelope. Zero lines = producer unmounted — dead
    # code wearing a green test suite. Legacy runs (no spec_hashes in the
    # manifest, no evidence file) predate the mechanism and are skipped.
    if manifest.get("spec_hashes") or os.path.exists(f"{run_dir}/evidence.jsonl"):
        evidence = _jsonl(f"{run_dir}/evidence.jsonl")
        infra_drops = sum(
            1 for h in rr.get("history", []) if h.get("status") == "infra_error"
        )
        expected = n + infra_drops
        if not evidence:
            report(DEAD, "evidence", "no envelopes (producer unmounted?)")
        elif len(evidence) < expected:
            report(WARN, "evidence",
                   f"{len(evidence)} envelopes < {expected} expected "
                   f"({n} candidates + {infra_drops} infra drops)")
        else:
            report(OK, "evidence",
                   f"{len(evidence)} envelopes ({infra_drops} infra)")

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
    # The verdict and the words have to agree. An empty buffer printed
    # "no offspring yet" whether or not there were any, so a DEAD line read
    # as "nothing to report yet" and was dismissed.
    report(OK if entries else (DEAD if started else "info"),
           "experience buffer",
           dict(kinds) if kinds
           else ("no entries despite offspring — producer unmounted?"
                 if started else "no offspring yet"))
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
        # A backend that cannot enforce the budget never terminates as
        # `turn_limit`; it runs past and the overrun is recorded on the
        # candidate. Counting only the termination reported "0 hit the limit"
        # for a run where every proposal had gone over.
        overran = sum(
            1 for row in rows
            if row[9] and json.loads(row[9]).get("limit_overruns")
        )
        report(WARN if pinned or overran else OK, "turn budget",
               f"turns {min(turns)}-{max(turns)}, {pinned} stopped at the "
               f"limit, {overran} ran past one the backend cannot enforce"
               + (" — the configured budget is advisory here, and timeout_s "
                  "is the bound that holds" if overran else ""))
        report(WARN if prompt_tokens / max(1, len(sessions)) > 150_000 else OK,
               "context cost",
               f"{prompt_tokens:,} prompt tokens over {len(sessions)} "
               f"sessions ({prompt_tokens // max(1, len(sessions)):,}/session)")
        report(OK if compactions else WARN, "context compaction",
               f"{compactions} compaction events")
        report("info", "tool usage", dict(tools.most_common(8)))

    # -- traceability: can each candidate be walked back to its session? ------
    #
    # An agentic candidate whose row carries no session id cannot be tied to
    # what the agent actually did. The trace is on disk either way, so nothing
    # errors and nothing looks wrong — the link is simply not there, and the
    # loss shows up much later, when a result is being explained.
    proposal = manifest.get("proposal", {})
    mode = proposal.get("mode")
    if mode in {"agentic", "hybrid", "conversational"} and offspring:
        linked = 0
        for row in rows:
            if row[1] == 0:
                continue  # seeds were not proposed by an agent
            meta = json.loads(row[9]) if row[9] else {}
            if meta.get("session_id") or meta.get("agent_session_id"):
                linked += 1
        # Hybrid routes only some proposals through a session, so a partial
        # link is the expected shape there and a total absence is not.
        floor = DEAD if linked == 0 else (
            OK if linked == offspring or mode == "hybrid" else WARN
        )
        report(floor, "traceability",
               f"{linked}/{offspring} agentic candidates carry a session id"
               + (" — no candidate can be walked back to what the agent did"
                  if linked == 0 else ""))

    # -- did the candidate's runtime really replace the in-process one? -------
    #
    # `_build_proposer` clears every in-process tool when a backend is
    # substituted, so a run under an external runtime that shows EvoHarness's
    # own tool names is one where the substitution did not happen. The
    # manifest saying which backend ran is a declaration; the tools the
    # sessions actually called are the evidence.
    backend = proposal.get("agent_backend")
    if backend and tools:
        declared = set(proposal.get("tools") or ())
        in_process = {"workspace_read", "workspace_write", "workspace_edit",
                      "inspect_candidate", "workspace_glob", "workspace_grep"}
        leaked = sorted(set(tools) & in_process)
        report(DEAD if leaked else OK, "runtime substitution",
               f"external backend, {len(declared)} in-process tools declared"
               + (f" — but sessions called {leaked}: the in-process runtime "
                  "was not replaced" if leaked else ""))

    # -- were the prompt's named tools actually available? -------------------
    #
    # The prompt offers reference programs as an inventory and names the tool
    # that expands one. Naming a tool the runtime does not mount costs the
    # source too, since the listing stood in for it. `prompt_tools` is
    # declared by the assembly; whether the model could call it is not.
    prompt_tools = proposal.get("prompt_tools") or {}
    peer_tool = prompt_tools.get("peer_fetch")
    if peer_tool and sessions:
        used = tools.get(peer_tool, 0)
        # Zero calls is not proof of absence — a run may simply never have had
        # a reference worth opening — so this warns rather than condemning.
        report(OK if used else WARN, "peer fetch",
               f"prompt named {peer_tool}, called {used} times"
               + (" — never exercised, so its availability is unverified"
                  if not used else ""))

    # -- prompt injection liveness -------------------------------------------
    sections = Counter()
    recorded_prompts = 0
    if os.path.isdir(base):
        for sid in os.listdir(base):
            events = _jsonl(f"{base}/{sid}/events.jsonl")
            if not events:
                continue
            system = events[0].get("event", {}).get("data", {}).get("system", "")
            if not system:
                continue
            recorded_prompts += 1
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
    # "Found no sections" and "could not look" print the same empty dict, and
    # they mean opposite things. A backend that records no system prompt makes
    # every prompt-side audit vacuous, so say that instead of an empty count.
    if sessions and not recorded_prompts:
        report(WARN, "prompt sections",
               "no session recorded its system prompt — prompt-side checks "
               "cannot run against this backend's trace")
    else:
        report("info", "prompt sections",
               f"{dict(sections)} of {recorded_prompts} sessions")

    # -- cost accounting ------------------------------------------------------
    report(WARN if not rr.get("total_llm_cost") else OK, "cost accounting",
           f"llm {rr.get('total_llm_cost')}, eval {rr.get('total_eval_cost')}"
           + (" — pricing unset, budget fuse inert" if not rr.get("total_llm_cost") else ""))

    audit_research(run_dir, research_root, report)

    return findings


if __name__ == "__main__":
    args = sys.argv[1:]
    research = None
    if "--research" in args:
        index = args.index("--research")
        research = args[index + 1]
        args = args[:index] + args[index + 2:]
    for target in args or ["results/e5s_s0"]:
        print(f"\n===== {target} =====")
        for status, mechanism, detail in audit(target, research):
            print(f"[{status}] {mechanism:<20} {detail}")
