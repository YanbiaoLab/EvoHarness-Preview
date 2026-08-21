"""活性纪律:agentic 那条 lane 的机制到底跑没跑。

单元测试断言的是"机制对不对",这里断言的是"机制在不在跑"。这一轮真实
接入里挖出来的几个缺陷——守卫判据对候选不生效、提示词点名运行时没有的
工具、单发模式配了 dsh 却被静默忽略——**单元测试全程是绿的**,因为它们
断言的东西根本没被执行到。

每个用例都配一个"活着"的对照:只断言缺失会被报出来,不断言存在会被认可,
那么一个永远返回 DEAD 的实现也能通过。
"""

import json
import sqlite3

from scripts.audit import audit

SESSION_TOOLS = ("bash", "str_replace_editor")


def _db(run_dir, rows):
    con = sqlite3.connect(run_dir / "run.db")
    con.execute(
        "CREATE TABLE candidates (id TEXT, generation INT, island_idx INT, "
        "operator TEXT, in_archive INT, embedding BLOB, "
        "behavior_signature TEXT, inspiration_ids TEXT, report TEXT, "
        "metadata TEXT)"
    )
    con.executemany(
        "INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?)", rows
    )
    con.commit()
    con.close()


def _candidate(cid, generation, *, session_id=None):
    metadata = {} if session_id is None else {"session_id": session_id}
    return (
        cid, generation, 0, "revise" if generation else "seed", 1, b"e",
        "sig", "[]", json.dumps({"fitness": 0.5}), json.dumps(metadata),
    )


def _session(run_dir, sid, *, tools=SESSION_TOOLS, system=None):
    directory = run_dir / "agent_sessions" / sid
    directory.mkdir(parents=True)
    (directory / "summary.json").write_text(
        json.dumps({"turns": 4, "prompt_tokens": 100, "termination": "completed"})
    )
    events = [
        {"event": {"kind": "session_start", "data":
                   {} if system is None else {"system": system}}},
    ]
    events += [
        {"event": {"kind": "tool_call", "tool_name": name, "data": {}}}
        for name in tools
    ]
    (directory / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n"
    )


def _run(tmp_path, *, proposal, candidates, sessions=(), name="run"):
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({
        "code": {"commit": "a" * 40, "branch": "dev", "dirty": False},
        "spec_hashes": {"run_hash": "r"},
        "proposal": proposal,
    }))
    (run_dir / "checkpoint.json").write_text(json.dumps({
        "generation": 1, "run_report": {"stopped_reason": "completed"},
    }))
    _db(run_dir, candidates)
    for spec in sessions:
        _session(run_dir, **spec)
    return run_dir


def _findings(run_dir):
    found = []
    audit_result = audit(str(run_dir))
    for status, mechanism, detail in audit_result:
        found.append((status, mechanism, detail))
    return {mechanism: (status, detail) for status, mechanism, detail in found}


AGENTIC = {"mode": "agentic", "tools": [], "agent_backend": {"class": "Dsh"}}


def test_an_agentic_candidate_without_a_session_id_is_dead(tmp_path):
    """The trace is on disk either way, so nothing errors — the link is just
    absent, and the loss surfaces when a result is being explained."""

    run_dir = _run(
        tmp_path,
        proposal=AGENTIC,
        candidates=[_candidate("c0", 0), _candidate("c1", 1)],
        sessions=[{"sid": "s1"}],
    )
    status, detail = _findings(run_dir)["traceability"]
    assert status == "DEAD"
    assert "walked back" in detail


def test_a_linked_agentic_candidate_passes(tmp_path):
    """The liveness control. Without it an implementation that always says
    DEAD would satisfy the case above."""

    run_dir = _run(
        tmp_path,
        proposal=AGENTIC,
        candidates=[_candidate("c0", 0), _candidate("c1", 1, session_id="s1")],
        sessions=[{"sid": "s1"}],
    )
    status, detail = _findings(run_dir)["traceability"]
    assert status.strip() == "ok"
    assert "1/1" in detail


def test_a_single_shot_run_is_not_asked_for_session_ids(tmp_path):
    """That lane opens no session, so a missing id is not a defect there."""

    run_dir = _run(
        tmp_path,
        proposal={"mode": "single_shot", "tools": []},
        candidates=[_candidate("c0", 0), _candidate("c1", 1)],
    )
    assert "traceability" not in _findings(run_dir)


def test_in_process_tool_names_under_an_external_backend_are_dead(tmp_path):
    """Substituting a backend clears every in-process tool, so a session that
    called one is a run where the substitution did not happen. The manifest
    naming the backend is a declaration; the calls are the evidence."""

    run_dir = _run(
        tmp_path,
        proposal=AGENTIC,
        candidates=[_candidate("c1", 1, session_id="s1")],
        sessions=[{"sid": "s1", "tools": ("workspace_read", "bash")}],
    )
    status, detail = _findings(run_dir)["runtime substitution"]
    assert status == "DEAD"
    assert "workspace_read" in detail


def test_an_external_backend_calling_only_its_own_tools_passes(tmp_path):
    run_dir = _run(
        tmp_path,
        proposal=AGENTIC,
        candidates=[_candidate("c1", 1, session_id="s1")],
        sessions=[{"sid": "s1"}],
    )
    assert _findings(run_dir)["runtime substitution"][0].strip() == "ok"


def test_a_prompt_tool_the_model_never_called_is_unverified(tmp_path):
    """Warns rather than condemns: a run may simply never have had a
    reference worth opening."""

    run_dir = _run(
        tmp_path,
        proposal={**AGENTIC, "prompt_tools": {"peer_fetch": "evo_inspect_candidate"}},
        candidates=[_candidate("c1", 1, session_id="s1")],
        sessions=[{"sid": "s1"}],
    )
    status, detail = _findings(run_dir)["peer fetch"]
    assert status == "WARN"
    assert "unverified" in detail


def test_a_prompt_tool_the_model_did_call_passes(tmp_path):
    run_dir = _run(
        tmp_path,
        proposal={**AGENTIC, "prompt_tools": {"peer_fetch": "evo_inspect_candidate"}},
        candidates=[_candidate("c1", 1, session_id="s1")],
        sessions=[{"sid": "s1", "tools": ("bash", "evo_inspect_candidate")}],
    )
    status, detail = _findings(run_dir)["peer fetch"]
    assert status.strip() == "ok"
    assert "called 1 times" in detail


def test_a_backend_that_records_no_system_prompt_says_so(tmp_path):
    """"Found no sections" and "could not look" print the same empty dict and
    mean opposite things."""

    run_dir = _run(
        tmp_path,
        proposal=AGENTIC,
        candidates=[_candidate("c1", 1, session_id="s1")],
        sessions=[{"sid": "s1"}],
    )
    status, detail = _findings(run_dir)["prompt sections"]
    assert status == "WARN"
    assert "cannot run" in detail


def test_a_recorded_system_prompt_is_actually_inspected(tmp_path):
    run_dir = _run(
        tmp_path,
        proposal=AGENTIC,
        candidates=[_candidate("c1", 1, session_id="s1")],
        sessions=[{"sid": "s1", "system": "# Direction hint\ngo left"}],
    )
    status, detail = _findings(run_dir)["prompt sections"]
    assert status == "info"
    assert "direction" in detail


def test_the_mainline_manifest_filename_is_read(tmp_path):
    """Two drivers write two filenames. Reading only the other one made every
    manifest-backed check vacuous on every run started through the launcher."""

    run_dir = _run(
        tmp_path,
        proposal=AGENTIC,
        candidates=[_candidate("c1", 1, session_id="s1")],
    )
    status, detail = _findings(run_dir)["code version"]
    assert status.strip() == "ok"
    assert detail.startswith("aaaa")


def test_a_dirty_tree_is_warned_about_under_either_spelling(tmp_path):
    run_dir = _run(
        tmp_path,
        proposal=AGENTIC,
        candidates=[_candidate("c1", 1, session_id="s1")],
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["code"]["dirty"] = True
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    status, detail = _findings(run_dir)["code version"]
    assert status == "WARN"
    assert "not reproducible" in detail


def test_a_budget_the_backend_ran_past_is_reported(tmp_path):
    """A backend that cannot enforce the budget never terminates as
    `turn_limit` — it runs past and the overrun lands on the candidate.
    Counting only the termination said "0 hit the limit" for a run where
    every proposal had gone over."""

    over = _candidate("c1", 1, session_id="s1")
    metadata = json.loads(over[9]) | {"limit_overruns": {"max_turns": 7}}
    run_dir = _run(
        tmp_path,
        proposal=AGENTIC,
        candidates=[over[:9] + (json.dumps(metadata),)],
        sessions=[{"sid": "s1"}],
    )
    status, detail = _findings(run_dir)["turn budget"]
    assert status == "WARN"
    assert "ran past" in detail
    assert "advisory" in detail


def test_a_run_that_stayed_inside_its_budget_is_not_warned_about(tmp_path):
    run_dir = _run(
        tmp_path,
        proposal=AGENTIC,
        candidates=[_candidate("c1", 1, session_id="s1")],
        sessions=[{"sid": "s1"}],
    )
    assert _findings(run_dir)["turn budget"][0].strip() == "ok"


def test_an_empty_experience_buffer_says_which_emptiness_it_is(tmp_path):
    """The verdict and the words have to agree: DEAD next to "no offspring
    yet" reads as "nothing to report" and gets dismissed."""

    run_dir = _run(
        tmp_path,
        proposal=AGENTIC,
        candidates=[_candidate("c0", 0), _candidate("c1", 1, session_id="s1")],
    )
    status, detail = _findings(run_dir)["experience buffer"]
    assert status == "DEAD"
    assert "despite offspring" in detail
