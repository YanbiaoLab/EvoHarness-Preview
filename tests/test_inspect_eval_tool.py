import json

import pytest

from evoharness.core import Candidate, EvalReport
from evoharness.core.agent.tools import InspectParentEvalTool
from evoharness.core.agent.tools import AgentToolContext, AgentToolError
from evoharness.core.artifacts import FileArtifactStore
from evoharness.core.llm import LLMToolCall
from evoharness.core.preflight import PreflightPipeline, ProposalPreflight
from evoharness.core.sanitize import AllowlistSanitizer


def _sanitizer():
    return AllowlistSanitizer(
        item_allow={"optimizer": frozenset(
            {"item_id", "passed", "proof", "grader_critique"}
        )},
        summary_allow={"optimizer": frozenset({"points_percentage"})},
    )


def _parent_with_trace(tmp_path):
    store = FileArtifactStore(tmp_path / "artifacts")
    ref = store.put("pcand", {
        "summary": {"points_percentage": 0.5, "secret_total": 999},
        "items": [
            {"item_id": "P1", "passed": True, "proof": "ok"},
            {"item_id": "P2", "passed": False, "proof": "bad",
             "grader_critique": "缺垂心论证", "solution": "REFERENCE"},
        ],
    })
    parent = Candidate(
        id="pcand", code="x = 1", generation=0, parent_id=None,
        island_idx=0, operator="seed",
        report=EvalReport(fitness=0.5, passed=True, artifacts_ref=ref.encode()),
    )
    return store, parent


def _ctx(tmp_path, parent):
    return AgentToolContext(
        workdir=tmp_path,
        parent=parent,
        operator="revise",
        preflight=ProposalPreflight(PreflightPipeline(())),
        remaining_timeout_s=60.0,
    )


def _call(action, **kw):
    return LLMToolCall(call_id="c1", name="inspect_parent_eval",
                       arguments={"action": action, **kw})


def test_summary_lists_failed_items(tmp_path):
    store, parent = _parent_with_trace(tmp_path)
    tool = InspectParentEvalTool(store, _sanitizer())
    out = json.loads(tool.invoke(_call("summary"), _ctx(tmp_path, parent)).content)
    assert out["failed_items"] == ["P2"]
    assert out["summary"] == {"points_percentage": 0.5}   # secret_total stripped


def test_item_returns_critique_but_strips_reference_solution(tmp_path):
    store, parent = _parent_with_trace(tmp_path)
    tool = InspectParentEvalTool(store, _sanitizer())
    out = json.loads(tool.invoke(_call("item", item_id="P2"), _ctx(tmp_path, parent)).content)
    item = out["item"]
    assert item["grader_critique"] == "缺垂心论证"   # allowed
    assert "solution" not in item                    # NEVER leaks


def test_unknown_item_and_missing_trace_are_tool_errors(tmp_path):
    store, parent = _parent_with_trace(tmp_path)
    tool = InspectParentEvalTool(store, _sanitizer())
    with pytest.raises(AgentToolError):
        tool.invoke(_call("item", item_id="nope"), _ctx(tmp_path, parent))

    orphan = Candidate(
        id="o", code="x = 1", generation=0, parent_id=None, island_idx=0,
        operator="seed", report=EvalReport(fitness=0.0, passed=False),
    )
    with pytest.raises(AgentToolError):
        tool.invoke(_call("summary"), _ctx(tmp_path, orphan))


def test_failed_digest_returns_all_failures_in_one_call(tmp_path):
    store = FileArtifactStore(tmp_path / "artifacts")
    long_critique = "细节冗长的批语 " * 100  # far beyond digest budget
    ref = store.put("pcand", {
        "summary": {"points_percentage": 0.25},
        "items": [
            {"item_id": "P1", "passed": True, "proof": "ok"},
            {"item_id": "P2", "passed": False, "proof": "bad",
             "grader_critique": long_critique, "solution": "REFERENCE"},
            {"item_id": "P3", "passed": False, "proof": "worse",
             "grader_critique": "短批语"},
        ],
    })
    parent = Candidate(
        id="pcand", code="x = 1", generation=0, parent_id=None,
        island_idx=0, operator="seed",
        report=EvalReport(fitness=0.25, passed=True, artifacts_ref=ref.encode()),
    )
    tool = InspectParentEvalTool(store, _sanitizer(), digest_field_chars=120)
    out = json.loads(
        tool.invoke(_call("failed_digest"), _ctx(tmp_path, parent)).content
    )
    assert out["failed_count"] == 2
    ids = [d["item_id"] for d in out["failed_items"]]
    assert ids == ["P2", "P3"]                      # passing P1 excluded
    p2 = out["failed_items"][0]
    assert len(p2["grader_critique"]) <= 140        # truncated to digest budget
    assert "solution" not in p2                     # sanitizer still applies
    assert out["failed_items"][1]["grader_critique"] == "短批语"
