"""Proposer seam: SingleShotProposer behavior and loop injection."""

import pytest

from evoharness.core import (
    Candidate,
    LLMClient,
    LLMResponse,
    Proposal,
    ProposeResult,
    SingleShotProposer,
    StaticRouter,
)
from test_loop import INITIAL, build_loop, make_rewrite_transport


def make_parent(code):
    """Wrap raw code as a Candidate: propose() takes the parent, not a str."""
    return Candidate(id="p", code=code, generation=0, parent_id=None,
                     island_idx=0, operator="seed")

REWRITE_OK = (
    "TITLE: t\nSUMMARY: s\n```python\n"
    "# EDIT-REGION-BEGIN\nx = 0\nx += 1\nx += 1\n# EDIT-REGION-END\n```"
)


def make_proposer(transport, max_resamples=3):
    return SingleShotProposer(
        llm=LLMClient(transport=transport, sleep=lambda s: None),
        model_router=StaticRouter(["m"]),
        max_resamples=max_resamples,
    )


def test_single_shot_success_rewrite_and_revise():
    proposer = make_proposer(
        lambda messages, model, **kw: LLMResponse(REWRITE_OK, model, cost=0.01)
    )
    result = proposer.propose("rewrite", make_parent(INITIAL), "sys", "user")
    assert result.ok and result.attempts == 1
    assert result.llm_cost == 0.01
    assert "x += 1\nx += 1" in result.proposal.code
    assert result.proposal.title == "t" and result.proposal.model == "m"

    revise_text = (
        "TITLE: r\nSUMMARY: s\n"
        "<<<<<<< ORIGINAL\nx += 1\n=======\nx += 1\nx += 1\n>>>>>>> UPDATED\n"
    )
    proposer = make_proposer(
        lambda messages, model, **kw: LLMResponse(revise_text, model, cost=0.02)
    )
    result = proposer.propose("revise", make_parent(INITIAL), "sys", "user")
    assert result.ok
    assert result.proposal.code.count("x += 1") == 2


def test_single_shot_retries_and_accumulates_cost():
    calls = {"n": 0}

    def transport(messages, model, **kw):
        calls["n"] += 1
        if calls["n"] < 3:
            return LLMResponse("garbage", model, cost=0.01)
        return LLMResponse(REWRITE_OK, model, cost=0.01)

    result = make_proposer(transport).propose("rewrite", make_parent(INITIAL), "s", "u")
    assert result.ok and result.attempts == 3
    assert result.llm_cost == 0.03  # failed attempts still cost money


def test_single_shot_exhaustion_returns_cost():
    result = make_proposer(
        lambda messages, model, **kw: LLMResponse("junk", model, cost=0.01),
        max_resamples=2,
    ).propose("rewrite", make_parent(INITIAL), "s", "u")
    assert not result.ok and result.proposal is None
    assert result.llm_cost == 0.02 and result.attempts == 2
    assert result.failure_reason == "resample-exhausted"


def test_failed_propose_result_requires_failure_reason():
    with pytest.raises(ValueError, match="requires failure_reason"):
        ProposeResult(None)


def test_successful_propose_result_rejects_failure_reason():
    proposal = Proposal(
        code="x = 2\n",
        title="change",
        summary="updated x",
        model="model",
    )

    with pytest.raises(ValueError, match="cannot have failure_reason"):
        ProposeResult(
            proposal,
            failure_reason="backend-error",
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"llm_cost": -1},
        {"llm_cost": float("inf")},
        {"attempts": -1},
        {"attempts": True},
        {"trace_path": ""},
    ],
)
def test_propose_result_rejects_invalid_accounting(changes):
    values = {
        "proposal": None,
        "failure_reason": "test-failure",
    }
    values.update(changes)

    with pytest.raises(ValueError):
        ProposeResult(**values)


def test_single_shot_survives_llm_errors():
    calls = {"n": 0}

    def transport(messages, model, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("api down")  # LLMClient retries then raises
        return LLMResponse(REWRITE_OK, model, cost=0.01)

    result = make_proposer(transport).propose("rewrite", make_parent(INITIAL), "s", "u")
    # first attempt dies inside LLMClient retries; proposer resamples
    assert result.ok


def test_loop_accepts_custom_proposer(tmp_path):
    """The seam works end-to-end: a stub proposer drives the whole loop."""

    class StubProposer:
        def propose(self, operator, parent, system, user):
            code = parent.workspace.main_text().replace("x += 1", "x += 1\nx += 1", 1)
            return ProposeResult(
                Proposal(code, "stub", "doubles increments", "stub-model"),
                llm_cost=0.005,
                attempts=1,
            )

    loop, store = build_loop(
        make_rewrite_transport(), ["rewrite"], [1.0], tmp_path, generations=5
    )
    loop.proposer = StubProposer()
    report = loop.run(INITIAL)
    assert report.generations_completed == 5
    assert report.total_llm_cost == 0.025  # 5 proposals x 0.005, no real LLM
    assert all(
        c.model_name == "stub-model"
        for c in store.all_candidates()
        if c.generation > 0
    )


# -- M2.5 multi-file lane --------------------------------------------------------

from evoharness.core.workspace import GitWorkspace

GIT_FILE_BLOCKS = (
    "TITLE: add helper\nSUMMARY: split logic\n"
    "### FILE: main.py\n```python\nimport helper\nx = helper.n()\n```\n"
    "### FILE: helper.py\n```python\ndef n():\n    return 2\n```\n"
)


def make_git_parent():
    ws = GitWorkspace(base_files={"main.py": "x = 1\n", "util.py": "y = 2\n"})
    return Candidate(id="gp", code=ws.serialize(), generation=0, parent_id=None,
                     island_idx=0, operator="seed", workspace_kind="git")


def test_multi_file_lane_builds_child_workspace(tmp_path):
    proposer = make_proposer(
        lambda messages, model, **kw: LLMResponse(
            GIT_FILE_BLOCKS, model, cost=0.01
        )
    )
    result = proposer.propose("rewrite", make_git_parent(), "s", "u")
    assert result.ok
    ws = result.proposal.workspace
    assert isinstance(ws, GitWorkspace) and len(ws.patches) == 1
    root = ws.materialize(tmp_path / "c")
    assert (root / "helper.py").read_text() == "def n():\n    return 2\n"
    assert (root / "util.py").read_text() == "y = 2\n"  # untouched sibling
    # the Proposal.code invariant: ALWAYS the new main-file text
    assert result.proposal.code == "import helper\nx = helper.n()\n"


def test_multi_file_lane_rejects_escape_paths():
    evil = "TITLE: t\n### FILE: ../evil.py\n```python\nx = 1\n```\n"
    result = make_proposer(
        lambda messages, model, **kw: LLMResponse(evil, model, cost=0.01),
        max_resamples=2,
    ).propose("rewrite", make_git_parent(), "s", "u")
    assert not result.ok and result.attempts == 2  # rejected, resampled, exhausted
    assert result.llm_cost == 0.02  # rejected attempts still billed
