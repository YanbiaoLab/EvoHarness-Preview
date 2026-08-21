"""A prompt may only name a tool the reader actually has.

Agentic mode used to imply the in-process tool set, so one flag could decide
both "is this a tool-using session" and "can it fetch a peer candidate". An
external runtime broke that: `_build_proposer` clears every in-process tool
and the runtime brings its own, so the prompt kept naming `inspect_candidate`
and `workspace_read` at a candidate that had neither.

The peer case is the one that loses data rather than a name. Reference
programs live only in the population store, so the prompt renders an inventory
INSTEAD of the source on the promise that a tool can expand it. With no such
tool the candidate gets a list of programs it has no way to open.
"""

import pytest

from evoharness.core import Candidate, EvalReport, MutationContext
from evoharness.core.operators import PromptBuilder
from evoharness.core.workspace import GitWorkspace

PEER_SOURCE = "peer_marker_value = 41\n"


def make_candidate(cid, code):
    return Candidate(
        id=cid,
        code=GitWorkspace(
            base_files={"main.py": code}, main_file="main.py"
        ).serialize(),
        generation=1,
        parent_id=None,
        island_idx=0,
        operator="revise",
        workspace_kind="git",
        change_title=f"{cid} change",
        report=EvalReport(fitness=0.5, passed=True),
    )


@pytest.fixture
def ctx():
    return MutationContext(
        make_candidate("p1", "parent_marker = 1\n"),
        [make_candidate("i1", PEER_SOURCE)],
        [],
        "revise",
        3,
    )


def in_process():
    """How `recipes/common.py` configures the in-process agentic lane."""

    return PromptBuilder(
        "task",
        contributors=[],
        workspace_agent=True,
        peer_fetch_tool="inspect_candidate",
        workspace_read_tool="workspace_read",
    )


def external(peer_fetch_tool=None):
    """How it configures a lane driven by an external runtime."""

    return PromptBuilder(
        "task",
        contributors=[],
        workspace_agent=True,
        peer_fetch_tool=peer_fetch_tool,
        workspace_read_tool=None,
    )


def test_a_runtime_without_a_peer_tool_is_given_the_source(ctx):
    _, user = external().build(ctx)
    assert PEER_SOURCE.strip() in user
    assert "inspect_candidate" not in user


def test_a_runtime_without_a_peer_tool_is_not_sent_after_one(ctx):
    system, user = external().build(ctx)
    # Neither the reference-program header nor the recombine intent may name
    # a tool. The intent lives in the system message, the header in the user
    # message, and each was a separate hardcoded string.
    recombine_ctx = MutationContext(
        ctx.parent, list(ctx.archive_inspirations), [], "recombine", 3
    )
    r_system, r_user = external().build(recombine_ctx)
    for text in (system, user, r_system, r_user):
        assert "inspect_candidate" not in text
        assert "workspace_read" not in text
    assert "Study the reference programs above" in r_system


def test_a_runtime_with_its_own_peer_tool_is_told_that_name(ctx):
    _, user = external("evo_inspect_candidate").build(ctx)
    assert "evo_inspect_candidate(candidate_id, path)" in user
    # Given a way to fetch it, the source stays out of the prompt — that is
    # what the inventory buys.
    assert PEER_SOURCE.strip() not in user
    assert "id=i1" in user


def test_the_recombine_intent_follows_the_same_name(ctx):
    recombine_ctx = MutationContext(
        ctx.parent, list(ctx.archive_inspirations), [], "recombine", 3
    )
    system, _ = external("evo_inspect_candidate").build(recombine_ctx)
    assert "Read the reference with evo_inspect_candidate first" in system


def test_the_in_process_lane_is_unchanged(ctx):
    """The behaviour every existing agentic experiment ran under."""

    system, user = in_process().build(ctx)
    assert "Read any of them with inspect_candidate(candidate_id, path)." in user
    assert "Read them with workspace_read before editing." in user
    assert PEER_SOURCE.strip() not in user
    assert "id=i1" in user
    assert "This mutation: REVISE" in system


def test_the_single_shot_lane_is_unchanged(ctx):
    """No session, no tools, source inline — and no tool name anywhere."""

    system, user = PromptBuilder("task", contributors=[]).build(ctx)
    assert PEER_SOURCE.strip() in user
    for text in (system, user):
        assert "inspect_candidate" not in text
        assert "workspace_read" not in text


def test_naming_no_tool_is_the_safe_default(ctx):
    """A builder told nothing renders the source rather than promising a tool.

    The defaults decide what a caller that has not been updated does. Sending
    the source is merely long; naming a tool that may not exist is the failure
    this module is about.
    """

    _, user = PromptBuilder("task", contributors=[], workspace_agent=True).build(ctx)
    assert PEER_SOURCE.strip() in user
    assert "inspect_candidate" not in user


@pytest.mark.parametrize("field", ["peer_fetch_tool", "workspace_read_tool"])
def test_a_blank_tool_name_is_refused(field):
    with pytest.raises(ValueError):
        PromptBuilder("task", contributors=[], **{field: "  "})
