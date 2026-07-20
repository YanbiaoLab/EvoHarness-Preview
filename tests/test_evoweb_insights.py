"""Research-copilot data layer: insights reader + proposal decision guardrails
(docs/research_copilot_design.md §3/§6)."""

import json

import pytest

from evoharness.evoweb.data import decide_proposal, insights, proposal_patch


def make_insights(tmp_path, proposal):
    base = tmp_path / "insights"
    (base / "proposals").mkdir(parents=True)
    (base / "findings.jsonl").write_text(
        json.dumps({"id": "F-1", "claim": "x", "severity": "observation",
                    "evidence": [], "proposals": []}) + "\n"
    )
    (base / "proposals" / f"{proposal['id']}.json").write_text(
        json.dumps(proposal)
    )
    (base / "proposals" / f"{proposal['id']}.patch").write_text("--- a\n+++ b\n")
    return tmp_path


def l1(status="draft"):
    return {"id": "P-1", "level": "L1", "title": "t", "motivation": ["F-1"],
            "status": status}


def test_reader_and_patch(tmp_path):
    run = make_insights(tmp_path, l1())
    data = insights(run)
    assert len(data["findings"]) == 1 and len(data["proposals"]) == 1
    assert proposal_patch(run, "P-1").startswith("---")
    assert proposal_patch(run, "P-9") is None
    assert insights(tmp_path / "nowhere") == {"findings": [], "proposals": []}


def test_accept_and_persistence(tmp_path):
    run = make_insights(tmp_path, l1())
    updated = decide_proposal(run, "P-1", "accept")
    assert updated["status"] == "accepted"
    assert insights(run)["proposals"][0]["status"] == "accepted"   # persisted


def test_rejection_requires_reason(tmp_path):
    run = make_insights(tmp_path, l1())
    with pytest.raises(ValueError, match="reason"):
        decide_proposal(run, "P-1", "reject", "  ")
    updated = decide_proposal(run, "P-1", "reject", "复评方差未测,先不动")
    assert updated["rejection_reason"] == "复评方差未测,先不动"


def test_l2_without_version_bump_cannot_be_accepted(tmp_path):
    p = {"id": "P-2", "level": "L2", "title": "t", "motivation": ["F-1"],
         "version_bumps": None, "status": "draft"}
    run = make_insights(tmp_path, p)
    with pytest.raises(ValueError, match="version bumps"):
        decide_proposal(run, "P-2", "accept")
    # ...but rejection is always allowed
    assert decide_proposal(run, "P-2", "reject", "no bump")["status"] == "rejected"


def test_decided_proposal_is_immutable(tmp_path):
    run = make_insights(tmp_path, l1(status="accepted"))
    with pytest.raises(ValueError, match="already"):
        decide_proposal(run, "P-1", "reject", "changed my mind")


def test_unknown_proposal_returns_none(tmp_path):
    run = make_insights(tmp_path, l1())
    assert decide_proposal(run, "P-404", "accept") is None
