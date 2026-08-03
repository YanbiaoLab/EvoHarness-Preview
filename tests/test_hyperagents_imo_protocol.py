"""Offline contract tests for the HyperAgents IMO comparison."""

import csv
import json

import pytest

from experiments.hyperagents_imo.protocol import ComparisonContract, verify_contract
from experiments.hyperagents_imo.scoring import (
    _split_summary,
    load_evoharness_proof,
    load_hyperagents_proof,
    load_hyperagents_predictions,
)
from experiments.hyperagents_imo.native_evolution import _select_parent
from experiments.hyperagents_imo.native_evolution import _ensure_runtime_gitignore
from experiments.hyperagents_imo.native_evolution import _ensure_editor_tool_aliases


def test_hyperagents_imo_contract_matches_the_frozen_baseline():
    try:
        contract = ComparisonContract.load()
    except FileNotFoundError as exc:
        pytest.skip(f"reference run artifact is not available: {exc}")
    result = verify_contract(contract)

    assert contract.candidate_budget == 5
    assert len(contract.train_ids) == 12
    assert result["ok"] is True
    assert result["not_a_live_run"] is True


def test_pair_scoring_loads_the_two_proof_formats(tmp_path):
    hyperagents_csv = tmp_path / "predictions.csv"
    with hyperagents_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("Problem ID", "prediction"))
        writer.writeheader()
        writer.writerow({"Problem ID": "PB-Basic-001", "prediction": "hyper proof"})

    evoharness_item = tmp_path / "item.json"
    evoharness_item.write_text(
        json.dumps({"item_id": "PB-Basic-001", "proof": "evo proof"}),
        encoding="utf-8",
    )

    assert (
        load_hyperagents_proof(hyperagents_csv, "PB-Basic-001") == "hyper proof"
    )
    assert (
        load_evoharness_proof(evoharness_item, "PB-Basic-001") == "evo proof"
    )
    assert load_hyperagents_predictions(hyperagents_csv) == {
        "PB-Basic-001": "hyper proof"
    }


def test_full_scoring_summary_uses_points_and_exact_correctness():
    results = {
        "a": {"label": "correct", "points": 7, "max_points": 7},
        "b": {"label": "almost", "points": 6, "max_points": 7},
        "c": {"label": "incorrect", "points": 0, "max_points": 7},
    }

    summary = _split_summary(("a", "b", "c"), results)

    assert summary == {
        "problem_count": 3,
        "points": 13,
        "max_points": 21,
        "points_percentage": 13 / 21,
        "correct_count": 1,
        "correct_percentage": 1 / 3,
        "label_counts": {"almost": 1, "correct": 1, "incorrect": 1},
    }


def test_native_parent_selection_is_deterministic_and_returns_valid_candidate():
    candidates = [
        {
            "candidate_id": "initial",
            "parent_id": None,
            "train_points_percentage": 0.25,
            "valid": True,
        },
        {
            "candidate_id": "001",
            "parent_id": "initial",
            "train_points_percentage": 0.5,
            "valid": True,
        },
        {
            "candidate_id": "002",
            "parent_id": "001",
            "train_points_percentage": 1.0,
            "valid": False,
        },
    ]

    first = _select_parent(candidates, generation=3, seed=7)
    second = _select_parent(candidates, generation=3, seed=7)

    assert first == second
    assert first in {"initial", "001"}


def test_native_workspace_ignores_runtime_bytecode(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    _ensure_runtime_gitignore(workspace)

    assert "__pycache__/" in (workspace / ".gitignore").read_text()
    assert "*.py[cod]" in (workspace / ".gitignore").read_text()


def test_native_workspace_normalizes_direct_editor_tool_names(tmp_path):
    workspace = tmp_path / "workspace"
    tool_file = workspace / "agent" / "llm_withtools.py"
    tool_file.parent.mkdir(parents=True)
    tool_file.write_text(
        "def process_tool_call(tools_dict, tool_name, tool_input):\n"
        "    return tools_dict[tool_name](**tool_input)\n"
    )

    _ensure_editor_tool_aliases(workspace)
    source = tool_file.read_text()

    assert 'tool_name in {"view", "create", "str_replace", "insert"}' in source
    assert 'tool_name = "editor"' in source
