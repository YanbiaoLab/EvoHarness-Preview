"""S8 acceptance: protocol probe, GitWorkspace smoke, and DOA replay."""

from __future__ import annotations

import json

import pytest

import recipes
from evoharness.evocore import (
    LLMClient,
    LLMResponse,
    LLMStopReason,
    LLMToolCall,
    PopulationConfig,
    ProposalConfig,
    SearchConfig,
)
from evoharness.evoplus.config import PlusConfig
from experiments.run_evolution import main as run_evolution
from experiments.s8_live_smoke import _protocol_probe, _with_token_pricing
from experiments.s8_replay import run_replay
from recipes.common import RecipeContext
from tasks import get_task


@pytest.mark.parametrize("mode", ["single_shot", "conversational", "agentic"])
def test_gitworkspace_multifile_smoke_runs_all_three_arms(tmp_path, mode):
    task = get_task("s8_multifile")
    ctx = RecipeContext(
        search=SearchConfig(
            num_generations=1,
            operators=["rewrite"],
            operator_probs=[1.0],
            llm_models=["offline-model"],
            task_sys_msg=task.task_sys_msg,
        ),
        population=PopulationConfig(num_islands=1),
        plus=PlusConfig(),
        proposal=ProposalConfig(mode=mode, timeout_s=30),
        grader=task.grader,
        llm=LLMClient(transport=task.transport, sleep=lambda _: None),
        run_dir=tmp_path,
        preflight_validators=task.preflight_validators,
        runner=task.runner,
    )
    loop = recipes.get_recipe("e0").build(ctx)

    report = loop.run(
        task.initial_code,
        initial_workspace=task.initial_workspace,
    )

    assert report.best_fitness == 1.0
    generated = [
        candidate
        for candidate in loop.store.all_candidates()
        if candidate.generation == 1
    ]
    assert len(generated) == 1
    child = generated[0]
    assert child.workspace_kind == "git"
    assert child.workspace.texts()["math_ops.py"].count("value + 1") == 1
    assert child.workspace.texts()["metadata.py"] == 'VERSION = "1"\n'
    assert len(child.workspace.patches) == 1


def test_driver_passes_task_gitworkspace_seed_into_search_loop(tmp_path):
    run_dir = tmp_path / "driver"
    return_code = run_evolution(
        [
            "--recipe",
            "e0",
            "--task",
            "s8_multifile",
            "--run-dir",
            str(run_dir),
            "--set",
            "search.num_generations=1",
            "population.num_islands=1",
            'search.operators=["rewrite"]',
            "search.operator_probs=[1.0]",
            "proposal.mode=agentic",
            "proposal.timeout_s=30",
        ]
    )

    assert return_code == 0
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["report"]["best_fitness"] == 1.0
    patches = list((run_dir / "agent_sessions").glob("*/final.patch"))
    assert len(patches) == 1
    patch = patches[0].read_text()
    assert "math_ops.py" in patch
    assert "metadata.py" in patch


def test_structural_doa_replay_compares_three_arms(tmp_path):
    report = run_replay(tmp_path)
    arms = {item["arm"]: item for item in report["arms"]}

    # Four are structurally dead on arrival. The fifth returned its parent
    # unchanged, which is now rejected at proposal time rather than graded —
    # a no-op is not a broken candidate, it is not a candidate. The finding
    # the arm exists to show is unchanged: single-shot yields nothing usable.
    assert arms["single_shot"]["structural_doa"] == 4
    assert arms["single_shot"]["valid_candidates"] == 0
    for arm in ("conversational", "agentic"):
        assert arms[arm]["structural_doa"] == 0
        assert arms[arm]["valid_candidates"] == 5
        assert arms[arm]["preflight_interceptions"] == 5
        assert arms[arm]["self_repair_rate"] == 1.0
        assert arms[arm]["tool_pairing_complete"] is True
    assert arms["conversational"]["average_tool_calls"] == 0.0
    assert arms["agentic"]["average_tool_calls"] == 2.0

    controlled_results = [
        item
        for item in report["results"]
        if item["arm"] != "single_shot"
    ]
    assert all(
        item["observed_initial_issue"] == item["expected_issue"]
        for item in controlled_results
    )
    assert json.loads((tmp_path / "report.json").read_text()) == report


def test_live_protocol_probe_requires_call_id_roundtrip():
    histories = []

    def transport(messages, model, **kwargs):
        histories.append(messages)
        if len(histories) == 1:
            return LLMResponse(
                text="",
                model=model,
                cost=0.01,
                stop_reason=LLMStopReason.TOOL_CALLS,
                tool_calls=(
                    LLMToolCall(
                        "provider-call-1",
                        "record_probe",
                        {"token": "s8"},
                    ),
                ),
            )
        return LLMResponse("probe complete", model, cost=0.02)

    probe = _protocol_probe(
        LLMClient(transport=transport, sleep=lambda _: None),
        model="probe-model",
        timeout_s=10,
    )

    assert probe["call_id"] == "provider-call-1"
    assert probe["cost_usd"] == pytest.approx(0.03)
    assert [message.role for message in histories[1]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert histories[1][-1].tool_results[0].call_id == "provider-call-1"


def test_live_transport_pricing_uses_provider_token_counts():
    def transport(**kwargs):
        return LLMResponse(
            "done",
            "priced-model",
            prompt_tokens=2_000,
            completion_tokens=500,
        )

    priced = _with_token_pricing(
        transport,
        input_cost_per_million=2.0,
        output_cost_per_million=8.0,
    )
    response = priced()

    assert response.cost == pytest.approx(0.008)
    with pytest.raises(ValueError, match="nonnegative"):
        _with_token_pricing(
            transport,
            input_cost_per_million=-1,
            output_cost_per_million=0,
        )
