"""I-1 contracts: frozen identity, profile validation and compilation."""

from __future__ import annotations

import json

import pytest

from evoharness import (
    BasicSearchProfile,
    ComponentSpec,
    EvolutionSearchProfile,
    ProposalLimits,
    ResolvedTask,
    RunSpec,
    WorkspaceGradeFnGrader,
    compile_specs,
    run,
    spec_hashes,
)
from evoharness.core.workspace import FileWorkspace, GitWorkspace
from evoharness.core import LLMResponse, PreflightIssue, PreflightResult


def _task(workspace=None):
    return ResolvedTask.create(
        task_id="demo",
        version="v1",
        grader=WorkspaceGradeFnGrader(lambda _root, _ctx: 1.0),
        initial_workspace=workspace or FileWorkspace("x = 1\n"),
        domain_prompt="Improve x",
    )


def _run(tmp_path):
    return RunSpec(
        models=("model-a",),
        proposer_backend=ComponentSpec.create(
            "proposer_backend", "tests.fake", version="v1"
        ),
        output_dir=str(tmp_path),
        seed=7,
        proposal_limits=ProposalLimits(max_turns=4),
    )


def test_component_config_is_canonical():
    left = ComponentSpec.create(
        "grader", "demo.grader", config={"b": 2, "a": 1}
    )
    right = ComponentSpec.create(
        "grader", "demo.grader", config={"a": 1, "b": 2}
    )
    assert left.config_json == '{"a":1,"b":2}'
    assert left.hash == right.hash


def test_task_hash_covers_non_main_workspace_files():
    left = _task(GitWorkspace(base_files={"main.py": "x=1\n", "a.py": "v=1\n"}))
    right = _task(GitWorkspace(base_files={"main.py": "x=1\n", "a.py": "v=2\n"}))
    assert left.spec.hash != right.spec.hash
    json.dumps(left.spec.to_payload())


def test_git_workspace_identity_ignores_mapping_insertion_order():
    left = _task(GitWorkspace(base_files={"main.py": "x=1\n", "a.py": "v=1\n"}))
    right = _task(GitWorkspace(base_files={"a.py": "v=1\n", "main.py": "x=1\n"}))
    assert left.spec.initial_workspace.blob == right.spec.initial_workspace.blob
    assert left.spec.hash == right.spec.hash


def test_run_hash_covers_seed(tmp_path):
    first = _run(tmp_path)
    second = RunSpec(
        models=first.models,
        proposer_backend=first.proposer_backend,
        output_dir=first.output_dir,
        seed=8,
    )
    assert first.hash != second.hash


def test_basic_and_evolution_have_distinct_identities():
    basic = BasicSearchProfile(num_trajectories=3)
    evolution = EvolutionSearchProfile(num_generations=3)
    assert basic.hash != evolution.hash
    assert basic.to_payload()["kind"] == "basic"
    assert evolution.to_payload()["kind"] == "evolution"


def test_evolution_rejects_invalid_probabilities():
    with pytest.raises(ValueError, match="sum to 1"):
        EvolutionSearchProfile(operator_probs=(0.5, 0.3, 0.1))


def test_basic_compiles_to_seed_only_core(tmp_path):
    task = _task()
    run = _run(tmp_path)
    profile = BasicSearchProfile(
        num_trajectories=4,
        proposal_mode="single_shot",
    )
    search, population, proposal = compile_specs(task.spec, run, profile)
    assert search.num_generations == 4
    assert population.parent_strategy == "seed_only"
    assert proposal.mode == "single_shot"
    assert set(spec_hashes(task.spec, run, profile)) == {
        "task_hash", "run_hash", "search_hash"
    }


def test_public_run_executes_frozen_basic_search(tmp_path):
    from tasks.demo_counter import make_task

    task = make_task()
    run_spec = RunSpec(
        models=("mock-model",),
        proposer_backend=ComponentSpec.create(
            "proposer_backend", "tasks.demo_counter.fake", version="v1"
        ),
        output_dir=str(tmp_path / "run"),
        seed=3,
    )
    profile = BasicSearchProfile(
        num_trajectories=2,
        proposal_mode="single_shot",
    )
    report = run(task, run_spec, profile)
    assert report.generations_completed == 2
    assert report.evaluations == 3
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert manifest["spec_hashes"] == spec_hashes(
        task.spec, run_spec, profile
    )
    # 活性断言:证据生产必须接在真实 run 链上(P0 修复),
    # 每次评分(含种子)都留下一个信封。
    evidence_lines = (
        (tmp_path / "run" / "evidence.jsonl").read_text().splitlines()
    )
    assert len(evidence_lines) == report.evaluations


def test_public_run_applies_task_preflight_to_single_shot(tmp_path):
    class RejectingValidator:
        name = "domain-contract"

        def __init__(self):
            self.calls = 0

        def validate(self, ctx):
            self.calls += 1
            return PreflightResult(
                stage=self.name,
                issues=(
                    PreflightIssue(
                        validator=self.name,
                        code="forbidden",
                        message="candidate violates the domain contract",
                        repairable=False,
                    ),
                ),
            )

    validator = RejectingValidator()

    def transport(messages, model, **_kwargs):
        assert messages
        return LLMResponse(
            text=(
                "TITLE: change\nSUMMARY: change x\n"
                "```python\n"
                "# EDIT-REGION-BEGIN\n"
                "x = 2\n"
                "# EDIT-REGION-END\n"
                "```"
            ),
            model=model,
        )

    task = ResolvedTask.create(
        task_id="preflight-demo",
        version="v1",
        grader=WorkspaceGradeFnGrader(lambda _root, _ctx: 1.0),
        initial_workspace=FileWorkspace(
            "# EDIT-REGION-BEGIN\nx = 1\n# EDIT-REGION-END\n"
        ),
        preflight_validators=(validator,),
        default_transport=transport,
    )
    run_spec = _run(tmp_path / "preflight-run")
    profile = BasicSearchProfile(
        num_trajectories=1,
        proposal_mode="single_shot",
    )

    report = run(task, run_spec, profile)

    assert validator.calls > 0
    assert report.evaluations == 1  # seed only; rejected proposal is not graded
    assert any(
        item["status"] == "preflight_failed" for item in report.history
    )


def test_run_identity_ignores_output_dir(tmp_path):
    """物理输出位置是部署细节:同一冻结实验换目录重跑,run_hash 不变。"""
    a = _run(tmp_path / "a")
    b = _run(tmp_path / "b")
    assert a.output_dir != b.output_dir
    assert a.hash == b.hash


def test_declared_knowledge_reaches_the_prompt(tmp_path):
    """knowledge 参与 task_hash,就必须真实出现在变异 prompt 里——
    身份声明与实际注入不一致,是最隐蔽的实验条件失效。"""
    seen = []

    def transport(messages, model, **_kwargs):
        seen.extend(m.content for m in messages)
        return LLMResponse(
            text=(
                "TITLE: change\nSUMMARY: change x\n"
                "```python\n"
                "# EDIT-REGION-BEGIN\n"
                "x = 2\n"
                "# EDIT-REGION-END\n"
                "```"
            ),
            model=model,
        )

    task = ResolvedTask.create(
        task_id="knowledge-demo",
        version="v1",
        grader=WorkspaceGradeFnGrader(lambda _root, _ctx: 1.0),
        initial_workspace=FileWorkspace(
            "# EDIT-REGION-BEGIN\nx = 1\n# EDIT-REGION-END\n"
        ),
        knowledge=("KNOWLEDGE-MARKER-XYZ: prefer closed forms",),
        default_transport=transport,
    )
    run_spec = _run(tmp_path / "knowledge-run")
    profile = BasicSearchProfile(
        num_trajectories=1,
        proposal_mode="single_shot",
    )

    run(task, run_spec, profile)

    assert any("KNOWLEDGE-MARKER-XYZ" in content for content in seen)
