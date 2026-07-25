"""Recipe pattern: registry, config loading, and the E0->E3r ladder built
against the offline demo task."""

import json

import pytest

import recipes
from evoharness.evocore import (
    AgentSessionProposer,
    HybridProposalSelector,
    LLMClient,
    PopulationConfig,
    ProposalConfig,
    SearchConfig,
    SingleShotProposer,
)
from evoharness.evocore.agent import ConversationalAgentBackend
from evoharness.evoplus.config import PlusConfig
from recipes import load_experiment_config
from recipes.common import RecipeContext, assemble
from tasks import get_task
from experiments.run_evolution import main as run_evolution


def test_registry_covers_experiment_matrix():
    assert set(recipes.REGISTRY) == {"b0", "e0", "e1", "e2", "e3g", "e3r"}
    assert "retriev" in recipes.get_recipe("E3R").DESCRIPTION
    with pytest.raises(ValueError, match="unknown recipe"):
        recipes.get_recipe("e99")


def test_config_loading_and_overrides(tmp_path):
    recipe_yaml = tmp_path / "hyper.yaml"
    recipe_yaml.write_text(
        "search:\n  num_generations: 30\nplus:\n  hamming_threshold: 1\n"
    )
    search, population, plus, proposal = load_experiment_config(
        recipe_yaml,
        [
            "search.seed=5",
            "population.num_islands=3",
            "plus.duplicate_penalty=0.5",
            "proposal.mode=conversational",
        ],
    )
    assert search.num_generations == 30 and search.seed == 5
    assert population.num_islands == 3
    assert plus.hamming_threshold == 1 and plus.duplicate_penalty == 0.5
    assert proposal.mode == "conversational"

    with pytest.raises(ValueError, match="unknown SearchConfig keys"):
        load_experiment_config(None, ["search.nope=1"])
    with pytest.raises(ValueError, match="not of form"):
        load_experiment_config(None, ["searchseed"])


def _ctx(tmp_path, task, generations=8, seed=3):
    return RecipeContext(
        search=SearchConfig(
            num_generations=generations,
            operators=["rewrite"],
            operator_probs=[1.0],
            # demo mock transports emit duplicate programs by design;
            # the novelty gate would legitimately reject them all.
            novelty_enabled=False,
            seed=seed,
            task_sys_msg=task.task_sys_msg,
        ),
        population=PopulationConfig(num_islands=1),
        plus=PlusConfig(),
        grader=task.grader,
        llm=LLMClient(transport=task.transport, sleep=lambda s: None),
        run_dir=tmp_path,
    )


@pytest.mark.parametrize(
    ("mode", "proposer_type"),
    [
        ("single_shot", SingleShotProposer),
        ("conversational", AgentSessionProposer),
        ("agentic", AgentSessionProposer),
    ],
)
def test_three_proposal_arms_build_and_run(
    tmp_path,
    mode,
    proposer_type,
):
    task = get_task("demo_counter")
    ctx = _ctx(tmp_path / mode, task, generations=2)
    ctx.proposal = ProposalConfig(
        mode=mode,
        max_turns=6,
        timeout_s=30,
    )
    loop = recipes.get_recipe("e0").build(ctx)

    assert isinstance(loop.proposer, proposer_type)
    assert loop.prompt_builder.workspace_agent is (mode == "agentic")
    if mode == "conversational":
        assert isinstance(
            loop.proposer.backend,
            ConversationalAgentBackend,
        )
        assert ctx.extras["proposal_manifest"]["tools"] == []
    elif mode == "agentic":
        assert "run" in ctx.extras["proposal_manifest"]["tools"]

    report = loop.run(task.initial_code)

    assert report.generations_completed == 2
    assert report.total_llm_cost > 0
    assert report.best_fitness is not None
    assert ctx.extras["proposal_manifest"]["mode"] == mode


@pytest.mark.parametrize("mode", ["single_shot", "conversational", "agentic"])
def test_driver_manifest_records_effective_proposal_arm_and_budget(
    tmp_path,
    mode,
):
    run_dir = tmp_path / mode
    return_code = run_evolution(
        [
            "--recipe",
            "e0",
            "--task",
            "demo_counter",
            "--run-dir",
            str(run_dir),
            "--budget-usd",
            "1.0",
            "--set",
            "search.num_generations=1",
            "population.num_islands=1",
            'search.operators=["rewrite"]',
            "search.operator_probs=[1.0]",
            f"proposal.mode={mode}",
            "proposal.timeout_s=30",
        ]
    )

    assert return_code == 0
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["proposal"]["mode"] == mode
    assert manifest["budget"]["hard_cap_usd"] == 1.0
    assert manifest["budget"]["spent_usd"] == pytest.approx(
        manifest["report"]["total_llm_cost"]
        + manifest["report"]["total_eval_cost"]
    )
    if mode == "single_shot":
        assert manifest["proposal"]["transcript"] is False
        assert manifest["proposal"]["tools"] == []
    else:
        assert manifest["proposal"]["transcript"] is True
        assert manifest["proposal"]["limits"]["timeout_s"] == 30
        summaries = list(
            (run_dir / "agent_sessions").glob("*/summary.json")
        )
        assert summaries
        transcript_cost = sum(
            json.loads(path.read_text())["cost_usd"]
            for path in summaries
        )
        assert transcript_cost == pytest.approx(
            manifest["report"]["total_llm_cost"]
        )


def test_driver_manifest_records_hybrid_routing_policy(tmp_path):
    run_dir = tmp_path / "hybrid"
    return_code = run_evolution(
        [
            "--recipe",
            "e0",
            "--task",
            "demo_counter",
            "--run-dir",
            str(run_dir),
            "--set",
            "search.num_generations=1",
            "population.num_islands=1",
            'search.operators=["rewrite"]',
            "search.operator_probs=[1.0]",
            "proposal.mode=hybrid",
            "proposal.hybrid_agent_probability=1.0",
            "proposal.hybrid_stagnation_generations=99",
        ]
    )

    assert return_code == 0
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["proposal"]["mode"] == "hybrid"
    assert manifest["proposal"]["transcript_scope"] == "agentic_routes"
    assert manifest["proposal"]["routing"] == {
        "agent_probability": 1.0,
        "stagnation_generations": 99,
    }
    assert manifest["budget"]["spent_usd"] == pytest.approx(
        manifest["report"]["total_llm_cost"]
        + manifest["report"]["total_eval_cost"]
    )


def test_agent_modes_require_an_explicit_model_for_multi_model_search(tmp_path):
    task = get_task("demo_counter")
    ctx = _ctx(tmp_path, task, generations=1)
    ctx.search.llm_models = ["model-a", "model-b"]
    ctx.proposal = ProposalConfig(mode="agentic")

    with pytest.raises(ValueError, match="proposal.model"):
        recipes.get_recipe("e0").build(ctx)

    ctx.proposal.model = "model-b"
    loop = recipes.get_recipe("e0").build(ctx)
    assert loop.proposer.backend.model == "model-b"


@pytest.mark.parametrize(
    ("agent_probability", "stagnation_generations", "expected_mode"),
    [
        (0.0, 99, "single_shot"),
        (1.0, 99, "agentic"),
        (0.0, 1, "agentic"),
    ],
)
def test_hybrid_routes_outside_agent_session_proposer(
    tmp_path,
    agent_probability,
    stagnation_generations,
    expected_mode,
):
    task = get_task("demo_counter")
    ctx = _ctx(tmp_path, task, generations=1)
    ctx.proposal = ProposalConfig(
        mode="hybrid",
        max_turns=6,
        timeout_s=30,
        hybrid_agent_probability=agent_probability,
        hybrid_stagnation_generations=stagnation_generations,
    )
    loop = recipes.get_recipe("e0").build(ctx)

    assert isinstance(loop.proposal_selector, HybridProposalSelector)
    assert isinstance(loop.proposer, SingleShotProposer)
    assert isinstance(
        loop.proposal_selector.agentic.proposer,
        AgentSessionProposer,
    )
    loop.run(task.initial_code)
    generated = [
        candidate
        for candidate in loop.store.all_candidates()
        if candidate.generation == 1
    ]

    assert len(generated) == 1
    assert generated[0].metadata["proposal_mode"] == expected_mode
    assert ctx.extras["proposal_manifest"]["mode"] == "hybrid"
    assert (
        ctx.extras["proposal_manifest"]["transcript_scope"]
        == "agentic_routes"
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"hybrid_agent_probability": -0.1},
        {"hybrid_agent_probability": 1.1},
        {"hybrid_stagnation_generations": 0},
    ],
)
def test_proposal_config_rejects_invalid_hybrid_policy(overrides):
    with pytest.raises(ValueError):
        ProposalConfig(mode="hybrid", **overrides)


def test_checkpoint_rejects_switching_proposal_arm(tmp_path):
    task = get_task("demo_counter")
    run_dir = tmp_path / "mixed-arm"
    first = _ctx(run_dir, task, generations=1)
    recipes.get_recipe("e0").build(first).run(task.initial_code)

    resumed = _ctx(run_dir, task, generations=1)
    resumed.proposal = ProposalConfig(mode="conversational")
    with pytest.raises(ValueError, match="different configuration"):
        recipes.get_recipe("e0").build(resumed).run(task.initial_code)


def test_checkpoint_rejects_switching_recipe_stack(tmp_path):
    task = get_task("demo_counter")
    run_dir = tmp_path / "mixed-recipe"
    first = _ctx(run_dir, task, generations=1)
    recipes.get_recipe("e0").build(first).run(task.initial_code)

    resumed = _ctx(run_dir, task, generations=1)
    with pytest.raises(ValueError, match="different configuration"):
        recipes.get_recipe("e1").build(resumed).run(task.initial_code)


@pytest.mark.parametrize("name", ["e0", "e1", "e2", "e3g", "e3r"])
def test_every_recipe_builds_and_runs(tmp_path, name):
    task = get_task("demo_counter")
    loop = recipes.get_recipe(name).build(_ctx(tmp_path / name, task))
    report = loop.run(task.initial_code)
    assert report.generations_completed == 8
    assert report.best_fitness > 0.2  # improved over the seed's 1/5


def test_b0_is_evaluate_once_no_evolution(tmp_path):
    task = get_task("demo_counter")
    loop = recipes.get_recipe("b0").build(_ctx(tmp_path, task))
    report = loop.run(task.initial_code)
    assert report.generations_completed == 0
    assert report.evaluations == 1  # the manual harness, graded once
    assert report.best_fitness == pytest.approx(0.2)


def test_ladder_diff_e0_vs_e1_is_feedback_injection(tmp_path):
    """The observable diff between E0 and E1 is exactly the C1 prompt
    section — verifying that recipe diffs correspond to behavior diffs."""
    seen = {}
    for name in ("e0", "e1"):
        task = get_task("demo_counter")
        captured = []
        inner = task.transport

        def spy(messages, model, _inner=inner, _cap=captured, **kw):
            _cap.append(messages[0].content)
            return _inner(messages=messages, model=model, **kw)

        task.transport = spy
        loop = recipes.get_recipe(name).build(_ctx(tmp_path / name, task))
        loop.llm = LLMClient(transport=spy, sleep=lambda s: None)
        loop.run(task.initial_code)
        seen[name] = any("Parent failure analysis" in s for s in captured)
    assert seen == {"e0": False, "e1": True}


def test_signatures_always_recorded_even_in_e0(tmp_path):
    task = get_task("demo_counter")
    loop = recipes.get_recipe("e0").build(_ctx(tmp_path, task))
    loop.run(task.initial_code)
    graded = [c for c in loop.store.all_candidates() if c.report]
    assert graded and all(c.behavior_signature for c in graded)
    # but the E0 intervention is off: nothing marked duplicate
    assert not any(c.behavior_duplicate for c in graded)


def test_e3r_extras_expose_stores(tmp_path):
    task = get_task("demo_counter")
    ctx = _ctx(tmp_path, task)
    loop = recipes.get_recipe("e3r").build(ctx)
    loop.run(task.initial_code)
    assert ctx.extras["experience_store"].entries
    assert (tmp_path / "experience.jsonl").exists()
    assert (tmp_path / "metrics.jsonl").exists()


def test_assemble_mounts_the_novelty_gate(tmp_path):
    """Regression: assemble() used to leave novelty_gate unset, so no
    candidate carried an embedding, novelty_rejections was pinned at 0 and
    the rejected_novelty experience channel could never fire."""
    import numpy as np

    from evoharness.evocore.novelty import (
        NoveltyGate,
        cosine_similarity,
        hashing_embedding,
    )

    task = get_task("demo_counter")
    ctx = _ctx(tmp_path, task, generations=1)
    ctx.search.novelty_enabled = True   # _ctx disables it for mock duplicates
    loop = assemble(ctx)
    assert isinstance(loop.novelty_gate, NoveltyGate)
    assert loop.novelty_gate.threshold == ctx.search.similarity_threshold

    # deterministic across processes (crc32, not salted str hashing)
    a = hashing_embedding("def solve(x):\n    return x + 1\n")
    assert a == hashing_embedding("def solve(x):\n    return x + 1\n")
    assert hashing_embedding("") == [0.0] * 512
    near = hashing_embedding("def solve(x):\n    return x + 2\n")
    far = hashing_embedding("class Wholly:\n    different = 'code'\n" * 4)
    assert cosine_similarity(np.array(a), np.array(near)) > cosine_similarity(
        np.array(a), np.array(far)
    )


def test_novelty_gate_identity_mode_accepts_small_edits(tmp_path):
    """A mutation is by construction almost character-identical to its own
    parent, and the parent sits in the same island — so a fuzzy gate at 0.99
    rejected 40% of good small edits live. Identity mode rejects only a
    proposal that adds nothing at all."""
    from evoharness.evocore import Candidate, EvalReport, IslandView
    from evoharness.evocore.novelty import NoveltyGate, hashing_embedding

    base = "def solve(x):\n    return x + 1\n" + "# filler\n" * 200
    existing = Candidate(
        id="c1", code=base, generation=1, parent_id=None, island_idx=0,
        operator="revise", report=EvalReport(fitness=1.0, passed=True),
    )
    island = IslandView(island_idx=0, candidates=[existing])
    gate = NoveltyGate(hashing_embedding)   # identity is the default

    from evoharness.evocore.novelty import novelty_text
    same = novelty_text(existing.workspace, existing.code)
    assert gate.check(same, island).accepted is False        # nothing new
    assert gate.check(same + "\n\n\n", island).accepted is False  # whitespace
    tweaked = same.replace("return x + 1", "return x + 2")
    verdict = gate.check(tweaked, island)
    assert verdict.accepted is True                          # a real edit
    assert verdict.embedding is not None    # still recorded as liveness proof

    # the fuzzy path would have rejected that same one-token edit
    fuzzy = NoveltyGate(hashing_embedding, mode="similarity")
    existing.embedding = hashing_embedding(same)
    assert fuzzy.check(tweaked, island).accepted is False
