"""The merge lane inside SearchLoop: a proposal that costs no model call,
carries a genome identical to its base, and reaches the domain as donors."""

from pathlib import Path

from evoharness.evocore import (
    EvalReport,
    InspirationSelector,
    LLMClient,
    LLMResponse,
    PopulationConfig,
    PopulationStore,
    PromptBuilder,
    SearchConfig,
    SearchLoop,
    StaticRouter,
    make_parent_selector,
)
from evoharness.evoplus.feedback import BehaviorSignature
from evoharness.evoplus.merge import StateMergePlanner

INITIAL = """# EDIT-REGION-BEGIN
x = 0
x += 1
# EDIT-REGION-END
print(x)
"""


class _SignedGrader:
    """Passes everything, and hands back a per-item pass vector.

    Alternating vectors per candidate manufacture the two-sided disagreement
    the planner requires, which is otherwise a property of a real domain.
    """

    def __init__(self):
        self.seen = 0
        self.donors_seen = []

    def grade(self, cand, workdir: Path) -> EvalReport:
        self.donors_seen.append(cand.metadata.get("state_donors"))
        self.seen += 1
        bits = (True, self.seen % 2 == 0, self.seen % 2 == 1, False)
        return EvalReport(
            fitness=1.0 + 0.01 * self.seen,
            passed=True,
            structured_feedback={
                "items": [
                    {"item_id": str(i), "passed": bool(p)}
                    for i, p in enumerate(bits)
                ]
            },
        )


class _CountingTransport:
    """Every model call is recorded, so a merge can be shown to make none."""

    def __init__(self):
        self.calls = 0

    def __call__(self, messages, model, **kw):
        self.calls += 1
        body = "x = 0\n" + "x += 1\n" * (self.calls + 1)
        code = f"# EDIT-REGION-BEGIN\n{body}# EDIT-REGION-END\nprint(x)\n"
        return LLMResponse(
            text=f"TITLE: t{self.calls}\nSUMMARY: s\n```python\n{code}```",
            model=model,
            cost=0.002,
        )


def _build(tmp_path, planner, grader, transport, generations=6):
    cfg = SearchConfig(
        num_generations=generations,
        operators=["rewrite"],
        operator_probs=[1.0],
        seed=3,
    )
    pop_cfg = PopulationConfig(num_islands=1)
    store = PopulationStore(pop_cfg)
    from evoharness.evoplus import SignatureRecorder

    loop = SearchLoop(
        cfg=cfg,
        pop_cfg=pop_cfg,
        store=store,
        grader=grader,
        llm=LLMClient(transport=transport, sleep=lambda s: None),
        prompt_builder=PromptBuilder("maximize"),
        parent_selector=make_parent_selector(pop_cfg),
        inspiration_selector=InspirationSelector(pop_cfg),
        model_router=StaticRouter(["mock-model"]),
        observers=[SignatureRecorder()],
        workdir=tmp_path,
        merge_planner=planner,
    )
    return loop, store


def test_merge_reaches_the_domain_without_a_model_call(tmp_path):
    grader, transport = _SignedGrader(), _CountingTransport()
    # Always fire once a complementary pair exists, so the run is decided by
    # the population rather than by the seeded roll.
    loop, store = _build(
        tmp_path, StateMergePlanner(probability=1.0), grader, transport
    )
    loop.run(INITIAL)

    merges = [c for c in store.all_candidates() if c.operator == "merge"]
    assert merges, "no merge was ever planned"
    for cand in merges:
        donors = cand.metadata["state_donors"]
        assert len(donors) == 2
        assert sum(d["weight"] for d in donors) == 1.0
        # The base is the first donor and the child's recorded parent.
        assert donors[0]["id"] == cand.parent_id
    # Grading saw the donors, which is the whole point of the channel.
    assert any(d for d in grader.donors_seen if d)


def test_a_merge_child_is_a_byte_identical_copy_of_its_base(tmp_path):
    grader, transport = _SignedGrader(), _CountingTransport()
    loop, store = _build(
        tmp_path, StateMergePlanner(probability=1.0), grader, transport
    )
    loop.run(INITIAL)

    merges = [c for c in store.all_candidates() if c.operator == "merge"]
    assert merges
    for cand in merges:
        base = store.get(cand.parent_id)
        assert base is not None
        # The mutation is in the inherited state, not the program text.
        assert cand.code == base.code


def test_merges_are_free(tmp_path):
    """The economic argument for the operator: no model call, no cost."""
    grader, transport = _SignedGrader(), _CountingTransport()
    loop, store = _build(
        tmp_path, StateMergePlanner(probability=1.0), grader, transport
    )
    report = loop.run(INITIAL)

    merges = [c for c in store.all_candidates() if c.operator == "merge"]
    ordinary = [
        c
        for c in store.all_candidates()
        if c.operator not in ("merge", "seed")
    ]
    assert merges and transport.calls == len(ordinary)
    assert report.total_llm_cost == len(ordinary) * 0.002


def test_the_novelty_gate_does_not_reject_merges(tmp_path):
    """A merge is a duplicate by construction; judging it on program text
    would reject every one the planner ever produced."""
    from evoharness.evocore.novelty import NoveltyGate, hashing_embedding

    grader, transport = _SignedGrader(), _CountingTransport()
    loop, store = _build(
        tmp_path, StateMergePlanner(probability=1.0), grader, transport
    )
    loop.novelty_gate = NoveltyGate(hashing_embedding, mode="identity")
    report = loop.run(INITIAL)

    assert [c for c in store.all_candidates() if c.operator == "merge"]
    merge_rejections = [
        h
        for h in report.history
        if h.get("status") == "proposal_failed"
        and h.get("operator") == "merge"
    ]
    assert not merge_rejections


def test_no_planner_means_no_behaviour_change(tmp_path):
    grader, transport = _SignedGrader(), _CountingTransport()
    loop, store = _build(tmp_path, None, grader, transport)
    loop.run(INITIAL)
    assert not [c for c in store.all_candidates() if c.operator == "merge"]


def test_donors_survive_the_trip_to_a_real_grade_context(tmp_path):
    """The last link: candidate metadata must arrive as GradeContext fields.

    Loop-level tests read `cand.metadata` directly, which would keep passing
    even if the adapter that builds the domain's context dropped the channel
    entirely.
    """
    from evoharness.evocore.population import Candidate
    from evoharness.evoserve import GradeContext
    from evoharness.task import WorkspaceGradeFnGrader

    seen: dict[str, GradeContext] = {}

    def grade_workspace(candidate_dir, ctx: GradeContext):
        seen["ctx"] = ctx
        return {"fitness": 1.0, "passed": True}

    grader = WorkspaceGradeFnGrader(grade_workspace, lineage_dir=tmp_path)
    donors = [
        {"id": "base", "weight": 0.25},
        {"id": "donor", "weight": 0.75},
    ]
    grader.grade(
        Candidate(
            id="merged",
            code="x = 1",
            generation=4,
            parent_id="base",
            island_idx=0,
            operator="merge",
            metadata={"state_donors": donors},
        ),
        tmp_path / "work",
    )

    assert list(seen["ctx"].state_donors) == donors
    assert seen["ctx"].operator == "merge"


def test_an_ordinary_candidate_reaches_the_domain_with_no_donors(tmp_path):
    from evoharness.evocore.population import Candidate
    from evoharness.evoserve import GradeContext
    from evoharness.task import WorkspaceGradeFnGrader

    seen: dict[str, GradeContext] = {}

    def grade_workspace(candidate_dir, ctx: GradeContext):
        seen["ctx"] = ctx
        return {"fitness": 1.0, "passed": True}

    WorkspaceGradeFnGrader(grade_workspace).grade(
        Candidate(
            id="plain",
            code="x = 1",
            generation=1,
            parent_id=None,
            island_idx=0,
            operator="rewrite",
        ),
        tmp_path / "work2",
    )
    assert seen["ctx"].state_donors == ()


def test_planner_state_is_checkpointed(tmp_path):
    grader, transport = _SignedGrader(), _CountingTransport()
    planner = StateMergePlanner(probability=1.0)
    loop, _ = _build(tmp_path, planner, grader, transport)
    loop.run(INITIAL)
    assert planner.attempted
    assert any(
        name.startswith("StateMergePlanner")
        for name in loop._stateful_components()
    )
