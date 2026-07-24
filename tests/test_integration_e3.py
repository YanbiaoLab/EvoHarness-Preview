"""End-to-end: SearchLoop with C1+C2+C3 plugged in (experiment group E3r
assembly), mock grader producing structured feedback, mock LLM."""

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
from evoharness.evoplus import (
    BehavioralNoveltyPolicy,
    ExperienceContributor,
    ExperienceStore,
    FeedbackContributor,
    ItemResult,
    SignatureRecorder,
    StructuredFeedback,
)

INITIAL = """# EDIT-REGION-BEGIN
SOLVED = ["q0"]
# EDIT-REGION-END
print(SOLVED)
"""

ITEMS = ["q0", "q1", "q2", "q3"]


class EquationalMockGrader:
    """Item qN passes iff the literal 'qN' appears in the code; structured
    feedback mirrors an equational verifier's per-item output."""

    def grade(self, cand, workdir) -> EvalReport:
        items = [
            ItemResult(
                item_id=q,
                passed=q in cand.code,
                predicted="?" if q not in cand.code else "yes",
                expected="yes",
                error_category="" if q in cand.code else f"missing-{q}",
            )
            for q in ITEMS
        ]
        feedback = StructuredFeedback(items=items)
        solved = sum(i.passed for i in items)
        return EvalReport(
            fitness=solved / len(ITEMS),
            passed=True,
            visible_metrics={"solved": solved},
            structured_feedback=feedback.to_json(),
            eval_cost_usd=0.001,
        )


def make_transport(captured_systems):
    """Solves one more item every second call; repeats itself otherwise so
    behavioral duplicates occur."""
    state = {"n": 0, "solved": 1}

    def transport(messages, model, **kw):
        captured_systems.append(messages[0].content)
        state["n"] += 1
        if state["n"] % 2 == 0 and state["solved"] < len(ITEMS):
            state["solved"] += 1
        names = ", ".join(f'"{q}"' for q in ITEMS[: state["solved"]])
        code = (
            "# EDIT-REGION-BEGIN\n"
            f"SOLVED = [{names}]\n"
            "# EDIT-REGION-END\n"
            "print(SOLVED)\n"
        )
        return LLMResponse(
            text=f"TITLE: solve more\nSUMMARY: extend list\n```python\n{code}```",
            model=model,
            cost=0.001,
        )

    return transport


def test_full_loop_with_all_three_extensions(tmp_path):
    captured_systems: list[str] = []
    cfg = SearchConfig(
        num_generations=12, operators=["rewrite"], operator_probs=[1.0], seed=3
    )
    pop_cfg = PopulationConfig(num_islands=1)
    store = PopulationStore(pop_cfg)

    llm = LLMClient(transport=make_transport(captured_systems), sleep=lambda s: None)
    recorder = SignatureRecorder()
    policy = BehavioralNoveltyPolicy(hamming_threshold=0, duplicate_penalty=0.25)
    xstore = ExperienceStore(tmp_path / "experience.jsonl")
    contributors = [
        FeedbackContributor(),
        ExperienceContributor(xstore, mode="retrieval"),
    ]

    loop = SearchLoop(
        cfg=cfg,
        pop_cfg=pop_cfg,
        store=store,
        grader=EquationalMockGrader(),
        llm=llm,
        prompt_builder=PromptBuilder("solve all items", contributors=contributors),
        parent_selector=make_parent_selector(pop_cfg, weight_policies=[policy]),
        inspiration_selector=InspirationSelector(pop_cfg),
        model_router=StaticRouter(["mock"]),
        observers=[recorder, policy, xstore],  # recorder MUST precede policy
        workdir=tmp_path,
    )
    report = loop.run(INITIAL)

    assert report.generations_completed == 12
    assert report.best_fitness == 1.0  # eventually solves all four items

    cands = store.all_candidates()
    graded = [c for c in cands if c.report is not None]
    # C1+C2: every graded candidate has a recorded behavior signature
    assert all(c.behavior_signature for c in graded)
    # C2: the repeating transport produced behavioral duplicates, and they
    # are excluded from the archive
    duplicates = [c for c in graded if c.behavior_duplicate]
    assert duplicates
    assert all(not c.in_archive for c in duplicates)
    # C3: experience accumulated and persisted
    assert len(xstore.entries) >= 5
    assert (tmp_path / "experience.jsonl").exists()

    # C1: some mutation prompt carried the parent failure analysis while the
    # parent still had failures
    assert any("Parent failure analysis" in s for s in captured_systems)
    # C3: once the buffer had outcomes they were injected into prompts
    assert any(
        "Experience from this run" in s for s in captured_systems
    )
    # solved candidates stop injecting the feedback section
    last_system = captured_systems[-1]
    assert "solve all items" in last_system
