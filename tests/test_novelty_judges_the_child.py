"""The novelty gate must judge the proposed child, never its parent.

Regression test for a defect that made a single-file genome unable to produce
any offspring at all while the gate was on: a single-shot proposer returns
code with no workspace, the gate fell back to the parent's workspace, and
novelty_text renders a workspace in full -- discarding the proposed code. In
identity mode every proposal then looked exactly like its parent.

Only single-file domains were affected, because a multi-file proposal ships a
child workspace and took the other branch. Both are covered here so neither
side can regress.
"""

from pathlib import Path

from evoharness.core import (
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
from evoharness.core.novelty import NoveltyGate, hashing_embedding, novelty_text
from evoharness.core.workspace import FileWorkspace

SINGLE_FILE = """# EDIT-REGION-BEGIN
x = 0
x += 1
# EDIT-REGION-END
print(x)
"""


class _Grader:
    def grade(self, cand, workdir: Path) -> EvalReport:
        return EvalReport(fitness=float(cand.code.count("x += 1")), passed=True)


def _transport():
    counter = {"n": 1}

    def transport(messages, model, **kw):
        counter["n"] += 1
        body = "x = 0\n" + "x += 1\n" * counter["n"]
        code = f"# EDIT-REGION-BEGIN\n{body}# EDIT-REGION-END\nprint(x)\n"
        return LLMResponse(
            text=f"TITLE: t{counter['n']}\nSUMMARY: s\n```python\n{code}```",
            model=model,
            cost=0.001,
        )

    return transport


def test_novelty_text_of_a_child_workspace_shows_the_child():
    """The unit underneath: rendering must reflect the proposed code."""
    parent = FileWorkspace("x = 0")
    child = parent.with_main_text("x = 999")
    assert "999" in novelty_text(child, "x = 999")
    # And the old fallback is exactly what it must never be handed.
    assert "999" not in novelty_text(parent, "x = 999")


def test_single_file_domain_still_produces_offspring_with_the_gate_on(tmp_path):
    cfg = SearchConfig(
        num_generations=5, operators=["rewrite"], operator_probs=[1.0], seed=3
    )
    pop_cfg = PopulationConfig(num_islands=1)
    store = PopulationStore(pop_cfg)
    loop = SearchLoop(
        cfg=cfg,
        pop_cfg=pop_cfg,
        store=store,
        grader=_Grader(),
        llm=LLMClient(transport=_transport(), sleep=lambda s: None),
        prompt_builder=PromptBuilder("maximize"),
        parent_selector=make_parent_selector(pop_cfg),
        inspiration_selector=InspirationSelector(pop_cfg),
        model_router=StaticRouter(["mock-model"]),
        novelty_gate=NoveltyGate(hashing_embedding, mode="identity"),
        workdir=tmp_path,
    )
    report = loop.run(SINGLE_FILE)

    children = [c for c in store.all_candidates() if c.operator != "seed"]
    assert len(children) == 5, "the gate rejected genuinely novel children"
    assert report.novelty_rejections == 0
    assert report.best_fitness > 1.0


def test_an_actual_duplicate_is_still_rejected(tmp_path):
    """The fix must not turn the gate off."""

    def echo_parent(messages, model, **kw):
        return LLMResponse(
            text=f"TITLE: none\nSUMMARY: none\n```python\n{SINGLE_FILE}```",
            model=model,
            cost=0.001,
        )

    cfg = SearchConfig(
        num_generations=3, operators=["rewrite"], operator_probs=[1.0], seed=3
    )
    pop_cfg = PopulationConfig(num_islands=1)
    store = PopulationStore(pop_cfg)
    loop = SearchLoop(
        cfg=cfg,
        pop_cfg=pop_cfg,
        store=store,
        grader=_Grader(),
        llm=LLMClient(transport=echo_parent, sleep=lambda s: None),
        prompt_builder=PromptBuilder("maximize"),
        parent_selector=make_parent_selector(pop_cfg),
        inspiration_selector=InspirationSelector(pop_cfg),
        model_router=StaticRouter(["mock-model"]),
        novelty_gate=NoveltyGate(hashing_embedding, mode="identity"),
        workdir=tmp_path,
    )
    report = loop.run(SINGLE_FILE)

    assert report.novelty_rejections > 0
    assert not [c for c in store.all_candidates() if c.operator != "seed"]
