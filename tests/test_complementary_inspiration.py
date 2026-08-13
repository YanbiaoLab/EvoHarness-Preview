"""Complementarity-driven inspiration: the reference chosen for what it
knows, and the note that tells the model why it is there."""

import numpy as np
import pytest

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
from evoharness.core.interfaces import MutationContext
from evoharness.core.population import Candidate
from evoharness.evoplus.feedback import BehaviorSignature
from evoharness.evoplus.inspiration import ComplementaryInspiration


def _sig(bits: str) -> str:
    return BehaviorSignature(
        pass_vector=tuple(c == "1" for c in bits), error_histogram=()
    ).encode()


def _cand(cid, fitness, bits, island=0, items=None):
    feedback = None
    if items is not None:
        feedback = {
            "items": [
                {"item_id": name, "passed": bits[i] == "1"}
                for i, name in enumerate(items)
            ]
        }
    return Candidate(
        id=cid,
        code=f"# {cid}",
        generation=1,
        parent_id=None,
        island_idx=island,
        operator="revise",
        report=EvalReport(
            fitness=fitness, passed=True, structured_feedback=feedback
        ),
        behavior_signature=_sig(bits) if bits else None,
    )


# -- the policy alone -----------------------------------------------------


def test_the_solver_beats_the_runner_up():
    """A weaker candidate that solves the parent's failure outranks a
    stronger one that fails exactly where the parent does."""
    parent = _cand("parent", 0.9, "1100")
    runner_up = _cand("runner", 0.85, "1100")     # same holes as parent
    solver = _cand("solver", 0.60, "1011")        # fills them
    picked = ComplementaryInspiration().pick(parent, [runner_up, solver])
    assert picked is not None and picked[0].id == "solver"


def test_one_sided_a_strictly_stronger_donor_is_still_shown():
    """The merge planner must exclude a dominating donor; the inspiration
    policy must NOT -- it is the most useful thing to read."""
    parent = _cand("parent", 0.9, "1100")
    stronger = _cand("stronger", 0.95, "1111")
    picked = ComplementaryInspiration().pick(parent, [stronger])
    assert picked is not None and picked[0].id == "stronger"


def test_the_note_names_the_items():
    items = ["t9#12", "t10#4", "t10#7", "t10#9"]
    parent = _cand("parent", 0.9, "1100", items=items)
    solver = _cand("solver", 0.8, "1011")
    _, note = ComplementaryInspiration().pick(parent, [solver])
    assert "t10#7" in note and "t10#9" in note
    assert "fails 1 the parent solves" in note


def test_degrades_to_silence():
    no_sig = _cand("parent", 0.9, "")
    assert (
        ComplementaryInspiration().pick(no_sig, [_cand("x", 0.8, "10")])
        is None
    )
    perfect = _cand("parent", 1.0, "1111")
    assert (
        ComplementaryInspiration().pick(perfect, [_cand("x", 0.8, "1110")])
        is None
    )
    parent = _cand("parent", 0.9, "1100")
    other_rung = _cand("x", 0.8, "101010")        # different length: skipped
    assert ComplementaryInspiration().pick(parent, [other_rung]) is None


def test_deterministic():
    parent = _cand("parent", 0.9, "1100")
    pool = [_cand("a", 0.8, "1011"), _cand("b", 0.8, "1011")]
    first = ComplementaryInspiration().pick(parent, pool)
    second = ComplementaryInspiration().pick(parent, list(reversed(pool)))
    assert first[0].id == second[0].id


@pytest.mark.parametrize(
    "kwargs", [{"min_gain": 0}, {"max_named_items": -1}]
)
def test_nonsense_configuration_is_refused(kwargs):
    with pytest.raises(ValueError):
        ComplementaryInspiration(**kwargs)


# -- inside the selector --------------------------------------------------


def _store_with(cands):
    cfg = PopulationConfig(num_islands=1)
    store = PopulationStore(cfg)
    for c in cands:
        store.insert(c)
    store.refresh_archive()
    return cfg, store


def test_selector_replaces_the_last_top_slot():
    parent = _cand("parent", 0.9, "1100")
    cands = [
        parent,
        _cand("best", 0.95, "1100"),      # archive route takes this
        _cand("second", 0.85, "1100"),    # top-k route would take this...
        _cand("solver", 0.60, "1011"),    # ...but the solver replaces it
    ]
    cfg, store = _store_with(cands)
    selector = InspirationSelector(cfg, policy=ComplementaryInspiration())
    draw = selector.sample(parent, store, np.random.default_rng(0))
    assert [c.id for c in draw.top_k] == ["solver"]
    assert "Solves" in draw.notes["solver"]


def test_an_already_chosen_pick_gets_a_note_not_a_slot():
    parent = _cand("parent", 0.9, "1100")
    cands = [
        parent,
        _cand("best", 0.95, "1011"),      # archive best IS the solver
        _cand("second", 0.85, "1100"),
    ]
    cfg, store = _store_with(cands)
    selector = InspirationSelector(cfg, policy=ComplementaryInspiration())
    draw = selector.sample(parent, store, np.random.default_rng(0))
    assert [c.id for c in draw.archive] == ["best"]
    assert [c.id for c in draw.top_k] == ["second"]   # slots untouched
    assert "Solves" in draw.notes["best"]


def test_no_policy_is_todays_behaviour():
    parent = _cand("parent", 0.9, "1100")
    cands = [
        parent,
        _cand("best", 0.95, "1100"),
        _cand("solver", 0.6, "1011"),
    ]
    cfg, store = _store_with(cands)
    draw = InspirationSelector(cfg).sample(
        parent, store, np.random.default_rng(0)
    )
    assert draw.notes == {}


# -- the note reaches the prompt ------------------------------------------


def test_the_note_is_rendered_under_the_heading():
    parent = _cand("parent", 0.9, "1100")
    solver = _cand("solver", 0.6, "1011")
    ctx = MutationContext(
        parent=parent,
        archive_inspirations=[],
        top_k_inspirations=[solver],
        operator="revise",
        generation=3,
        inspiration_notes={"solver": "Solves 2 item(s) this parent fails."},
    )
    _system, user = PromptBuilder("maximize").build(ctx)
    assert "Note: Solves 2 item(s) this parent fails." in user


def test_no_note_no_line():
    parent = _cand("parent", 0.9, "1100")
    solver = _cand("solver", 0.6, "1011")
    ctx = MutationContext(
        parent=parent,
        archive_inspirations=[],
        top_k_inspirations=[solver],
        operator="revise",
        generation=3,
    )
    _system, user = PromptBuilder("maximize").build(ctx)
    assert "Note:" not in user


# -- liveness: the note appears from inside a real run ---------------------


def test_the_note_appears_in_a_live_runs_prompt(tmp_path):
    """Unit tests call pick() by hand; this one demands the whole chain fire
    unprompted -- recorder writes signatures, policy reads them, selector
    swaps the slot, builder renders the note, and the model actually
    RECEIVES it. Any dropped link leaves every prompt note-free."""
    from evoharness.evoplus import SignatureRecorder

    class _AlternatingGrader:
        """Even and odd candidates fail complementary halves of the items."""

        def __init__(self):
            self.count = 0

        def grade(self, cand, workdir):
            self.count += 1
            bits = (True, self.count % 2 == 0, self.count % 2 == 1, False)
            return EvalReport(
                fitness=0.5 + 0.01 * self.count,
                passed=True,
                structured_feedback={
                    "items": [
                        {"item_id": f"item#{i}", "passed": bool(p)}
                        for i, p in enumerate(bits)
                    ]
                },
            )

    seen_prompts: list[str] = []

    def transport(messages, model, **kw):
        seen_prompts.extend(str(m) for m in messages)
        n = len(seen_prompts)
        body = "x = 0\n" + "x += 1\n" * (n + 1)
        code = f"# EDIT-REGION-BEGIN\n{body}# EDIT-REGION-END\nprint(x)\n"
        return LLMResponse(
            text=f"TITLE: t{n}\nSUMMARY: s\n```python\n{code}```",
            model=model,
            cost=0.0,
        )

    pop_cfg = PopulationConfig(num_islands=1)
    store = PopulationStore(pop_cfg)
    loop = SearchLoop(
        cfg=SearchConfig(
            num_generations=6,
            operators=["rewrite"],
            operator_probs=[1.0],
            seed=5,
        ),
        pop_cfg=pop_cfg,
        store=store,
        grader=_AlternatingGrader(),
        llm=LLMClient(transport=transport, sleep=lambda s: None),
        prompt_builder=PromptBuilder("maximize"),
        parent_selector=make_parent_selector(pop_cfg),
        inspiration_selector=InspirationSelector(
            pop_cfg, policy=ComplementaryInspiration()
        ),
        model_router=StaticRouter(["mock-model"]),
        observers=[SignatureRecorder()],
        workdir=tmp_path,
    )
    loop.run(
        "# EDIT-REGION-BEGIN\nx = 0\nx += 1\n# EDIT-REGION-END\nprint(x)\n"
    )

    noted = [p for p in seen_prompts if "Note: Solves" in p]
    assert noted, "no prompt ever carried a complementarity note"
    assert any("item#" in p for p in noted), "the note never named an item"
