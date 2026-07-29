import numpy as np
import pytest

from conftest import make_candidate
from evoharness.evocore import (
    LLMClient,
    LLMResponse,
    MutationContext,
    PopulationConfig,
    PopulationStore,
    WeightedSelector,
)
from evoharness.evoplus import (
    BehaviorSignature,
    BehavioralNoveltyPolicy,
    ExperienceContributor,
    ExperienceStore,
    FeedbackContributor,
    ItemResult,
    SignatureRecorder,
    StructuredFeedback,
)


def feedback_fixture(pattern="1010"):
    items = [
        ItemResult(
            item_id=f"q{i}",
            passed=(c == "1"),
            predicted="B" if c == "0" else "A",
            expected="A",
            error_category="" if c == "1" else ("parse" if i % 2 else "logic"),
        )
        for i, c in enumerate(pattern)
    ]
    return StructuredFeedback(items=items, summary="two failures")


# -- C1 -----------------------------------------------------------------------

def test_signature_encode_decode_hamming():
    sig = feedback_fixture("1010").signature()
    assert sig.pass_vector == (True, False, True, False)
    decoded = BehaviorSignature.decode(sig.encode())
    assert decoded == sig
    other = feedback_fixture("1000").signature()
    assert sig.hamming(other) == 1
    shorter = BehaviorSignature(pass_vector=(True,), error_histogram=())
    # common prefix (T vs T) has 0 mismatches, length difference adds 3
    assert sig.hamming(shorter) == 3


def test_structured_feedback_roundtrip_and_render():
    fb = feedback_fixture("1000")
    rt = StructuredFeedback.from_json(fb.to_json())
    assert rt.error_histogram == fb.error_histogram
    text = fb.render(top_k=2)
    assert "failed 3/4 items" in text
    assert "parse" in text and "logic" in text
    assert "expected 'A', got 'B'" in text


def test_feedback_contributor_injects_only_on_failures():
    parent = make_candidate("p", 1.0)
    parent.report.structured_feedback = feedback_fixture("1010").to_json()
    ctx = MutationContext(parent, [], [], "revise", 3)
    section = FeedbackContributor().contribute(ctx)
    assert section.startswith("# Parent failure analysis")

    clean = make_candidate("c", 1.0)
    clean.report.structured_feedback = feedback_fixture("1111").to_json()
    assert FeedbackContributor().contribute(
        MutationContext(clean, [], [], "revise", 3)
    ) is None
    no_fb = make_candidate("n", 1.0)
    assert FeedbackContributor().contribute(
        MutationContext(no_fb, [], [], "revise", 3)
    ) is None


# -- C2 -----------------------------------------------------------------------

def test_recorder_and_duplicate_marking():
    store = PopulationStore(PopulationConfig())
    recorder = SignatureRecorder()
    policy = BehavioralNoveltyPolicy(hamming_threshold=0, duplicate_penalty=0.25)

    first = make_candidate("a", 1.0, island=0)
    first.report.structured_feedback = feedback_fixture("1010").to_json()
    recorder.on_candidate_graded(first, store)
    policy.on_candidate_graded(first, store)
    assert first.behavior_signature is not None
    assert not first.behavior_duplicate  # nothing to collide with yet
    store.insert(first)

    dup = make_candidate("b", 2.0, island=0)
    dup.report.structured_feedback = feedback_fixture("1010").to_json()
    recorder.on_candidate_graded(dup, store)
    policy.on_candidate_graded(dup, store)
    assert dup.behavior_duplicate
    store.insert(dup)

    novel = make_candidate("c", 2.0, island=0)
    novel.report.structured_feedback = feedback_fixture("1110").to_json()
    recorder.on_candidate_graded(novel, store)
    policy.on_candidate_graded(novel, store)
    assert not novel.behavior_duplicate
    store.insert(novel)

    stats = policy.coverage_stats(store)
    assert stats.distinct_signatures == 2 and stats.duplicates_marked == 1


def test_duplicate_excluded_from_archive_and_downweighted():
    cfg = PopulationConfig(archive_size=10)
    store = PopulationStore(cfg)
    ok = make_candidate("ok", 1.0)
    dup = make_candidate("dup", 5.0)
    dup.behavior_duplicate = True
    store.insert(ok)
    store.insert(dup)
    store.refresh_archive()
    archived = {c.id for c in store.all_candidates() if c.in_archive}
    assert archived == {"ok"}  # duplicate excluded despite higher fitness

    policy = BehavioralNoveltyPolicy(duplicate_penalty=0.5)
    probs = WeightedSelector(cfg, [policy]).probabilities([ok, dup])
    baseline = WeightedSelector(cfg).probabilities([ok, dup])
    assert probs[1] < baseline[1]  # penalty reduces the duplicate's share


def test_hamming_threshold_relaxation():
    store = PopulationStore(PopulationConfig())
    near = make_candidate("near", 1.0, island=0)
    near.behavior_signature = feedback_fixture("1010").signature().encode()
    store.insert(near)
    policy = BehavioralNoveltyPolicy(hamming_threshold=1)
    cand = make_candidate("x", 1.0, island=0)
    cand.behavior_signature = feedback_fixture("1000").signature().encode()
    policy.on_candidate_graded(cand, store)
    assert cand.behavior_duplicate  # distance 1 <= threshold 1


# -- C3 -----------------------------------------------------------------------

def _store_with_lineage(tmp_path):
    pop = PopulationStore(PopulationConfig())
    parent = make_candidate("p", 1.0)
    pop.insert(parent)
    xs = ExperienceStore(tmp_path / "exp.jsonl")

    win = make_candidate("w", 1.5, parent_id="p", generation=2)
    win.operator, win.change_title = "revise", "add cache"
    win.change_summary = "memoize the verifier"
    xs.on_candidate_graded(win, pop)

    loss = make_candidate("l", 0.5, parent_id="p", generation=3)
    loss.operator, loss.change_title = "rewrite", "raise temperature"
    xs.on_candidate_graded(loss, pop)
    return pop, xs


def test_experience_store_records_wins_and_losses(tmp_path):
    pop, xs = _store_with_lineage(tmp_path)
    assert len(xs.entries) == 2
    assert [e.change_title for e in xs.recent_wins()] == ["add cache"]
    assert [e.change_title for e in xs.recent_losses()] == ["raise temperature"]
    assert all(e.kind == "evaluated" for e in xs.entries)
    assert xs.entries[0].parent_id == "p" and xs.entries[0].child_id == "w"
    # jsonl persistence roundtrip
    reloaded = ExperienceStore(tmp_path / "exp.jsonl")
    assert len(reloaded.entries) == 2


def test_experience_store_loads_v1_rows(tmp_path):
    path = tmp_path / "exp.jsonl"
    path.write_text(
        '{"parent_error_categories": ["logic"], "operator": "revise", '
        '"change_title": "old", "change_summary": "s", '
        '"fitness_delta": 0.2, "success": true, "generation": 1}\n'
    )
    xs = ExperienceStore(path)
    assert len(xs.entries) == 1
    entry = xs.entries[0]
    assert entry.kind == "evaluated"  # v1 rows default to evaluated
    assert entry.redeemed_by is None
    assert entry.change_title == "old"


def test_experience_redemption(tmp_path):
    pop, xs = _store_with_lineage(tmp_path)
    pop.insert(make_candidate("l", 0.5, parent_id="p", generation=3))
    # A child of the regressed candidate improves past it: the parent's
    # negative entry is redeemed and leaves the loss sections.
    redeemer = make_candidate("r", 0.9, parent_id="l", generation=4)
    xs.on_candidate_graded(redeemer, pop)
    entry = next(e for e in xs.entries if e.child_id == "l")
    assert entry.redeemed_by == "r"
    assert all(e.child_id != "l" for e in xs.recent_losses())
    assert all(e.child_id != "l" for e in xs.recent_negatives(generation=5))
    # redemption row survives the jsonl roundtrip
    reloaded = ExperienceStore(tmp_path / "exp.jsonl")
    assert next(e for e in reloaded.entries if e.child_id == "l").redeemed_by == "r"


def test_experience_store_records_rejections(tmp_path):
    from evoharness.evocore import RejectionEvent

    xs = ExperienceStore(tmp_path / "exp.jsonl")
    parent = make_candidate(
        "p", 1.0, island=1, code="print('hello world one')\n"
    )
    xs.on_proposal_rejected(
        RejectionEvent(
            kind="novelty",
            generation=4,
            operator="rewrite",
            parent=parent,
            proposal_code="print('hello world two')\n",
            change_title="tweak print",
            max_similarity=0.995,
            most_similar_id="abc",
        )
    )
    xs.on_proposal_rejected(
        RejectionEvent(
            kind="proposal_failed",
            generation=5,
            operator="diff",
            parent=parent,
            failure_reason="no valid diff produced",
        )
    )
    kinds = [e.kind for e in xs.entries]
    assert kinds == ["rejected_novelty", "proposal_failed"]
    novelty = xs.entries[0]
    assert novelty.island_idx == 1 and novelty.max_similarity == 0.995
    assert "hello world two" in novelty.change_summary  # diff vs parent text
    rendered = novelty.render(show_origin=True)
    assert "REJECTED before eval" in rendered and "island 1" in rendered
    assert "PROPOSAL FAILED" in xs.entries[1].render()
    # same-island entries sort first for a same-island parent
    negatives = xs.recent_negatives(generation=5, island_idx=1)
    assert len(negatives) == 2
    reloaded = ExperienceStore(tmp_path / "exp.jsonl")
    assert [e.kind for e in reloaded.entries] == kinds


def test_experience_contributor_retrieval(tmp_path):
    pop, xs = _store_with_lineage(tmp_path)
    contrib = ExperienceContributor(xs, mode="retrieval")
    parent = pop.get("p")
    section = contrib.contribute(MutationContext(parent, [], [], "revise", 4))
    assert "Experience from this run" in section
    assert "add cache" in section and "helped" in section
    assert "raise temperature" in section and "did NOT help" in section
    # losses render before wins (strongest positive example last)
    assert section.index("raise temperature") < section.index("add cache")

    off = ExperienceContributor(xs, mode="off")
    assert off.contribute(MutationContext(parent, [], [], "revise", 4)) is None


def test_experience_contributor_rejected_section(tmp_path):
    from evoharness.evocore import RejectionEvent

    pop, xs = _store_with_lineage(tmp_path)
    parent = pop.get("p")
    xs.on_proposal_rejected(
        RejectionEvent(
            kind="proposal_failed",
            generation=4,
            operator="diff",
            parent=parent,
            failure_reason="no valid diff produced",
        )
    )
    contrib = ExperienceContributor(xs, mode="retrieval+rejected")
    section = contrib.contribute(MutationContext(parent, [], [], "revise", 5))
    assert "avoid repeating without variation" in section
    assert "PROPOSAL FAILED" in section
    # plain retrieval mode omits the negative section
    plain = ExperienceContributor(xs, mode="retrieval")
    assert "PROPOSAL FAILED" not in plain.contribute(
        MutationContext(parent, [], [], "revise", 5)
    )


def test_experience_contributor_global_distills_on_interval(tmp_path):
    pop, xs = _store_with_lineage(tmp_path)
    calls = {"n": 0}

    def transport(messages, model, **kw):
        calls["n"] += 1
        assert "cheatsheet" in messages[0].content
        return LLMResponse(text=f"cheatsheet content v{calls['n']}", model=model)

    llm = LLMClient(transport=transport, sleep=lambda s: None)
    contrib = ExperienceContributor(
        xs, mode="global", llm=llm, model="m", interval=5
    )
    parent = pop.get("p")
    s1 = contrib.contribute(MutationContext(parent, [], [], "revise", 5))
    assert "cheatsheet (v1)" in s1 and calls["n"] == 1
    s2 = contrib.contribute(MutationContext(parent, [], [], "revise", 7))
    assert "cheatsheet (v1)" in s2 and calls["n"] == 1  # cached, not stale yet
    s3 = contrib.contribute(MutationContext(parent, [], [], "revise", 10))
    assert "cheatsheet (v2)" in s3 and calls["n"] == 2  # refreshed

    with pytest.raises(ValueError):
        ExperienceContributor(xs, mode="global")  # llm required
    with pytest.raises(ValueError):
        ExperienceContributor(xs, mode="bogus")


# -- L1 reflection (items 6-7) --------------------------------------------------

def _reflect_transport(captured, verdict: str = "improved"):
    """Echoes one lesson per '### mutation <id>' block it sees."""
    import json
    import re

    def transport(messages, model, **kw):
        captured.append(messages[1].content)
        ids = re.findall(r"### mutation (\S+)", messages[1].content)
        body = {
            "lessons": [
                {
                    "child_id": i,
                    "verdict": verdict,
                    "why": f"why-{i}",
                    "advice": f"advice-{i}",
                    "tags": ["Missing Case"],
                }
                for i in ids
            ],
            "scratchpad": "S" * 3000,
        }
        return LLMResponse(text=json.dumps(body), model=model)

    return transport


def test_reflector_batches_lessons_and_scratchpad(tmp_path):
    from evoharness.evoplus import MutationReflector

    pop = PopulationStore(PopulationConfig())
    pop.insert(make_candidate("p", 1.0))
    xs = ExperienceStore(tmp_path / "exp.jsonl")
    captured: list[str] = []
    llm = LLMClient(
        transport=_reflect_transport(captured), sleep=lambda s: None
    )
    reflector = MutationReflector(xs, llm, model="m", batch_size=3)

    for i in range(3):
        child = make_candidate(
            f"c{i}", 1.0 + 0.1 * (i + 1), parent_id="p", generation=i + 1
        )
        pop.insert(child)
        xs.on_candidate_graded(child, pop)          # store BEFORE reflector
        reflector.on_candidate_graded(child, pop)

    assert len(captured) == 1  # one batched call, triggered by the 3rd entry
    assert all(e.lesson for e in xs.entries)
    assert xs.entries[0].lesson["why"] == "why-c0"
    assert xs.entries[0].lesson["tags"] == ["missing-case"]  # slug-normalized
    assert not xs.pending_reflection()
    assert reflector.scratchpad_version == 1
    assert len(reflector.scratchpad) <= 2000  # hard budget enforced
    # lessons survive the jsonl roundtrip
    reloaded = ExperienceStore(tmp_path / "exp.jsonl")
    assert all(e.lesson for e in reloaded.entries)
    # checkpoint roundtrip
    fresh = MutationReflector(xs, llm, batch_size=3)
    fresh.set_state(reflector.state())
    assert fresh.scratchpad == reflector.scratchpad
    assert fresh.scratchpad_version == 1


def test_reflector_skips_bad_llm_output(tmp_path):
    from evoharness.evoplus import MutationReflector

    pop = PopulationStore(PopulationConfig())
    pop.insert(make_candidate("p", 1.0))
    xs = ExperienceStore(tmp_path / "exp.jsonl")
    llm = LLMClient(
        transport=lambda messages, model, **kw: LLMResponse(
            text="not json at all", model=model
        ),
        sleep=lambda s: None,
    )
    reflector = MutationReflector(xs, llm, batch_size=2)
    for i in range(2):
        child = make_candidate(f"c{i}", 1.2, parent_id="p", generation=i + 1)
        pop.insert(child)
        xs.on_candidate_graded(child, pop)
        reflector.on_candidate_graded(child, pop)
    # batch attempted and failed: run survives, entries stay pending
    assert all(e.lesson is None for e in xs.entries)
    assert len(xs.pending_reflection()) == 2
    assert reflector.scratchpad_version == 0


# -- Contributor v2 (item 8) -----------------------------------------------------

def _lessoned_store(tmp_path):
    pop, xs = _store_with_lineage(tmp_path)  # entries: w (win), l (loss)
    xs.attach_lesson("w", {
        "verdict": "improved",
        "why": "cache removed rework",
        "advice": "memoize expensive verifier calls",
        "tags": ["slow-verify"],
    })
    xs.attach_lesson("l", {
        "verdict": "regressed",
        "why": "temperature added noise",
        "advice": "avoid raising temperature blindly",
        "tags": ["prompt-noise"],
    })
    return pop, xs


def test_contributor_lessons_mode(tmp_path):
    pop, xs = _lessoned_store(tmp_path)
    contrib = ExperienceContributor(xs, mode="lessons")
    parent = pop.get("p")
    section = contrib.contribute(MutationContext(parent, [], [], "revise", 4))
    assert "Lessons from past mutations" in section
    assert "parent not attributed yet" in section  # root has no own lesson
    assert "advice: memoize expensive verifier calls" in section
    assert "advice: avoid raising temperature blindly" in section
    # regressed lesson renders before the improved one
    assert section.index("raise temperature") < section.index("add cache")
    # the lessoned regression left the mechanical negative section
    assert all(e.child_id != "l" for e in xs.recent_negatives(generation=5))


def test_contributor_lessons_tag_match(tmp_path):
    pop, xs = _lessoned_store(tmp_path)
    extra = make_candidate("x", 1.2, parent_id="p", generation=4)
    extra.operator, extra.change_title = "revise", "tune sampling"
    pop.insert(extra)
    xs.on_candidate_graded(extra, pop)
    xs.attach_lesson("x", {
        "verdict": "improved",
        "why": "less sampling noise",
        "advice": "keep temperature low",
        "tags": ["prompt-noise"],
    })
    # mutate FROM l, whose own lesson carries tags ["prompt-noise"]
    fake_parent = make_candidate("l", 0.5)
    section = ExperienceContributor(xs, mode="lessons").contribute(
        MutationContext(fake_parent, [], [], "revise", 5)
    )
    assert "matched on the parent's failure tags: prompt-noise" in section
    # tag-matched win (x) outranks the bigger-delta win (w), so after the
    # worst->best flip it renders LAST (recency-bias slot)
    assert section.index("add cache") < section.index("tune sampling")


def test_contributor_lessons_falls_back_before_first_batch(tmp_path):
    pop, xs = _store_with_lineage(tmp_path)  # no lessons attached
    section = ExperienceContributor(xs, mode="lessons").contribute(
        MutationContext(pop.get("p"), [], [], "revise", 4)
    )
    assert "Experience from this run" in section  # mechanical fallback


def test_contributor_scratchpad_hint(tmp_path):
    class FakeReflector:
        scratchpad = (
            "Successful patterns\n- pattern A\n"
            "Unexplored directions\n- try operator X\n- try shorter prompts\n"
        )

    pop, xs = _lessoned_store(tmp_path)
    contrib = ExperienceContributor(
        xs, mode="lessons+scratchpad", reflector=FakeReflector()
    )
    parent = pop.get("p")
    ctx = MutationContext(parent, [], [], "revise", 6)
    section = contrib.contribute(ctx)
    assert "Direction hint" in section
    hint = section.split("Direction hint (from the evolution scratchpad)")[1]
    # sampled from the Unexplored section only, deterministically
    assert "try operator" in hint or "try shorter prompts" in hint
    assert "pattern A" not in hint
    assert contrib.contribute(ctx) == section  # same ctx -> same sample

    with pytest.raises(ValueError):
        ExperienceContributor(xs, mode="lessons+scratchpad")  # needs reflector


# -- L2 consolidation + wiring (items 9-10) ---------------------------------------

def test_reflector_consolidation_merges_tags_and_resolves_contradictions(tmp_path):
    import json
    from evoharness.evoplus import MutationReflector

    pop, xs = _store_with_lineage(tmp_path)
    xs.attach_lesson("w", {
        "verdict": "improved", "why": "", "advice": "keep caching",
        "tags": ["missing-case"],
    })
    xs.attach_lesson("l", {
        "verdict": "regressed", "why": "", "advice": "avoid temperature",
        "tags": ["missing-cases"],
    })
    # redeem l so its lesson qualifies for the stepping-stone rewrite
    pop.insert(make_candidate("l", 0.5, parent_id="p", generation=3))
    redeemer = make_candidate("r", 0.9, parent_id="l", generation=4)
    pop.insert(redeemer)
    xs.on_candidate_graded(redeemer, pop)

    def transport(messages, model, **kw):
        assert "lesson memory" in messages[0].content
        assert "REDEEMED by r" in messages[1].content
        body = {
            "tag_map": {"missing-cases": "missing-case"},
            "rewrites": [
                {"child_id": "l",
                 "advice": "stepping stone: keep direction, smaller steps"},
                {"child_id": "w", "advice": "cache only hot paths"},
                {"child_id": "ghost", "advice": "MUST NOT APPLY"},  # invented
            ],
        }
        return LLMResponse(text=json.dumps(body), model=model)

    llm = LLMClient(transport=transport, sleep=lambda s: None)
    reflector = MutationReflector(xs, llm, model="m", consolidate_threshold=2)
    reflector._consolidate()

    l_entry = next(e for e in xs.entries if e.child_id == "l")
    w_entry = next(e for e in xs.entries if e.child_id == "w")
    assert l_entry.lesson["tags"] == ["missing-case"]      # synonym merged
    assert "stepping stone" in l_entry.lesson["advice"]    # redeemed rewrite
    # contradictions need BOTH sides rewritten, so a non-redeemed lesson may
    # be rewritten too — but only one that was actually shown
    assert w_entry.lesson["advice"] == "cache only hot paths"
    assert all(
        "MUST NOT APPLY" not in (e.lesson or {}).get("advice", "")
        for e in xs.entries
    )
    assert reflector.consolidated_at == 2  # w+l lessoned; r still pending
    # consolidated lessons survive the jsonl roundtrip
    reloaded = ExperienceStore(tmp_path / "exp.jsonl")
    l_again = next(e for e in reloaded.entries if e.child_id == "l")
    assert l_again.lesson["tags"] == ["missing-case"]


def test_reflector_triggers_consolidation_and_charges_budget(tmp_path):
    from evoharness.evoplus import MutationReflector

    pop = PopulationStore(PopulationConfig())
    pop.insert(make_candidate("p", 1.0))
    xs = ExperienceStore(tmp_path / "exp.jsonl")
    calls: list[str] = []

    def transport(messages, model, **kw):
        import json
        import re
        if "attribution analyst" in messages[0].content:
            calls.append("reflect")
            ids = re.findall(r"### mutation (\S+)", messages[1].content)
            body = {
                "lessons": [
                    {"child_id": i, "verdict": "improved", "why": "w",
                     "advice": "a", "tags": ["t"]}
                    for i in ids
                ],
                "scratchpad": "- keep going",
            }
        else:
            calls.append("consolidate")
            body = {"tag_map": {}, "rewrites": []}
        return LLMResponse(text=json.dumps(body), model=model, cost=0.5)

    class SpyBudget:
        def __init__(self):
            self.charged: list[float] = []

        def charge(self, usd: float) -> None:
            self.charged.append(usd)

        def should_stop(self) -> bool:
            return False

    budget = SpyBudget()
    llm = LLMClient(transport=transport, sleep=lambda s: None)
    reflector = MutationReflector(
        xs, llm, model="m", batch_size=2, consolidate_threshold=2, budget=budget
    )
    for i in range(2):
        child = make_candidate(f"c{i}", 1.2, parent_id="p", generation=i + 1)
        pop.insert(child)
        xs.on_candidate_graded(child, pop)
        reflector.on_candidate_graded(child, pop)

    assert calls == ["reflect", "consolidate"]  # threshold hit right after batch
    assert budget.charged == [0.5, 0.5]         # both calls metered
    assert reflector.consolidated_at == 2
    # watermark survives the checkpoint roundtrip
    fresh = MutationReflector(xs, llm, budget=budget)
    fresh.set_state(reflector.state())
    assert fresh.consolidated_at == 2


# -- OperatorBandit + LessonDirective (backlog items 2 & 6) -----------------------

def test_operator_bandit_shifts_share_but_keeps_floor():
    from evoharness.evoplus import OperatorBandit

    bandit = OperatorBandit(["diff", "full", "recombine"], floor=0.1)
    pop = PopulationStore(PopulationConfig())
    pop.insert(make_candidate("p", 1.0))
    # "full" keeps winning, "diff" keeps losing
    for i in range(6):
        good = make_candidate(f"g{i}", 1.3, parent_id="p", generation=i + 1)
        good.operator = "full"
        bandit.on_candidate_graded(good, pop)
        bad = make_candidate(f"b{i}", 0.8, parent_id="p", generation=i + 1)
        bad.operator = "diff"
        bandit.on_candidate_graded(bad, pop)
    ops, probs = bandit.probabilities(has_inspirations=True)
    by_op = dict(zip(ops, probs))
    assert by_op["full"] > by_op["diff"]          # winner gains share
    assert all(p >= 0.1 - 1e-9 for p in probs)    # loser never starved
    assert abs(sum(probs) - 1.0) < 1e-9
    # recombine excluded without inspirations, probs renormalized
    ops2, probs2 = bandit.probabilities(has_inspirations=False)
    assert "recombine" not in ops2
    assert abs(sum(probs2) - 1.0) < 1e-9


def test_operator_bandit_sampling_and_state_roundtrip():
    from evoharness.evoplus import OperatorBandit

    bandit = OperatorBandit(["diff", "full"])
    bandit.ema["full"] = 0.4
    bandit.n["full"] = 3
    rng = np.random.default_rng(0)
    draws = [bandit.sample_operator(True, rng) for _ in range(200)]
    assert draws.count("full") > draws.count("diff")
    fresh = OperatorBandit(["diff", "full"])
    fresh.set_state(bandit.state())
    assert fresh.ema == bandit.ema and fresh.n == bandit.n
    # seed/no-parent candidates are ignored, unknown operators too
    pop = PopulationStore(PopulationConfig())
    seed = make_candidate("s", 1.0)
    seed.operator = "seed"
    bandit.on_candidate_graded(seed, pop)
    assert bandit.n == {"diff": 0, "full": 3}


def test_lesson_directive_contributor(tmp_path):
    from evoharness.evoplus import LessonDirectiveContributor

    pop, xs = _lessoned_store(tmp_path)  # w: improved, l: regressed lessons
    always = LessonDirectiveContributor(xs, probability=1.0)
    never = LessonDirectiveContributor(xs, probability=0.0)

    # parent l -> its creation lesson is regressed -> repair framing
    parent_l = make_candidate("l", 0.5)
    section = always.contribute(MutationContext(parent_l, [], [], "revise", 5))
    assert "Mutation directive" in section
    assert "HURT fitness" in section
    assert "avoid raising temperature blindly" in section
    # parent w -> improved lesson -> continue-direction framing
    parent_w = make_candidate("w", 1.5)
    section_w = always.contribute(MutationContext(parent_w, [], [], "revise", 5))
    assert "HELPED fitness" in section_w and "variation" in section_w
    # probability 0 never fires; parent without a lesson never fires
    assert never.contribute(MutationContext(parent_l, [], [], "revise", 5)) is None
    parent_p = make_candidate("p", 1.0)
    assert always.contribute(MutationContext(parent_p, [], [], "revise", 5)) is None
    # noise lessons are never promoted to directives
    xs.attach_lesson("l", {"verdict": "noise", "why": "", "advice": "x", "tags": []})
    assert always.contribute(MutationContext(parent_l, [], [], "revise", 5)) is None


def test_lesson_age_is_rendered_as_a_staleness_caveat(tmp_path):
    pop, xs = _lessoned_store(tmp_path)   # entries created at generations 2/3
    contrib = ExperienceContributor(xs, mode="lessons")
    parent = pop.get("p")

    fresh = contrib.contribute(MutationContext(parent, [], [], "revise", 4))
    assert "generation ago" in fresh or "generations ago" in fresh
    assert "verify before relying" not in fresh   # still recent

    stale = contrib.contribute(MutationContext(parent, [], [], "revise", 12))
    assert "9 generations ago" in stale
    assert "verify before relying on it" in stale


def test_regression_soft_penalty_discounts_a_dead_parent():
    """children_count discounts a parent by how MANY children it produced;
    nothing discounted it by how they turned out, so one island re-selected
    the same parent five times while every child regressed."""
    from evoharness.evoplus import RegressionSoftPenalty

    pop = PopulationStore(PopulationConfig())
    parent = make_candidate("p", 1.0)
    pop.insert(parent)
    policy = RegressionSoftPenalty(decay=0.6, floor=0.1)

    assert policy.weight_multiplier(parent) == 1.0        # untried
    for i in range(3):
        child = make_candidate(f"r{i}", 0.5, parent_id="p", generation=i + 1)
        policy.on_candidate_graded(child, pop)
    assert policy.weight_multiplier(parent) == pytest.approx(0.6 ** 3)

    # a single improving child clears the record — the regressed lineage
    # stays selectable, which is the whole point of a soft gate
    winner = make_candidate("w", 1.4, parent_id="p", generation=9)
    policy.on_candidate_graded(winner, pop)
    assert policy.weight_multiplier(parent) == 1.0

    # the discount never reaches zero
    for i in range(40):
        child = make_candidate(f"x{i}", 0.5, parent_id="p", generation=20 + i)
        policy.on_candidate_graded(child, pop)
    assert policy.weight_multiplier(parent) == 0.1

    fresh = RegressionSoftPenalty()
    fresh.set_state(policy.state())
    assert fresh.streaks == policy.streaks


def test_reflector_is_told_when_the_evaluation_never_ran(tmp_path):
    """Run e5s_r2: a child whose problems were cut off at the solver call
    limit was attributed to "over-editing valid solutions", and that
    invented cause was injected into later prompts as advice. The model
    cannot tell a cut-off run from a wrong answer unless the record says
    so — and an "unscored" lesson must not reach any prompt."""
    from evoharness.evoplus import ExperienceContributor, MutationReflector

    pop = PopulationStore(PopulationConfig())
    parent = make_candidate("p", 1.0)
    pop.insert(parent)
    xs = ExperienceStore(tmp_path / "exp.jsonl")
    captured: list[str] = []
    llm = LLMClient(
        transport=_reflect_transport(captured, verdict="unscored"),
        sleep=lambda s: None,
    )
    reflector = MutationReflector(xs, llm, model="m", batch_size=1)

    child = make_candidate("c0", 0.8, parent_id="p", generation=1)
    child.report.visible_metrics = {"unscored_items": 9}
    pop.insert(child)
    xs.on_candidate_graded(child, pop)
    reflector.on_candidate_graded(child, pop)

    assert "UNSCORED: 9 problems produced no answer" in captured[0]
    assert xs.entries[0].lesson["verdict"] == "unscored"
    # An unscored lesson carries no usable direction: it must stay out of
    # the prompt rather than steer the next mutation.
    contributor = ExperienceContributor(xs, mode="lessons")
    rendered = contributor.contribute(
        MutationContext(
            parent=parent,
            operator="revise",
            generation=2,
            archive_inspirations=[],
            top_k_inspirations=[],
        )
    )
    assert rendered is None or "advice-c0" not in rendered


# -- C3 v3: evidence + convergence clustering ---------------------------------

def test_evaluated_entry_carries_report_evidence(tmp_path):
    pop = PopulationStore(PopulationConfig())
    parent = make_candidate("p", 0.85)
    parent.report.visible_metrics = {
        "acc_tier_2": 1.0, "infer_s_tier_0": 121.2, "warm_start": "full",
        "rung": "R2",
    }
    pop.insert(parent)
    xs = ExperienceStore(tmp_path / "exp.jsonl")
    child = make_candidate("c", 0.06, parent_id="p", generation=2)
    child.report.fault = None
    child.report.visible_metrics = {
        "acc_tier_2": 0.03, "infer_s_tier_0": 51.2,
        "warm_start": "parent-partial", "rung": "R2",
    }
    xs.on_candidate_graded(child, pop)
    entry = xs.entries[0]
    # Largest RELATIVE shift first: the accuracy collapse outranks the
    # timing halving; unchanged strings ("rung") never appear.
    assert entry.evidence.index("acc_tier_2 1->0.03") < entry.evidence.index(
        "infer_s_tier_0 121.2->51.2"
    )
    assert "warm_start full->parent-partial" in entry.evidence
    assert "rung" not in entry.evidence
    assert "evidence:" in entry.render()
    # v3 rows roundtrip; v1/v2 rows without the field load as "".
    reloaded = ExperienceStore(tmp_path / "exp.jsonl")
    assert reloaded.entries[0].evidence == entry.evidence


def test_evidence_leads_with_the_fault(tmp_path):
    pop = PopulationStore(PopulationConfig())
    pop.insert(make_candidate("p", 0.85))
    xs = ExperienceStore(tmp_path / "exp.jsonl")
    child = make_candidate("c", 0.0, parent_id="p", passed=False, generation=2)
    child.report.fault = "perturbation-insensitive (L3): randomizing the\nweights left accuracy at 0.33"
    xs.on_candidate_graded(child, pop)
    assert xs.entries[0].evidence.startswith(
        "fault: perturbation-insensitive (L3): randomizing the weights"
    )


def _loss(xs, pop, cid, fitness, summary, generation=2):
    child = make_candidate(cid, fitness, parent_id="p", generation=generation)
    child.operator, child.change_title = "rewrite", f"edit {cid}"
    xs.on_candidate_graded(child, pop)
    # describe_change synthesizes an empty diff for identical texts, so the
    # store fell back to change_summary="", overwrite with the fixture.
    xs.entries[-1].change_summary = summary
    return child


def test_loss_clusters_group_convergent_failures(tmp_path):
    pop = PopulationStore(PopulationConfig())
    pop.insert(make_candidate("p", 0.85))
    xs = ExperienceStore(tmp_path / "exp.jsonl")
    for cid, fit, val in (("a", 0.08, 2), ("b", 0.06, 2), ("c", 0.05, 3), ("d", 0.04, 2)):
        _loss(xs, pop, cid, fit, f"arch.py +1/-1 | added: RADIX_BITS = {val}")
    _loss(xs, pop, "solo", 0.50, "model.py +3/-1 | added: CHUNK = 64")
    clusters = xs.loss_clusters(top_n=2)
    # The four same-identifier deaths are ONE cluster and rank above the
    # singleton despite the singleton's smaller delta magnitude.
    assert [len(c) for c in clusters] == [4, 1]
    line = ExperienceStore.render_cluster(clusters[0])
    assert "tried 4 times INDEPENDENTLY" in line
    assert "helped 0 times" in line
    ctx = _ctx_for(pop)
    contrib = ExperienceContributor(xs, mode="retrieval")
    text = contrib.contribute(ctx)
    assert "tried 4 times INDEPENDENTLY" in text


def test_redeemed_entries_leave_their_cluster(tmp_path):
    pop = PopulationStore(PopulationConfig())
    pop.insert(make_candidate("p", 0.85))
    xs = ExperienceStore(tmp_path / "exp.jsonl")
    for cid in ("a", "b"):
        _loss(xs, pop, cid, 0.05, "arch.py +1/-1 | added: ROUNDS = 1")
    pop.insert(make_candidate("a", 0.05, parent_id="p", generation=2))
    redeemer = make_candidate("r", 0.95, parent_id="a", generation=3)
    xs.on_candidate_graded(redeemer, pop)
    assert [len(c) for c in xs.loss_clusters(top_n=2)] == [1]


def _ctx_for(pop):
    return MutationContext(
        parent=pop.get("p"),
        archive_inspirations=[],
        top_k_inspirations=[],
        operator="revise",
        generation=3,
    )
