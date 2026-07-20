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
    parent.report.structured_feedback = feedback_fixture("1000").to_json()
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


def test_experience_store_records_and_queries(tmp_path):
    pop, xs = _store_with_lineage(tmp_path)
    assert len(xs.entries) == 2
    wins, losses = xs.query(["logic"])
    assert [e.change_title for e in wins] == ["add cache"]
    assert [e.change_title for e in losses] == ["raise temperature"]
    assert xs.query(["unrelated-cat"]) == ([], [])
    # jsonl persistence roundtrip
    reloaded = ExperienceStore(tmp_path / "exp.jsonl")
    assert len(reloaded.entries) == 2


def test_experience_contributor_retrieval(tmp_path):
    pop, xs = _store_with_lineage(tmp_path)
    contrib = ExperienceContributor(xs, mode="retrieval")
    parent = pop.get("p")
    section = contrib.contribute(MutationContext(parent, [], [], "revise", 4))
    assert "Experience from similar failure modes" in section
    assert "add cache" in section and "helped" in section
    assert "raise temperature" in section and "did NOT help" in section

    off = ExperienceContributor(xs, mode="off")
    assert off.contribute(MutationContext(parent, [], [], "revise", 4)) is None


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
