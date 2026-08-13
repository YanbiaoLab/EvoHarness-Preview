from conftest import make_candidate
from evoharness.core.interfaces import MutationContext
from evoharness.evoplus import ResourceLedgerContributor


def _cand(cid, fitness, **metrics):
    c = make_candidate(cid, fitness)
    c.report.visible_metrics = dict(metrics)
    return c


def _ctx(parent, *inspirations):
    return MutationContext(
        parent=parent,
        archive_inspirations=list(inspirations),
        top_k_inspirations=[],
        operator="recombine",
        generation=3,
    )


def test_silent_when_the_parent_does_not_report_the_metric():
    led = ResourceLedgerContributor("eval/solver_bytes")
    assert led.contribute(_ctx(_cand("p", 0.8))) is None


def test_silent_when_everything_on_hand_costs_the_same():
    """A one-value table states no relationship the model could act on."""
    led = ResourceLedgerContributor("bytes")
    parent = _cand("p", 0.8, bytes=499_000)
    twin = _cand("q", 0.7, bytes=499_000)
    assert led.contribute(_ctx(parent, twin)) is None


def test_lists_cheaper_donors_with_headroom_and_score():
    led = ResourceLedgerContributor(
        "bytes", cap=500_000, quality_metric="raw", unit="bytes")
    parent = _cand("champion", 0.848, bytes=499_921, raw=0.848)
    donor = _cand("compact00001", 0.0, bytes=490_689, raw=0.777)
    out = led.contribute(_ctx(parent, donor))
    assert "Cap 500,000 bytes" in out
    assert "leaving 79" in out
    # The donor was gated to fitness 0 for a regression; the ledger must show
    # what it actually achieved, since that is why it is worth recombining with.
    assert "0.7770" in out
    assert "compact00001" in out
    assert "this parent" in out
    # Cheapest first: reclamation targets should read top-down. Compare inside
    # the table, since the header states the parent's own cost first.
    table = out.split("who\n", 1)[1]
    assert table.index("490,689") < table.index("499,921")


def test_falls_back_to_fitness_when_no_quality_metric_declared():
    led = ResourceLedgerContributor("bytes", cap=500_000)
    out = led.contribute(
        _ctx(_cand("p", 0.80, bytes=499_000), _cand("q", 0.65, bytes=480_000)))
    assert "0.6500" in out and "0.8000" in out


def test_reads_the_store_when_selection_cannot_reach_cheap_candidates():
    """The failure this exists for.

    Inspiration selection ranks on `fitness`. When a gate zeroes every cheap
    candidate, the draw is all ceiling-pinned programs, the table has one
    distinct cost, and the ledger would go silent -- exactly when the model
    most needs to know a cheaper program exists.
    """
    from evoharness.core import PopulationConfig, PopulationStore

    cfg = PopulationConfig(
        archive_feature_metric="bytes",
        archive_feature_quality="raw",
        archive_feature_bucket=10_000.0,
    )
    store = PopulationStore(cfg)
    gated = _cand("reclaimer001", 0.0, bytes=467_958, raw=0.8422)
    store.insert(gated)

    parent = _cand("champion0001", 0.848, bytes=499_921, raw=0.848)
    twin = _cand("alsoceiling01", 0.847, bytes=499_454, raw=0.847)

    blind = ResourceLedgerContributor("bytes", cap=500_000, quality_metric="raw")
    assert "reclaimer001" not in (blind.contribute(_ctx(parent, twin)) or "")

    led = ResourceLedgerContributor(
        "bytes", store=store, cap=500_000, quality_metric="raw")
    out = led.contribute(_ctx(parent, twin))
    assert "reclaimer001" in out
    assert "467,958" in out and "32,042" in out  # cost and headroom
    assert "0.8422" in out                        # pre-gate score, not the 0


def test_store_without_the_expected_api_is_ignored_not_fatal():
    led = ResourceLedgerContributor("bytes", store=object(), cap=500_000)
    parent = _cand("p", 0.8, bytes=499_000)
    assert led.contribute(_ctx(parent, _cand("q", 0.7, bytes=480_000))) is not None
