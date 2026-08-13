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
