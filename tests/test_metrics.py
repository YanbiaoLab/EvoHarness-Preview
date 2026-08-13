from evoharness.core import MetricLog, flatten
from evoharness.core.metrics import MetricPoint


def test_flatten_nested_dicts():
    flat = flatten({"a": 1, "b": {"c": 2, "d": {"e": "x"}}})
    assert flat == {"a": 1, "b/c": 2, "b/d/e": "x"}


def test_log_read_roundtrip(tmp_path):
    path = tmp_path / "metrics.jsonl"
    log = MetricLog(path)
    log.log(0, {"sys": {"fitness": 0.5}, "custom": 1}, candidate_id="abc")
    log.log(1, {"sys": {"fitness": 0.7}})
    log.log(2, {})  # empty is a no-op

    assert log.keys() == ["custom", "sys/fitness"]
    assert log.series("sys/fitness") == [(0, 0.5), (1, 0.7)]
    assert log.summary()["sys/fitness"] == 0.7

    reloaded = MetricLog(path)
    assert reloaded.series("sys/fitness") == [(0, 0.5), (1, 0.7)]
    assert reloaded.points[0].candidate_id == "abc"


def test_memory_only_log():
    log = MetricLog(None)
    log.log(3, {"k": 9})
    assert log.points == [
        MetricPoint(step=3, key="k", value=9, candidate_id=None, ts=log.points[0].ts)
    ]


def test_loop_logs_sys_and_eval_namespaces(tmp_path):
    from test_loop import INITIAL, build_loop, make_rewrite_transport

    loop, store = build_loop(
        make_rewrite_transport(), ["rewrite"], [1.0], tmp_path, generations=5
    )
    log = MetricLog(tmp_path / "metrics.jsonl")
    loop.metric_log = log
    loop.run(INITIAL)

    keys = log.keys()
    assert "sys/fitness" in keys and "sys/best_fitness" in keys
    assert "eval/increments" in keys  # grader's custom visible metric
    # step 0 is the seed, then one point per generation
    steps = [s for s, _ in log.series("sys/fitness")]
    assert steps == [0, 1, 2, 3, 4, 5]
    # best_fitness is monotonically non-decreasing
    best = [v for _, v in log.series("sys/best_fitness")]
    assert all(b2 >= b1 for b1, b2 in zip(best, best[1:]))
    # every point carries the candidate id
    assert all(p.candidate_id for p in log.points)
