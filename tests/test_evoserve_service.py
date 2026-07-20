import threading

import pytest

from evoharness.evoserve import EvalService


def make_payload(code="print(1)", key="k1", **hints):
    return {
        "protocol_version": 1,
        "candidate_id": "cand-1",
        "code": code,
        "idempotency_key": key,
        "hints": hints or None,
    }


def make_service(grade_fn, tmp_path, **kw):
    return EvalService(
        grade_fn, task_version="toy-v1", eval_set_version="full",
        base_dir=tmp_path, **kw,
    )


def test_happy_path_envelope(tmp_path):
    svc = make_service(lambda code, ctx: 0.7, tmp_path)
    job_id = svc.submit(make_payload())
    env = svc.wait(job_id)
    assert env["status"] == "done"
    assert env["report"]["fitness"] == 0.7
    assert env["task_version"] == "toy-v1"
    assert env["error"] is None


def test_in_flight_duplicate_returns_same_job(tmp_path):
    gate = threading.Event()
    calls = []

    def slow_grade(code, ctx):
        calls.append(1)
        gate.wait(5)
        return 1.0

    svc = make_service(slow_grade, tmp_path)
    a = svc.submit(make_payload(key="same"))
    b = svc.submit(make_payload(key="same"))   # still running -> same job
    gate.set()
    assert a == b
    svc.wait(a)
    assert len(calls) == 1                      # executed exactly once


def test_grade_fn_exception_is_verdict_not_infra(tmp_path):
    def boom(code, ctx):
        raise RuntimeError("task author bug")

    svc = make_service(boom, tmp_path)
    env = svc.wait(svc.submit(make_payload()))
    assert env["status"] == "done"              # NOT infra_error
    assert env["report"]["passed"] is False
    assert "task author bug" in env["report"]["stderr_log"]
    assert env["report"]["stage_reached"] == 0


def test_bad_payload_and_version_rejected(tmp_path):
    svc = make_service(lambda c, x: 1.0, tmp_path)
    with pytest.raises(ValueError, match="[Mm]issing"):
        svc.submit({"protocol_version": 1})
    with pytest.raises(ValueError, match="version"):
        svc.submit({**make_payload(), "protocol_version": 99})


def test_workdir_kept_on_failure_cleaned_on_success(tmp_path):
    def grade(code, ctx):
        (ctx.workdir / "trace.txt").write_text("debug me")
        return {"fitness": 0.0, "passed": "fail" not in code}

    svc = make_service(grade, tmp_path)
    ok = svc.wait(svc.submit(make_payload(code="good", key="a")))
    bad = svc.wait(svc.submit(make_payload(code="fail", key="b")))
    dirs = list(tmp_path.iterdir())
    assert len(dirs) == 1                       # success cleaned, failure kept
    assert (dirs[0] / "trace.txt").read_text() == "debug me"
    assert ok["report"]["passed"] and not bad["report"]["passed"]


def test_unknown_job_raises(tmp_path):
    svc = make_service(lambda c, x: 1.0, tmp_path)
    with pytest.raises(KeyError):
        svc.poll("job-9999")


def test_infra_error_does_not_satisfy_replay(tmp_path):
    from evoharness.evoserve import InfraError

    state = {"first": True}

    def flaky(code, ctx):
        if state["first"]:
            state["first"] = False
            raise InfraError("transient dependency outage")
        return 0.9

    svc = make_service(flaky, tmp_path)
    a = svc.wait(svc.submit(make_payload(key="retry-me")))
    assert a["status"] == "infra_error"
    b = svc.wait(svc.submit(make_payload(key="retry-me")))   # same key!
    assert b["status"] == "done" and b["report"]["fitness"] == 0.9
    assert a["job_id"] != b["job_id"]                        # fresh job, not the corpse


def test_infra_error_is_not_a_verdict(tmp_path):
    from evoharness.evoserve import InfraError

    def flaky_dependency(code, ctx):
        raise InfraError("sandbox fusion unreachable")

    svc = make_service(flaky_dependency, tmp_path)
    env = svc.wait(svc.submit(make_payload()))
    assert env["status"] == "infra_error"           # NOT done+passed=false
    assert "sandbox fusion unreachable" in env["error"]
    assert env["report"] is None                    # no verdict was made
