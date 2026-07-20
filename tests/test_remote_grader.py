import pytest

from evoharness.evocore.population import Candidate, EvalReport
from evoharness.evocore.remote import (
    EvalInfraError,
    EvalProtocolError,
    HttpReply,
    RemoteEvalConfig,
    RemoteGrader,
)

META = {"protocol_version": 1, "task_version": "toy-v1", "eval_set_version": "full"}


def wire_report(fitness=0.7, passed=True):
    return {
        "schema_version": 1, "fitness": fitness, "passed": passed, "fault": None,
        "visible_metrics": {}, "hidden_metrics": {}, "notes": "",
        "structured_feedback": None, "stdout_log": "", "stderr_log": "",
        "stage_reached": 3, "execution_time": 0.1, "eval_cost_usd": 0.02,
    }


def envelope(status, job_id="job_0001", report=None, error=None,
             task_version="toy-v1", eval_set_version="full"):
    return {"protocol_version": 1, "job_id": job_id, "status": status,
            "task_version": task_version, "eval_set_version": eval_set_version,
            "error": error, "report": report}


class FakeTransport:
    """Scripted transport: each step is an HttpReply, a (status, body) pair,
    or an Exception to raise (ConnectionError = transient network fault)."""

    def __init__(self, script):
        self.calls, self._script = [], list(script)

    def __call__(self, method, url, headers, payload, timeout_s):
        self.calls.append((method, url))
        step = self._script.pop(0)
        if isinstance(step, Exception):
            raise step
        if isinstance(step, HttpReply):
            return step
        status, body = step
        return HttpReply(status, body)


def make_cand(code="print(1)"):
    return Candidate(id="cand-1", code=code, generation=3, parent_id=None,
                     island_idx=0, operator="revise")


def make_grader(script, **cfg_kw):
    cfg = RemoteEvalConfig(base_url="http://x", poll_interval_s=0.0,
                           poll_backoff_cap_s=0.0, **cfg_kw)
    transport = FakeTransport(script)
    grader = RemoteGrader(cfg, transport=transport, sleep=lambda s: None)
    return grader, transport


def test_pregate_short_circuits_no_network(tmp_path):
    gate_report = EvalReport(fitness=0.0, passed=False,
                             fault="L0: banned import", stage_reached=0)
    grader, transport = make_grader([])
    grader.pregate = lambda code: gate_report
    assert grader.grade(make_cand(), tmp_path) is gate_report
    assert transport.calls == []                    # zero round trips


def test_happy_path_metadata_and_audit_trail(tmp_path):
    grader, transport = make_grader([
        (200, META), (202, {"job_id": "job_0001"}),
        (200, envelope("running")),
        (200, envelope("done", report=wire_report())),
    ])
    cand = make_cand()
    report = grader.grade(cand, tmp_path)
    assert report.fitness == 0.7
    assert cand.metadata["task_version"] == "toy-v1"
    assert cand.metadata["eval_set_version"] == "full"
    assert (tmp_path / "remote_reply.json").exists()    # audit trail


def test_http_400_fails_fast_no_retry(tmp_path):
    grader, transport = make_grader([(200, META), (400, {"error": "bad payload"})])
    with pytest.raises(EvalProtocolError, match="400"):
        grader.grade(make_cand(), tmp_path)
    assert len(transport.calls) == 2                # meta + one submit, NO retry


def test_infra_error_retries_with_fresh_job(tmp_path):
    grader, _ = make_grader([
        (200, META),
        (202, {"job_id": "job_0001"}),
        (200, envelope("infra_error", error="disk full")),
        (202, {"job_id": "job_0002"}),              # service made a fresh attempt
        (200, envelope("done", job_id="job_0002", report=wire_report())),
    ])
    assert grader.grade(make_cand(), tmp_path).fitness == 0.7


def test_network_failures_exhaust_to_infra_error(tmp_path):
    grader, transport = make_grader(
        [ConnectionError("refused"), ConnectionError("refused")],
        max_attempts=2,
    )
    with pytest.raises(EvalInfraError, match="2 attempts"):
        grader.grade(make_cand(), tmp_path)
    assert len(transport.calls) == 2                # meta attempt each round


def test_lost_job_404_is_transient_and_resubmits(tmp_path):
    grader, _ = make_grader([
        (200, META),
        (202, {"job_id": "job_0001"}),
        (404, {"error": "unknown job"}),            # service restarted, job lost
        (202, {"job_id": "job_0002"}),
        (200, envelope("done", job_id="job_0002", report=wire_report())),
    ])
    assert grader.grade(make_cand(), tmp_path).passed is True


def test_task_version_drift_raises(tmp_path):
    grader, _ = make_grader([
        (200, META), (202, {"job_id": "job_0001"}),
        (200, envelope("done", report=wire_report(), task_version="toy-v2")),
    ])
    with pytest.raises(EvalProtocolError, match="task_version drift"):
        grader.grade(make_cand(), tmp_path)


def test_eval_set_version_drift_raises(tmp_path):
    grader, _ = make_grader([
        (200, META), (202, {"job_id": "job_0001"}),
        (200, envelope("done", report=wire_report(), eval_set_version="curriculum-L3")),
    ])
    with pytest.raises(EvalProtocolError, match="eval_set_version drift"):
        grader.grade(make_cand(), tmp_path)


def test_meta_protocol_mismatch_raises(tmp_path):
    grader, _ = make_grader([(200, {**META, "protocol_version": 99})])
    with pytest.raises(EvalProtocolError, match="protocol_version mismatch"):
        grader.grade(make_cand(), tmp_path)


def test_malformed_report_raises(tmp_path):
    grader, _ = make_grader([
        (200, META), (202, {"job_id": "job_0001"}),
        (200, envelope("done", report={"schema_version": 1, "oops": True})),
    ])
    with pytest.raises(EvalProtocolError, match="malformed report"):
        grader.grade(make_cand(), tmp_path)
