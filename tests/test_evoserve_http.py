import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from evoharness.serve import EvalService
from evoharness.serve.http import serve


def toy_grade(code, ctx):
    if "fail" in code:
        return {"fitness": 0.0, "passed": False, "fault": "asked to fail"}
    return 0.7


def make_server(tmp_path, token=None):
    svc = EvalService(
        toy_grade, task_version="toy-v1", eval_set_version="full", base_dir=tmp_path
    )
    srv = serve(svc, port=0, token=token)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


@pytest.fixture()
def base(tmp_path):
    srv, url = make_server(tmp_path)
    yield url
    srv.shutdown()


def request(base, path, body=None, token=None, raw=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = raw if raw is not None else (
        json.dumps(body).encode() if body is not None else None
    )
    req = urllib.request.Request(base + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def make_payload(code="ok", key="k1"):
    return {
        "protocol_version": 1,
        "candidate_id": "cand-1",
        "code": code,
        "idempotency_key": key,
        "hints": None,
    }


def wait_done(base, job_id, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _, env = request(base, f"/v1/evaluations/{job_id}")
        if env["status"] in ("done", "infra_error"):
            return env
        time.sleep(0.02)
    raise TimeoutError(job_id)


def test_meta(base):
    status, body = request(base, "/v1/meta")
    assert status == 200
    assert body["task_version"] == "toy-v1"


def test_submit_poll_roundtrip(base):
    status, body = request(base, "/v1/evaluations", body=make_payload())
    assert status == 202
    env = wait_done(base, body["job_id"])
    assert env["status"] == "done"
    assert env["report"]["fitness"] == 0.7


def test_failed_candidate_is_still_http_200(base):
    _, body = request(base, "/v1/evaluations", body=make_payload(code="fail", key="kf"))
    env = wait_done(base, body["job_id"])
    assert env["report"]["passed"] is False        # candidate failed...
    status, _ = request(base, f"/v1/evaluations/{body['job_id']}")
    assert status == 200                            # ...but the poll succeeded


def test_replay_indistinguishable_202_same_job(base):
    s1, b1 = request(base, "/v1/evaluations", body=make_payload(key="same"))
    s2, b2 = request(base, "/v1/evaluations", body=make_payload(key="same"))
    assert (s1, s2) == (202, 202)
    assert b1["job_id"] == b2["job_id"]


def test_bad_json_400(base):
    status, _ = request(base, "/v1/evaluations", raw=b"{not json")
    assert status == 400


def test_missing_keys_400(base):
    status, _ = request(base, "/v1/evaluations", body={"protocol_version": 1})
    assert status == 400


def test_unknown_job_404_and_unknown_route_404(base):
    assert request(base, "/v1/evaluations/job-9999")[0] == 404
    assert request(base, "/v1/nope")[0] == 404


def test_bearer_auth(tmp_path):
    srv, url = make_server(tmp_path, token="s3cret")
    try:
        assert request(url, "/v1/meta")[0] == 401
        assert request(url, "/v1/meta", token="wrong")[0] == 401
        assert request(url, "/v1/meta", token="s3cret")[0] == 200
    finally:
        srv.shutdown()
