# EvoHarness original: framework-side client for docs/eval_protocol.md v1.
# Layering: core must not import guard, so the local L0 pre-gate is an
# injected callable (wired by recipes/tasks); stdlib-only HTTP via urllib,
# with the transport injectable like LLMClient/ResolvedTask for offline tests.
"""RemoteGrader: submit/poll HTTP client implementing the Grader protocol.

Failure taxonomy (protocol §5):
- weak candidate        → done + passed=false  → returned as a normal report
- infra failure         → retried; exhausted   → raises EvalInfraError
                          (SearchLoop skips the candidate; NOT inserted)
- non-conformant server → raises EvalProtocolError (hard stop)
- task_version drift    → raises EvalProtocolError (§6: frozen within a run)
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from .population import EvalReport
from .workspace import Workspace

if TYPE_CHECKING:
    from .population import Candidate

PROTOCOL_VERSION = 1
REPORT_SCHEMA_VERSION = 1

class EvalInfraError(RuntimeError):
    """Evaluation infrastructure failed after retries; the candidate carries
    no evaluation signal and must not enter the population."""


class EvalProtocolError(RuntimeError):
    """The evaluation service violated the protocol (bad schema, auth,
    version drift). Not retryable: stop loudly rather than evolve against a
    broken judge."""


@dataclass
class HttpReply:
    status: int
    body: dict | None = None
    headers: dict = field(default_factory=dict)


# transport(method, url, headers, payload, timeout_s) -> HttpReply.
# Network-level failures raise ConnectionError (transient); HTTP error
# statuses are returned as HttpReply so the caller applies the taxonomy.
Transport = Callable[[str, str, dict, dict | None, float], HttpReply]


def _urllib_transport(
    method: str, url: str, headers: dict, payload: dict | None, timeout_s: float
) -> HttpReply:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode() or "null")
            return HttpReply(resp.status, body, dict(resp.headers))
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode() or "null")
        except (ValueError, OSError):
            body = None
        return HttpReply(e.code, body, dict(e.headers or {}))
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise ConnectionError(str(e)) from e


def idempotency_key(code: str, task_version: str) -> str:
    return f"{hashlib.sha256(code.encode()).hexdigest()}:{task_version}"


@dataclass
class RemoteEvalConfig:
    base_url: str
    auth_token: str = ""
    language: str = "python"
    request_timeout_s: float = 30.0
    poll_interval_s: float = 5.0
    poll_backoff_cap_s: float = 30.0
    job_timeout_s: float = 900.0
    max_attempts: int = 3


class RemoteGrader:
    """Implements the Grader protocol against a remote evaluation service.

    pregate: optional callable(workspace) -> EvalReport | None. A non-None
    report short-circuits the remote round-trip (local L0: syntax / edit
    markers / antihack static scan). Wire guard here from the recipe layer.
    The pregate receives the WHOLE workspace (M2.5: cheats can hide in side
    files); what view it takes — main_text() or texts() — is its decision.
    """

    def __init__(
        self,
        cfg: RemoteEvalConfig,
        pregate: Callable[[Workspace], EvalReport | None] | None = None,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.cfg = cfg
        self.pregate = pregate
        self.transport = transport or _urllib_transport
        self.sleep = sleep
        self.clock = clock
        self._meta: dict | None = None  # pinned at first grade (§6)

    # -- Grader protocol ---------------------------------------------------------

    def grade(self, cand: "Candidate", workdir: Path) -> EvalReport:
        if self.pregate is not None:
            local = self.pregate(cand.workspace)
            if local is not None:
                return local

        key = None
        last_error = "no attempt made"
        for attempt in range(self.cfg.max_attempts):
            if attempt:
                self.sleep(min(2.0**attempt, self.cfg.poll_backoff_cap_s))
            try:
                # First contact may fail too: meta fetch belongs to the retry
                # domain, and the idempotency key depends on its task_version.
                meta = self._ensure_meta()
                if key is None:
                    # Keyed on main_text: exact in the single-file degenerate
                    # case. Multi-file genomes need protocol v2's tree_hash key
                    # (two repos differing only outside main_file would collide).
                    key = idempotency_key(
                        cand.workspace.main_text(), meta["task_version"]
                    )
                job_id = self._submit(cand, key)
                envelope = self._poll(job_id)
            except ConnectionError as e:
                last_error = str(e)
                continue
            if envelope.get("status") == "infra_error":
                last_error = envelope.get("error") or "infra_error (unspecified)"
                continue
            return self._accept(envelope, cand, workdir)
        raise EvalInfraError(
            f"evaluation infra failed after {self.cfg.max_attempts} attempts "
            f"(idempotency_key={key}): {last_error}"
        )

    # -- protocol steps ----------------------------------------------------------

    def _headers(self) -> dict:
        headers = {"Accept": "application/json"}
        if self.cfg.auth_token:
            headers["Authorization"] = f"Bearer {self.cfg.auth_token}"
        return headers

    def _ensure_meta(self) -> dict:
        if self._meta is not None:
            return self._meta
        reply = self._request("GET", "/v1/meta", None)
        meta = reply.body or {}
        if meta.get("protocol_version") != PROTOCOL_VERSION:
            raise EvalProtocolError(
                f"protocol_version mismatch: expected {PROTOCOL_VERSION}, "
                f"got {meta.get('protocol_version')!r}"
            )
        for field_name in ("task_version", "eval_set_version"):
            if not meta.get(field_name):
                raise EvalProtocolError(f"/v1/meta missing {field_name!r}")
        self._meta = meta
        return meta

    def _submit(self, cand: "Candidate", key: str) -> str:
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "candidate_id": cand.id,
            # Wire carries program TEXT, never a serialized workspace: the eval
            # side must stay framework-free (eval_protocol.md) and cannot
            # deserialize genomes. Multi-file candidates wait for protocol v2
            # workspace endpoints (git bundle / blob).
            "code": cand.workspace.main_text(),
            "language": self.cfg.language,
            "idempotency_key": key,
            "hints": {"operator": cand.operator, "generation": cand.generation},
        }
        reply = self._request("POST", "/v1/evaluations", payload)
        job_id = (reply.body or {}).get("job_id")
        if not job_id:
            raise EvalProtocolError("submit reply missing job_id")
        return str(job_id)

    def _poll(self, job_id: str) -> dict:
        deadline = self.clock() + self.cfg.job_timeout_s
        wait = self.cfg.poll_interval_s
        while True:
            reply = self._request(
                "GET", f"/v1/evaluations/{job_id}", None, transient_404=True
            )
            envelope = reply.body or {}
            status = envelope.get("status")
            if status in ("done", "infra_error"):
                return envelope
            if status not in ("queued", "running"):
                raise EvalProtocolError(f"unknown job status: {status!r}")
            if self.clock() >= deadline:
                # The eval side may be wedged; treat as transient so the
                # attempt loop resubmits under the same idempotency key.
                raise ConnectionError(
                    f"job {job_id} still {status} after "
                    f"{self.cfg.job_timeout_s:.0f}s"
                )
            retry_after = reply.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else wait
            except ValueError:               # RFC allows HTTP-date form; ignore
                delay = wait
            self.sleep(delay)
            wait = min(wait * 2.0, self.cfg.poll_backoff_cap_s)

    def _accept(
        self, envelope: dict, cand: "Candidate", workdir: Path
    ) -> EvalReport:
        meta = self._meta or {}
        for name in ("task_version", "eval_set_version"):   # §6: both are frozen
            if envelope.get(name) != meta.get(name):
                raise EvalProtocolError(
                    f"{name} drift within run: pinned {meta.get(name)!r}, "
                    f"job reports {envelope.get(name)!r} (§6: evaluation "
                    "standards are frozen for the whole run)"
                )
        raw = envelope.get("report")
        if not isinstance(raw, dict):
            raise EvalProtocolError("done job carries no report object")
        if raw.get("schema_version") != REPORT_SCHEMA_VERSION:
            raise EvalProtocolError(
                f"report schema_version {raw.get('schema_version')!r} != "
                f"{REPORT_SCHEMA_VERSION}"
            )
        try:
            report = EvalReport.from_json(raw)
        except (ValueError, TypeError) as e:
            raise EvalProtocolError(f"malformed report: {e}") from e

        cand.metadata["task_version"] = envelope.get("task_version")
        cand.metadata["eval_set_version"] = envelope.get("eval_set_version")
        (workdir / "remote_reply.json").write_text(
            json.dumps(envelope, indent=2, default=str)
        )
        return report

    def _request(
        self, method: str, path: str, payload: dict | None,
        transient_404: bool = False,
    ) -> HttpReply:
        url = self.cfg.base_url.rstrip("/") + path
        reply = self.transport(
            method, url, self._headers(), payload, self.cfg.request_timeout_s
        )
        if reply.status >= 500:
            raise ConnectionError(f"{method} {path} -> HTTP {reply.status}")
        if reply.status == 404 and transient_404:
            # Poll-side 404 = job lost (eval service restarted, in-memory job
            # table gone). Resubmitting under the same idempotency key is safe.
            raise ConnectionError(f"{method} {path} -> job lost (HTTP 404)")
        if reply.status >= 400:
            raise EvalProtocolError(f"{method} {path} -> HTTP {reply.status}")
        return reply
