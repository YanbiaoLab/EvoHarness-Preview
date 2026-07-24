"""Optional HTTP transport for the task-owned evaluation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Mapping

from ..protocol import BenchmarkSpec, default_spec_path
from .contract import (
    CandidateEvaluation,
    EvaluationBackend,
    EvaluationProtocolError,
    EvaluationUnavailable,
)


PROTOCOL_VERSION = 2


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or value != path.as_posix()
    ):
        raise ValueError(f"unsafe candidate path: {value!r}")
    return path


def snapshot_directory(root: Path) -> dict[str, str]:
    """Create the canonical, text-only multi-file candidate payload."""

    root = Path(root).resolve()
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.is_symlink():
            raise EvaluationProtocolError(
                f"candidate snapshot cannot contain symlink: {path}"
            )
        relative = path.relative_to(root).as_posix()
        _safe_relative_path(relative)
        try:
            files[relative] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise EvaluationProtocolError(
                f"candidate snapshot contains non-UTF-8 file: {relative}"
            ) from exc
    if not files:
        raise EvaluationProtocolError("candidate snapshot is empty")
    return files


def workspace_tree_hash(files: Mapping[str, str]) -> str:
    payload = json.dumps(dict(files), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def idempotency_key(
    tree_hash: str,
    protocol_fingerprint: str,
    profile: str,
) -> str:
    return f"{tree_hash}:{protocol_fingerprint}:{profile}"


@dataclass
class EvaluationJob:
    job_id: str
    status: str = "queued"
    evaluation: dict[str, object] | None = None
    error: str | None = None


class EvaluationService:
    """Asynchronous service core; HTTP is only a stateless shell around it."""

    def __init__(
        self,
        backend: EvaluationBackend,
        spec: BenchmarkSpec,
        *,
        profile: str = "train",
        max_workers: int = 2,
        base_dir: Path | None = None,
    ):
        spec.splits.ids(profile)
        self.backend = backend
        self.spec = spec
        self.profile = profile
        self.base_dir = Path(base_dir).resolve() if base_dir is not None else None
        self._jobs: dict[str, EvaluationJob] = {}
        self._by_key: dict[str, str] = {}
        self._counter = 0
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_workers)

    def meta(self) -> dict[str, object]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "experiment_id": self.spec.benchmark_id,
            "protocol_fingerprint": self.spec.fingerprint,
            "evaluation_profile": self.profile,
        }

    def submit(self, payload: Mapping[str, object]) -> str:
        required = {
            "protocol_version",
            "candidate_id",
            "protocol_fingerprint",
            "tree_hash",
            "files",
            "idempotency_key",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"missing required evaluation fields: {missing}")
        if payload["protocol_version"] != PROTOCOL_VERSION:
            raise ValueError("unsupported evaluation protocol version")
        if payload["protocol_fingerprint"] != self.spec.fingerprint:
            raise ValueError("protocol fingerprint mismatch")
        candidate_id = payload["candidate_id"]
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError("candidate_id must be non-empty")
        raw_files = payload["files"]
        if not isinstance(raw_files, Mapping) or not raw_files:
            raise ValueError("files must be a non-empty object")
        files: dict[str, str] = {}
        total_bytes = 0
        for name, content in raw_files.items():
            if not isinstance(name, str) or not isinstance(content, str):
                raise ValueError("candidate files must map paths to UTF-8 text")
            _safe_relative_path(name)
            total_bytes += len(content.encode())
            files[name] = content
        max_bytes = self.spec.candidate.max_file_bytes * max(
            1,
            len(self.spec.candidate.mutable_files) + 1,
        )
        if total_bytes > max_bytes:
            raise ValueError("candidate workspace exceeds request size limit")
        observed_hash = workspace_tree_hash(files)
        if payload["tree_hash"] != observed_hash:
            raise ValueError("candidate tree hash mismatch")
        expected_key = idempotency_key(
            observed_hash,
            self.spec.fingerprint,
            self.profile,
        )
        if payload["idempotency_key"] != expected_key:
            raise ValueError("invalid idempotency key")

        with self._lock:
            prior_id = self._by_key.get(expected_key)
            if prior_id is not None:
                prior = self._jobs[prior_id]
                if prior.status != "infra_error":
                    return prior_id
            self._counter += 1
            job = EvaluationJob(job_id=f"eval_{self._counter:06d}")
            self._jobs[job.job_id] = job
            self._by_key[expected_key] = job.job_id
        self._pool.submit(self._run, job, candidate_id, files)
        return job.job_id

    def poll(self, job_id: str) -> dict[str, object]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return {
                **self.meta(),
                "job_id": job.job_id,
                "status": job.status,
                "evaluation": job.evaluation,
                "error": job.error,
            }

    def wait(self, job_id: str, timeout_s: float = 30.0) -> dict[str, object]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            result = self.poll(job_id)
            if result["status"] in {"done", "infra_error", "protocol_error"}:
                return result
            time.sleep(0.01)
        raise TimeoutError(f"evaluation job {job_id} did not finish")

    def close(self) -> None:
        self._pool.shutdown(wait=True)

    def _run(
        self,
        job: EvaluationJob,
        candidate_id: str,
        files: Mapping[str, str],
    ) -> None:
        workdir = Path(
            tempfile.mkdtemp(prefix=f"{job.job_id}-", dir=self.base_dir)
        )
        candidate_root = workdir / "candidate"
        result_dir = workdir / "result"
        candidate_root.mkdir()
        try:
            with self._lock:
                job.status = "running"
            for name, content in files.items():
                path = candidate_root / _safe_relative_path(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            evaluation = self.backend.evaluate_directory(
                candidate_id=candidate_id,
                candidate_root=candidate_root,
                split=self.profile,
                output_dir=result_dir,
            )
            if not isinstance(evaluation, CandidateEvaluation):
                raise EvaluationProtocolError(
                    "evaluation backend did not return CandidateEvaluation"
                )
            if (
                evaluation.candidate_id != candidate_id
                or evaluation.split != self.profile
            ):
                raise EvaluationProtocolError(
                    "evaluation backend returned mismatched candidate or split"
                )
        except EvaluationUnavailable as exc:
            with self._lock:
                job.status = "infra_error"
                job.error = str(exc)
            return
        except EvaluationProtocolError as exc:
            with self._lock:
                job.status = "protocol_error"
                job.error = str(exc)
            return
        except Exception as exc:
            with self._lock:
                job.status = "infra_error"
                job.error = f"evaluation service failed: {type(exc).__name__}: {exc}"
            return
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        with self._lock:
            job.evaluation = evaluation.to_dict()
            job.status = "done"


@dataclass(frozen=True)
class HttpReply:
    status: int
    body: dict[str, object] | None = None
    headers: Mapping[str, str] = field(default_factory=dict)


Transport = Callable[
    [str, str, Mapping[str, str], Mapping[str, object] | None, float],
    HttpReply,
]


def _urllib_transport(
    method: str,
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, object] | None,
    timeout_s: float,
) -> HttpReply:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers=dict(headers),
    )
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            body = json.loads(response.read().decode() or "null")
            return HttpReply(response.status, body, dict(response.headers))
    except urllib.error.HTTPError as exc:
        try:
            body = json.loads(exc.read().decode() or "null")
        except (ValueError, OSError):
            body = None
        return HttpReply(exc.code, body, dict(exc.headers or {}))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise EvaluationUnavailable(str(exc)) from exc


class HttpEvaluationBackend:
    """Client-side backend implementing the same EvaluationBackend contract."""

    def __init__(
        self,
        base_url: str,
        spec: BenchmarkSpec,
        *,
        profile: str = "train",
        auth_token: str = "",
        request_timeout_s: float = 30.0,
        poll_interval_s: float = 1.0,
        job_timeout_s: float = 3600.0,
        max_attempts: int = 3,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        spec.splits.ids(profile)
        self.base_url = base_url.rstrip("/")
        self.spec = spec
        self.profile = profile
        self.auth_token = auth_token
        self.request_timeout_s = request_timeout_s
        self.poll_interval_s = poll_interval_s
        self.job_timeout_s = job_timeout_s
        self.max_attempts = max_attempts
        self.transport = transport or _urllib_transport
        self.sleep = sleep
        self.clock = clock
        self._meta: dict[str, object] | None = None

    def evaluate_directory(
        self,
        *,
        candidate_id: str,
        candidate_root: Path,
        split: str,
        output_dir: Path | None = None,
    ) -> CandidateEvaluation:
        if split != self.profile:
            raise EvaluationProtocolError(
                f"HTTP backend is pinned to {self.profile!r}, not {split!r}"
            )
        files = snapshot_directory(candidate_root)
        tree_hash = workspace_tree_hash(files)
        key = idempotency_key(tree_hash, self.spec.fingerprint, self.profile)
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "candidate_id": candidate_id,
            "protocol_fingerprint": self.spec.fingerprint,
            "tree_hash": tree_hash,
            "files": files,
            "idempotency_key": key,
        }
        last_error = "no attempt made"
        for attempt in range(self.max_attempts):
            if attempt:
                self.sleep(min(2.0**attempt, 10.0))
            try:
                self._ensure_meta()
                reply = self._request("POST", "/v2/evaluations", payload)
                job_id = (reply.body or {}).get("job_id")
                if not isinstance(job_id, str) or not job_id:
                    raise EvaluationProtocolError(
                        "evaluation submit response is missing job_id"
                    )
                envelope = self._poll(job_id)
            except EvaluationUnavailable as exc:
                last_error = str(exc)
                continue
            if envelope.get("status") == "infra_error":
                last_error = str(envelope.get("error") or "infra_error")
                continue
            if envelope.get("status") == "protocol_error":
                raise EvaluationProtocolError(
                    str(envelope.get("error") or "evaluation protocol error")
                )
            raw = envelope.get("evaluation")
            if not isinstance(raw, Mapping):
                raise EvaluationProtocolError(
                    "completed evaluation job is missing evaluation result"
                )
            try:
                evaluation = CandidateEvaluation.from_dict(raw)
            except (KeyError, TypeError, ValueError) as exc:
                raise EvaluationProtocolError(
                    f"invalid CandidateEvaluation response: {exc}"
                ) from exc
            if evaluation.split != split:
                raise EvaluationProtocolError(
                    "evaluation service returned mismatched split"
                )
            if evaluation.candidate_id != candidate_id:
                # Content-addressed jobs can be reused across candidate labels.
                evaluation = replace(evaluation, candidate_id=candidate_id)
            if output_dir is not None:
                evaluation.write(Path(output_dir) / "evaluation.json")
            return evaluation
        raise EvaluationUnavailable(
            f"evaluation service failed after {self.max_attempts} attempts: "
            f"{last_error}"
        )

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    def _ensure_meta(self) -> None:
        if self._meta is not None:
            return
        reply = self._request("GET", "/v2/meta", None)
        meta = reply.body or {}
        expected = {
            "protocol_version": PROTOCOL_VERSION,
            "experiment_id": self.spec.benchmark_id,
            "protocol_fingerprint": self.spec.fingerprint,
            "evaluation_profile": self.profile,
        }
        for name, value in expected.items():
            if meta.get(name) != value:
                raise EvaluationProtocolError(
                    f"evaluation service {name} mismatch: {meta.get(name)!r}"
                )
        self._meta = meta

    def _poll(self, job_id: str) -> Mapping[str, object]:
        deadline = self.clock() + self.job_timeout_s
        while True:
            reply = self._request("GET", f"/v2/evaluations/{job_id}", None)
            envelope = reply.body or {}
            status = envelope.get("status")
            if status in {"done", "infra_error", "protocol_error"}:
                return envelope
            if status not in {"queued", "running"}:
                raise EvaluationProtocolError(
                    f"unknown evaluation job status: {status!r}"
                )
            if self.clock() >= deadline:
                raise EvaluationUnavailable(
                    f"evaluation job {job_id} timed out"
                )
            self.sleep(self.poll_interval_s)

    def _request(
        self,
        method: str,
        path: str,
        payload: Mapping[str, object] | None,
    ) -> HttpReply:
        reply = self.transport(
            method,
            self.base_url + path,
            self._headers(),
            payload,
            self.request_timeout_s,
        )
        if reply.status in {408, 429} or reply.status >= 500:
            raise EvaluationUnavailable(f"evaluation HTTP {reply.status}")
        if reply.status < 200 or reply.status >= 300:
            raise EvaluationProtocolError(
                f"evaluation HTTP {reply.status}: {reply.body}"
            )
        return reply


class EvaluationHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address,
        service: EvaluationService,
        token: str | None = None,
    ):
        super().__init__(address, _EvaluationHandler)
        self.service = service
        self.token = token


class _EvaluationHandler(BaseHTTPRequestHandler):
    server: EvaluationHTTPServer

    def log_message(self, fmt, *args):
        pass

    def _authorized(self) -> bool:
        token = self.server.token
        return token is None or self.headers.get("Authorization") == f"Bearer {token}"

    def _send_json(self, status: int, body: Mapping[str, object]) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self._authorized():
            self._send_json(401, {"error": "unauthorized"})
            return
        if self.path == "/v2/meta":
            self._send_json(200, self.server.service.meta())
            return
        prefix = "/v2/evaluations/"
        if self.path.startswith(prefix):
            try:
                result = self.server.service.poll(self.path.removeprefix(prefix))
            except KeyError:
                self._send_json(404, {"error": "unknown evaluation job"})
                return
            self._send_json(200, result)
            return
        self._send_json(404, {"error": "unknown route"})

    def do_POST(self):
        if not self._authorized():
            self._send_json(401, {"error": "unauthorized"})
            return
        if self.path != "/v2/evaluations":
            self._send_json(404, {"error": "unknown route"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            job_id = self.server.service.submit(payload)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": str(exc)})
            return
        self._send_json(202, {"job_id": job_id})


def serve(
    service: EvaluationService,
    *,
    host: str = "127.0.0.1",
    port: int = 8322,
    token: str | None = None,
) -> EvaluationHTTPServer:
    return EvaluationHTTPServer((host, port), service, token)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve one pinned IMO evaluation profile over HTTP.",
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=default_spec_path())
    parser.add_argument(
        "--profile",
        choices=("train", "validation", "test"),
        required=True,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8322)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--worker-timeout-s", type=float, default=3600.0)
    parser.add_argument(
        "--allow-unauthenticated",
        action="store_true",
        help="allow requests without EVOHARNESS_EVAL_TOKEN",
    )
    args = parser.parse_args(argv)

    token = os.environ.get("EVOHARNESS_EVAL_TOKEN")
    if not token and not args.allow_unauthenticated:
        parser.error(
            "set EVOHARNESS_EVAL_TOKEN or pass --allow-unauthenticated"
        )
    spec = BenchmarkSpec.load(args.spec)
    spec.verify_workspace(args.project_root)
    from .worker import LocalProcessBackend

    backend = LocalProcessBackend(
        project_root=args.project_root,
        spec_path=args.spec,
        timeout_s=args.worker_timeout_s,
    )
    service = EvaluationService(
        backend,
        spec,
        profile=args.profile,
        max_workers=args.max_workers,
    )
    server = serve(
        service,
        host=args.host,
        port=args.port,
        token=token,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EvaluationHTTPServer",
    "EvaluationJob",
    "EvaluationService",
    "HttpEvaluationBackend",
    "HttpReply",
    "PROTOCOL_VERSION",
    "idempotency_key",
    "main",
    "serve",
    "snapshot_directory",
    "workspace_tree_hash",
]
