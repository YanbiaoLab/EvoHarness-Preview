from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
import threading
import time
import traceback


from evoharness.evoserve.grading import GradeContext, GradeFn, InfraError, coerce_grade


PROTOCOL_VERSION = 1

_REQUIRED_SUBMIT_KEYS = ("protocol_version", "candidate_id", "code", "idempotency_key")

@dataclass
class Job:
    job_id: str
    status: str = "queued"
    report: dict | None = None
    error: str | None = None
    workdir: Path | None = None

class EvalService:
    def __init__(
            self,
            grade_fn: GradeFn,
            task_version: str,
            eval_set_version: str,
            max_workers: int = 2,
            keep_workdir: str = "failed",
            base_dir: Path | None = None
 
    ):
        self._grade_fn = grade_fn
        self._task_version = task_version
        self._eval_set_version = eval_set_version
        self._keep_workdirs = keep_workdir
        self._base_dir = base_dir
        self._jobs: dict[str, Job] = {}
        self._by_key: dict[str, str] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._counter = 0

    
    def meta(self) -> dict:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "task_version": self._task_version,
            "eval_set_version": self._eval_set_version
        }
    
    def submit(self, payload: dict) -> str:
        missing = [k for k in _REQUIRED_SUBMIT_KEYS if k not in payload]
        if missing:
            raise ValueError(f"Missing required keys: {missing}")
        
        if payload["protocol_version"] != PROTOCOL_VERSION:
            raise ValueError(f"Unsupported protocol version: {payload['protocol_version']}")

        with self._lock:
            key = payload["idempotency_key"]
            if key in self._by_key:              # replay OR in-flight duplicate
                prior = self._jobs[self._by_key[key]]
                if prior.status != "infra_error":
                    return prior.job_id
               
            self._counter += 1
            job = Job(job_id=f"job_{self._counter:04d}")
            self._jobs[job.job_id] = job
            self._by_key[key] = job.job_id

        self._pool.submit(self._run, job, payload)

        return job.job_id

    def poll(self, job_id: str) -> dict:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return {
                "protocol_version": PROTOCOL_VERSION,
                "job_id": job.job_id,
                "status": job.status,
                "task_version": self._task_version,
                "eval_set_version": self._eval_set_version,
                "error": job.error,
                "report": job.report,
            }
    
    def wait(self, job_id: str, timeout_s: float = 30.0) -> dict:
        """Convenience for tests/CLI; RemoteGrader does its own polling."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            envelope = self.poll(job_id)
            if envelope["status"] in ("done", "infra_error"):
                return envelope
            time.sleep(0.01)
        raise TimeoutError(f"job {job_id} not terminal after {timeout_s}s")
    
    def _cleanup(self, job: Job) -> None:
        failed = not job.report["passed"]
        if self._keep_workdirs == "always" or (self._keep_workdirs == "failed" and failed):
            return
        shutil.rmtree(job.workdir, ignore_errors=True)

    def _run(self, job: Job, payload: dict) -> None:
        # --- infra domain: retry might help, no verdict on the candidate ---
        try:
             workdir = Path(tempfile.mkdtemp(prefix=f"{job.job_id}-", dir=self._base_dir))
             with self._lock:
                 job.workdir = workdir
                 job.status = "running"
        except Exception as exc:
            with self._lock:
                job.status = "infra_error"
                job.error = f"workdir setup failed: {exc}"
            return
        
        hints = payload.get("hints") or {}
        started = time.monotonic()
        try:
            ctx = GradeContext(
                candidate_id=payload["candidate_id"],
                workdir=workdir,
                operator=hints.get("operator"),
                generation=hints.get("generation"),
            )
            report = coerce_grade(self._grade_fn(payload["code"], ctx))
        except InfraError as exc:
            # Dependency failure (sandbox service down etc.): no verdict on the
            # candidate — retryable, must not enter the population as a failure.
            with self._lock:
                job.status = "infra_error"
                job.error = f"grade_fn dependency failed: {exc}"
            return
        except Exception:
            report = coerce_grade(
                {
                    "fitness": 0.0,
                    "passed": False,
                    "fault": "uncaught exception in grade_fn",
                    "stderr_log": traceback.format_exc(),
                    "stage_reached": 0
                }
            )
        if not report["execution_time"]:
            report["execution_time"] = time.monotonic() - started
        
        with self._lock:
            job.report = report
            job.status = "done"

        self._cleanup(job)
        
