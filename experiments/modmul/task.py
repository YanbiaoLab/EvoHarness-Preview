"""Framework-side task bundle for modmul.

Consumer layer: this file IS the composition point for the task, so it may
import both the framework and the eval side. Two grading paths:

- default: GradeFnGrader — grade_fn runs in-process (no HTTP; identical
  verdicts; simplest for single-machine runs);
- remote:  set MODMUL_EVAL_URL and the bundle wires a RemoteGrader with the
  AntiHackScanner pregate instead (production path, GPU box via serve.sh).

Env knobs:
  MODMUL_QUICK=1      laptop-scale seed (TRAIN_STEPS 3000 -> 500)
  MODMUL_EVAL_URL     e.g. http://gpu-box:8321  (switches to RemoteGrader)
"""

from __future__ import annotations

import os
from pathlib import Path

from evoharness.evocore import EvalReport
from evoharness.evoserve import GradeContext, coerce_grade
from recipes.common import TaskBundle

_HERE = Path(__file__).resolve().parent


class GradeFnGrader:
    """In-process adapter: an evoserve grade_fn used directly as a Grader.
    Same verdicts as the remote path minus transport. Thread-safe under
    eval_batch_size > 1: grade_fn isolates per-candidate module names and
    runs training in its own subprocess."""

    def __init__(self, fn):
        self._fn = fn

    def grade(self, cand, workdir: Path) -> EvalReport:
        grade = self._fn(
            cand.code, GradeContext(candidate_id=cand.id, workdir=Path(workdir))
        )
        return EvalReport.from_json(coerce_grade(grade))


def _make_pregate():
    from evoharness.evocore.workspace import Workspace
    from evoharness.evoguard import AntiHackScanner

    scanner = AntiHackScanner()

    def pregate(ws: Workspace) -> EvalReport | None:
        # Whole-genome scan (M2.5): cheats can hide in side files.
        findings = scanner.scan_files(ws.texts())
        if findings:
            f = findings[0]
            return EvalReport(fitness=0.0, passed=False, stage_reached=0,
                              fault=f"L0 {f.rule}@{f.path}:{f.lineno}: {f.detail}")
        return None

    return pregate


def _make_grader():
    url = os.environ.get("MODMUL_EVAL_URL")
    if url:
        from evoharness.evocore.remote import RemoteEvalConfig, RemoteGrader

        cfg = RemoteEvalConfig(
            base_url=url,
            auth_token=os.environ.get("EVOSERVE_TOKEN", ""),
            job_timeout_s=1200.0,          # train budget + queue headroom
        )
        return RemoteGrader(cfg, pregate=_make_pregate())
    from modmul.grade import grade_fn

    return GradeFnGrader(grade_fn)


def make_task() -> TaskBundle:
    seed = (_HERE / "seeds" / "serial_ar.py").read_text()
    horner = (_HERE / "seeds" / "horner_cell.py").read_text()
    if os.environ.get("MODMUL_QUICK") == "1":
        seed = seed.replace("TRAIN_STEPS = 3000", "TRAIN_STEPS = 500")
        horner = horner.replace("TRAIN_STEPS = 3000", "TRAIN_STEPS = 500")
    return TaskBundle(
        grader=_make_grader(),
        initial_code=seed,
        # gpu_run1: the LLM tried hard to reach Horner-style structures but
        # could not land one in single-shot edits (34/48 crashed). Give
        # island 1 a WORKING implementation of that family to evolve from.
        extra_seeds=[horner],
        task_sys_msg=(_HERE / "sys_msg.md").read_text(),
        research_brief=(_HERE / "research_msg.md").read_text(),
    )
