"""Score one ETP submission by asking the Lean judge.

One problem, one certificate, one verdict from Lean. The candidate writes
`submission.lean`; this sends it to the judge and reports whether the judge
accepted it. Nothing here decides whether a proof is correct — that is the
judge's job, and the whole point of the task is that the answer comes from
Lean rather than from anything that could be talked into agreeing.

**No fallback.** `experiments/etp_stage2/grade.py` picks between the official
judge and an offline Python proxy depending on what is reachable, which is
right for a long evolution run where the judge costs seconds per submission.
It is wrong here. This task exists to show a Lean-backed loop, and a run that
quietly scored itself with no Lean in it would look exactly like a run that
worked: same shape, same numbers, same green. So a missing judge is an error
that stops the run, not a lower gear it shifts into.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

SUBMISSION = "submission.lean"

#: The axioms the official Stage-2 policy permits. Passed explicitly on every
#: request: the judge's own default is zero axioms, which is far stricter than
#: the official rule and rejects certificates that were accepted in the real
#: evaluation. Leaving it out does not fail loudly — it just marks good work
#: wrong.
OFFICIAL_POLICY = {"allowed_axioms": ["propext", "Quot.sound", "Classical.choice"]}

#: Per-submission ceiling. The official spec allows 300s; a demo that hangs
#: for five minutes on one bad certificate is not demonstrating anything.
TIMEOUT_S = int(os.environ.get("ETP_ONE_TIMEOUT_S", "180"))


class JudgeUnavailable(RuntimeError):
    """The judge could not be reached, so nothing can be scored."""


def _problem(candidate_dir: Path) -> dict:
    """The row this task is about, and the verdict its label implies."""

    here = Path(__file__).resolve().parent
    problem_id = json.loads((here / "task.json").read_text())["problem_id"]
    row = json.loads((here / "problem.json").read_text())[problem_id]
    return row


def _judge(row: dict, verdict: str, code: str) -> dict:
    url = os.environ.get("ETP_JUDGE_URL", "").rstrip("/")
    if not url:
        raise JudgeUnavailable(
            "ETP_JUDGE_URL is not set. This task is scored by Lean and has no "
            "offline mode: a run without the judge would report numbers that "
            "look like results and are not."
        )
    problem = {
        key: row[key]
        for key in ("id", "eq1_id", "eq2_id", "equation1", "equation2")
    }
    problem["proof_policy"] = OFFICIAL_POLICY
    payload = json.dumps({
        "problem": problem,
        "verdict": verdict,
        "code": code,
        "timeout_seconds": TIMEOUT_S,
    }).encode()
    request = urllib.request.Request(
        f"{url}/verify", data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S + 60) as response:
            return json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # Reachability is an infrastructure fault, not a wrong answer. Scoring
        # it zero would blame the candidate for the judge being down and put a
        # fabricated data point in the population.
        raise JudgeUnavailable(f"judge at {url} did not answer: {exc}") from exc


def grade(candidate_dir, ctx):
    row = _problem(Path(candidate_dir))
    # The label decides which kind of certificate is being asked for: a True
    # implication wants a proof, a False one wants a counterexample. Sending
    # the wrong verdict is a different question, and the judge answers it
    # correctly by rejecting.
    verdict = "true" if row["label"] else "false"

    path = Path(candidate_dir) / SUBMISSION
    if not path.is_file():
        return {
            "fitness": 0.0,
            "visible_metrics": {"submitted": 0, "accepted": 0},
            "notes": f"no {SUBMISSION} in the workspace",
            "n_units": 1,
            "trustworthy_units": 1,
        }
    code = path.read_text(encoding="utf-8")

    started = time.monotonic()
    result = _judge(row, verdict, code)
    accepted = result.get("status") == "accepted"

    return {
        "fitness": 1.0 if accepted else 0.0,
        "visible_metrics": {
            "submitted": 1,
            "accepted": int(accepted),
            "judge_status": result.get("status"),
            "error_code": result.get("error_code"),
            # Whether Lean actually ran for this answer. A cached result is
            # still the judge's verdict, but a run where every submission was
            # a cache hit has demonstrated nothing about the judge being up.
            "judge_cached": bool(result.get("cached")),
            "judge_elapsed_s": round(time.monotonic() - started, 1),
        },
        # The judge's own words. A candidate that reads this is being told
        # exactly what Lean objected to, which is the feedback this task is
        # built to provide.
        "notes": str(result.get("message") or "")[:2000],
        "n_units": 1,
        "trustworthy_units": 1,
    }
