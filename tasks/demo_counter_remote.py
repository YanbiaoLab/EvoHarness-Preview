# EvoHarness original: eval-side twin of tasks/demo_counter.py.
"""demo_counter_remote: the same grading semantics as demo_counter, expressed
as an evoserve grade_fn.

Imports ONLY evoserve (protocol §8: the eval side never touches the
framework). The structured_feedback dict hand-mirrors the wire shape that
evoplus.StructuredFeedback.from_json() reads on the framework side — the
JSON is the shared truth, not a shared class.

Serve it manually with:
    python -m evoharness.evoserve --grade-fn tasks.demo_counter_remote:grade_fn \
        --task-version demo-v1 --eval-set-version items-q0-q4
"""

from __future__ import annotations

from evoharness.evoserve import Grade, GradeContext

ITEMS = ["q0", "q1", "q2", "q3", "q4"]


def grade_fn(code: str, ctx: GradeContext) -> Grade:
    """Item qN passes iff the literal 'qN' appears in the code."""
    items = [
        {
            "item_id": q,
            "passed": q in code,
            "predicted": "yes" if q in code else "?",
            "expected": "yes",
            "error_category": "" if q in code else f"missing-{q}",
        }
        for q in ITEMS
    ]
    solved = sum(i["passed"] for i in items)
    return Grade(
        fitness=solved / len(ITEMS),
        visible_metrics={"solved": solved},
        structured_feedback={
            "schema_version": 1,
            "items": items,
            "summary": f"{solved}/{len(ITEMS)} items solved",
        },
        eval_cost_usd=0.001,
    )
