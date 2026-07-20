# EvoHarness original: fully offline demo task (no API key, no network).
# Referenced by recipes/common.py TaskBundle; used by CI, the recipe smoke
# tests and `python -m experiments.run_evolution --task demo_counter`.
"""demo_counter: solve items q0..q4 by naming them in the SOLVED list.

Equational-task-shaped on purpose: the grader emits per-item structured
feedback (so C1/C2/C3 all exercise) and the bundled fake transport improves
the program every other call, giving visible fitness curves offline."""

from __future__ import annotations

from pathlib import Path

from evoharness.evocore import (
    EvalReport,
    LLMResponse,
    LLMStopReason,
    LLMToolCall,
)
from evoharness.evoplus import ItemResult, StructuredFeedback

from recipes.common import TaskBundle

ITEMS = ["q0", "q1", "q2", "q3", "q4"]

INITIAL = """# EDIT-REGION-BEGIN
SOLVED = ["q0"]
# EDIT-REGION-END
print(SOLVED)
"""

TASK_SYS_MSG = "Extend SOLVED to cover all items q0..q4."


class DemoCounterGrader:
    """Item qN passes iff the literal 'qN' appears in the code."""

    def grade(self, cand, workdir: Path) -> EvalReport:
        items = [
            ItemResult(
                item_id=q,
                passed=q in cand.code,
                predicted="yes" if q in cand.code else "?",
                expected="yes",
                error_category="" if q in cand.code else f"missing-{q}",
            )
            for q in ITEMS
        ]
        solved = sum(i.passed for i in items)
        return EvalReport(
            fitness=solved / len(ITEMS),
            passed=True,
            visible_metrics={"solved": solved},
            structured_feedback=StructuredFeedback(items=items).to_json(),
            eval_cost_usd=0.001,
        )


def _fake_transport_factory():
    state = {"calls": 0, "solved": 1}

    def transport(messages, model, **kw):
        state["calls"] += 1
        tools = kw.get("tools", ())
        tool_result_turn = messages[-1].role == "tool"
        if (
            (tools and not tool_result_turn)
            or (not tools and state["calls"] % 2 == 0)
        ) and state["solved"] < len(ITEMS):
            state["solved"] += 1
        names = ", ".join(f'"{q}"' for q in ITEMS[: state["solved"]])
        code = (
            "# EDIT-REGION-BEGIN\n"
            f"SOLVED = [{names}]\n"
            "# EDIT-REGION-END\n"
            "print(SOLVED)\n"
        )
        if tools and not tool_result_turn:
            return LLMResponse(
                text="",
                model=model,
                cost=0.001,
                stop_reason=LLMStopReason.TOOL_CALLS,
                tool_calls=(
                    LLMToolCall(
                        call_id=f"demo-write-{state['calls']}",
                        name="workspace_write",
                        arguments={"path": "main.py", "content": code},
                    ),
                ),
            )
        return LLMResponse(
            text=(
                "TITLE: extend\n"
                "SUMMARY: solve more items\n"
                f"```python\n{code}```"
            ),
            model=model,
            cost=0.001,
        )

    return transport


def make_task() -> TaskBundle:
    return TaskBundle(
        grader=DemoCounterGrader(),
        initial_code=INITIAL,
        task_sys_msg=TASK_SYS_MSG,
        transport=_fake_transport_factory(),
    )
