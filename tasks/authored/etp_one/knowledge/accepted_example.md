# One certificate the judge accepted

A different problem, deliberately — this shows the shape a submission takes,
not the answer to yours.

    id     goldfalse_0002
    from   x = (((y ◇ z) ◇ (z ◇ x)) ◇ y) ◇ x
    to     x = y ◇ ((y ◇ z) ◇ (z ◇ (y ◇ x)))
    答案   false

```lean
-- stage:stage1.9_rulebook_explicit_table_refutation5
import JudgeProblem
import JudgeDecide.DecideBang
import JudgeFinOp.MemoFinOp

set_option maxRecDepth 1000000
set_option maxHeartbeats 0
open MemoFinOp

def submission : Goal := by
  let m : Magma (Fin 8) := {
    op := finOpTable "[[0,2,3,4,5,6,7,1],[6,1,5,7,3,0,4,2],[7,3,2,6,1,4,0,5],[1,6,4,3,7,2,5,0],[2,0,7,5,4,1,3,6],[3,7,0,1,6,5,2,4],[4,5,1,0,2,7,6,3],[5,4,6,2,0,3,1,7]]"
  }
  refine ⟨Fin 8, m, ?_⟩
  decideFin!
```

Worth noticing:

- The imports are the Stage-2 preamble's, not arbitrary Mathlib. `JudgeProblem`
  supplies `Goal`; `JudgeFinOp.MemoFinOp` supplies the table helper.
- A counterexample is a carrier plus an operation table plus `decide`. The
  table is a literal string, row-major, and its size is the carrier's order.
- Nothing here proves anything by hand. `decide` does the work, which is why
  the carrier has to stay small.
