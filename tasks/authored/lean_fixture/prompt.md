# One Lean goal, checked by the Lean compiler

Finish the proof of `fixture_main` in `fixture.lean`. The file is checked by
Lean itself; nothing here is graded by a model.

## The goal

    theorem fixture_main (a b c : Nat) :
        (a + b) * c = a * c + b * c
        ∧ (a + b) + c = a + (b + c)
        ∧ a * 0 = 0

Three independent conjuncts, each of which follows from a single lemma in core
`Nat`. There is no import: everything you need is already in scope.

## What to produce

Edit `fixture.lean` in your workspace, between the `-- EDIT-REGION-BEGIN` and
`-- EDIT-REGION-END` markers. Keep both markers: a proposal without them is
rejected before it is ever compiled. You may add helper lemmas inside the
region; they must appear before `fixture_main`, as Lean requires.

**Do not touch the locked footer.** It restates the goal and applies your
theorem to it, so a weakened `fixture_main` does not compile — the footer stops
typechecking. Deleting the footer is scored as a malformed submission.

## How it is scored

    file does not compile                            0.0
    compiles, still depends on sorryAx               partial credit
    compiles, depends only on the allowed axioms     1.0

The allowed axioms are exactly `propext`, `Quot.sound` and `Classical.choice`.
Anything else — `sorryAx`, or what `native_decide` pulls in — is not a proof
here, whatever the file looks like.

Partial credit rises as fewer of your declarations are flagged by Lean as using
`sorry`. It orders the search and nothing more: **only the axiom report can
make this task solved.**

## What you get back

Lean's own diagnostics, verbatim, in the evaluation notes. Read them and fix
what they name. Do not switch to a different approach before you know why the
last one failed.

## Two things that waste attempts

`sorry` left anywhere `fixture_main` depends on will show up as `sorryAx` in
the axiom report, however clean the file reads.

A tactic that searches too widely can run past the compile time limit, which is
scored as a failure rather than a slow success.
