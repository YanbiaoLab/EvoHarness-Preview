# One equational-implication problem, judged by Lean

You are given two equations over a single binary operation `◇`. Decide the
implication and submit a Lean certificate that the official Stage-2 judge
accepts.

## The problem

    id     goldfalse_0001
    from   x = ((y ◇ z) ◇ z) ◇ ((y ◇ x) ◇ y)
    to     x ◇ (y ◇ y) = (z ◇ (x ◇ y)) ◇ z

The expected answer is **a COUNTEREXAMPLE: a finite magma satisfying the first equation but not the second**.

## What to produce

Write the whole certificate to `submission.lean` in your workspace. That file
is the only thing submitted; nothing else in the workspace is read.

The judge compiles it against the Stage-2 preamble and checks which axioms the
result depends on. Only these three are permitted:

    propext   Quot.sound   Classical.choice

Anything else — including `sorry` — is a rejection, whatever the file looks
like.

## What you get back

The judge's own message, verbatim, in the evaluation notes. When it rejects,
that message says what Lean objected to. Read it and fix that; do not guess at
a different approach until you know why the last one failed.

## Two things that waste attempts

The operation is spelled `◇`, not `*`. A submission written with the wrong
symbol fails deep inside the engine with an error that does not mention the
symbol at all.

A counterexample is checked by `decide`, so the carrier has to be small enough
to decide. Order 4 and 5 are fine; a large carrier times out and is scored a
rejection rather than a slow success.
