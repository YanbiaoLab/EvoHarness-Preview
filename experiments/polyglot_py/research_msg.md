<!-- polyglot_py research_msg v1 (2026-08-04). Frozen domain brief, shared
     identically by every arm (e0 / e3r / e6p) so the brief is not itself an
     ablation variable. Terrain and published facts only. -->

# Domain brief: agentic coding on Exercism-style exercises

## What the corpus is

Aider's polyglot benchmark, Python half: 34 practice exercises taken from
Exercism, each with a written specification, a stub file, and a test suite
written independently of any particular solution. The tests are the
specification made executable, and they are given to the harness in full.
Exercises range from a dozen lines (`bob`, `bowling`) to parsers and small
interpreters (`sgf-parsing`, `forth`, `react`, `zebra-puzzle`).

The benchmark's own framing is that a strong model with a naive harness leaves
a large amount on the table: the published leaderboard's gap between one-shot
prompting and an edit-test-repair loop with the same model is the whole reason
the benchmark exists.

## The failure modes this corpus actually produces

Measured on Exercism-style Python suites generally, not on this run:

1. **Contract mismatch.** The tests import specific names, call specific
   signatures, and often expect a specific exception type with a specific
   message. Solutions that implement the right algorithm under the wrong name
   fail exactly as hard as ones that implement nothing.
2. **Partial edge coverage.** The specification's worked examples are a
   subset of what the tests check; the remaining cases are stated in prose or
   only in the tests.
3. **Silent truncation.** A long solution cut off by a token limit yields a
   syntax error, which is indistinguishable from a bad answer unless the
   failure category is read.

## What the harness controls

The model, its temperature, and the corpus are fixed. What varies between
candidates is: what goes into the prompt and in what order, whether the tests
are read before writing, how the reply is turned into a file, how many
attempts are spent and on what basis, what is carried between attempts, and
how the per-exercise budget is allocated. That is the entire search space.

## Variance

One evaluation is a single pass over a fixed exercise set with a
non-deterministic model. The reported standard error is Bernoulli over the
exercise set, which understates the true variance because it ignores
run-to-run sampling of the model. Two candidates whose intervals overlap are
not ordered by a single evaluation.

## Deliberately not in this brief

Whether reading the tests first helps, how many repair rounds are worth their
cost, what a good prompt says, and how to handle multi-file exercises. Those
are what the search is for.
