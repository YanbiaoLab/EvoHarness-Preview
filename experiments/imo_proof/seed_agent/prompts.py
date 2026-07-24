"""Prompt templates used by the seed IMO solver."""

SOLVE_PROMPT = """Solve the following olympiad problem. Produce one rigorous,
self-contained proof. Do not cite a reference solution and do not include
scratch work, review commentary, unverifiable claims, or <think> tags.

Write as tersely as rigor allows. Do not restate the problem. State each
key claim once, then justify it in the fewest steps that remain complete;
combine routine algebra and omit elementary justifications a grader assumes.
Prefer symbols and short lemmas over prose. Output only the final proof and
keep it under 1200 words.

PROBLEM
{problem}
"""

REVIEW_PROMPT = """Act as a strict IMO reviewer. Return exactly one of these
two schemas and no other text:

<verdict>PASS</verdict>

or

<verdict>REVISE</verdict>
<feedback>Identify the single most important mathematical flaw or missing
justification.</feedback>

PROBLEM
{problem}

PROOF
{proof}
"""

REVISE_PROMPT = """Return one corrected, self-contained proof. Fix the review
issue only if it is valid, independently verify every step, and output no
review commentary, scratch work, or <think> tags.

Write as tersely as rigor allows. Do not restate the problem. State each key
claim once, then justify it in the fewest complete steps; combine routine
algebra and omit elementary justifications a grader assumes. Prefer symbols
and short lemmas over prose. Output only the final proof and keep it under
1200 words.

PROBLEM
{problem}

CURRENT PROOF
{proof}

REVIEW
{review}
"""
