"""Seed IMO proof-solving policy entrypoint."""

from policy import needs_revision
from prompts import REVIEW_PROMPT, REVISE_PROMPT, SOLVE_PROMPT


def _fill(template, **fields):
    """Brace-safe template fill. Math prompts contain literal { } and $, so
    str.format()/string.Template would choke; substitute placeholders textually.
    """
    text = template
    for key, value in fields.items():
        text = text.replace("{" + key + "}", value)
    return text


def solve(problem, llm):
    proof = llm.complete("solve", _fill(SOLVE_PROMPT, problem=problem))
    review = llm.complete(
        "review",
        _fill(REVIEW_PROMPT, problem=problem, proof=proof),
    )
    revise = needs_revision(review)
    if revise:
        proof = llm.complete(
            "revise",
            _fill(REVISE_PROMPT, problem=problem, proof=proof, review=review),
        )
    return {
        "proof": proof,
        "status": "completed",
        "metadata": {"review_passed": not revise},
    }
