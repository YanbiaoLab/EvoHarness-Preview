"""Seed proof-revision policy."""


def review_verdict(review):
    opening = "<VERDICT>"
    closing = "</VERDICT>"
    normalized = review.upper()
    if normalized.count(opening) != 1 or normalized.count(closing) != 1:
        return "REVISE"
    start = normalized.index(opening) + len(opening)
    end = normalized.index(closing, start)
    verdict = normalized[start:end].strip()
    return verdict if verdict in {"PASS", "REVISE"} else "REVISE"


def needs_revision(review):
    return review_verdict(review) != "PASS"
