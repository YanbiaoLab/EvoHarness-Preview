# EvoHarness original research extension (plan C1: structured verifier
# feedback). Upstream only carries free-text feedback; the structured
# object, error clustering and behavior signature are this project's work.
"""C1: structured verifier feedback and the behavior signature it derives."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from evoharness.core.interfaces import MutationContext


@dataclass(frozen=True)
class BehaviorSignature:
    """Per-item pass/fail vector + error-category histogram. Item order must
    be the fixed train-set order so signatures are comparable."""

    pass_vector: tuple[bool, ...]
    error_histogram: tuple[tuple[str, int], ...]

    def hamming(self, other: "BehaviorSignature") -> int:
        a, b = self.pass_vector, other.pass_vector
        common = sum(x != y for x, y in zip(a, b))
        return common + abs(len(a) - len(b))

    def encode(self) -> str:
        bits = "".join("1" if p else "0" for p in self.pass_vector)
        hist = ",".join(f"{k}:{v}" for k, v in self.error_histogram)
        return f"{bits}|{hist}"

    @classmethod
    def decode(cls, s: str) -> "BehaviorSignature":
        bits, _, hist = s.partition("|")
        histogram = []
        if hist:
            for part in hist.split(","):
                k, _, v = part.rpartition(":")
                histogram.append((k, int(v)))
        return cls(
            pass_vector=tuple(c == "1" for c in bits),
            error_histogram=tuple(histogram),
        )


@dataclass
class ItemResult:
    item_id: str
    passed: bool
    predicted: str = ""
    expected: str = ""
    error_category: str = ""


@dataclass
class StructuredFeedback:
    items: list[ItemResult]
    summary: str = ""

    @property
    def error_histogram(self) -> dict[str, int]:
        return dict(
            Counter(
                i.error_category or "uncategorized"
                for i in self.items
                if not i.passed
            )
        )

    def to_json(self) -> dict:
        return {
            "schema_version": 1,
            "items": [
                {
                    "item_id": i.item_id,
                    "passed": i.passed,
                    "predicted": i.predicted,
                    "expected": i.expected,
                    "error_category": i.error_category,
                }
                for i in self.items
            ],
            "summary": self.summary,
        }

    @classmethod
    def from_json(cls, d: dict) -> "StructuredFeedback":
        return cls(
            items=[
                ItemResult(
                    item_id=str(i["item_id"]),
                    passed=bool(i["passed"]),
                    predicted=i.get("predicted", ""),
                    expected=i.get("expected", ""),
                    error_category=i.get("error_category", ""),
                )
                for i in d.get("items", [])
            ],
            summary=d.get("summary", ""),
        )

    def signature(self) -> BehaviorSignature:
        return BehaviorSignature(
            pass_vector=tuple(i.passed for i in self.items),
            error_histogram=tuple(sorted(self.error_histogram.items())),
        )

    def top_error_categories(self, k: int = 3) -> list[str]:
        hist = self.error_histogram
        return [c for c, _ in sorted(hist.items(), key=lambda kv: -kv[1])[:k]]

    def render(self, top_k: int = 3, examples_per_cat: int = 1) -> str:
        """Failure summary for the mutation prompt: top-k error categories
        with one representative example each (plan C1)."""
        failed = [i for i in self.items if not i.passed]
        if not failed:
            return ""
        n = len(self.items)
        lines = [f"The parent program failed {len(failed)}/{n} items."]
        if self.summary:
            lines.append(self.summary)
        for cat in self.top_error_categories(top_k):
            examples = [
                i for i in failed
                if (i.error_category or "uncategorized") == cat
            ]
            lines.append(f"- {cat} ({len(examples)} failures)")
            for ex in examples[:examples_per_cat]:
                detail = f"  e.g. item {ex.item_id}"
                if ex.expected or ex.predicted:
                    detail += f": expected {ex.expected!r}, got {ex.predicted!r}"
                lines.append(detail)
        return "\n".join(lines)


class FeedbackContributor:
    """Implements PromptContributor: injects the parent's failure summary.
    Enabled/disabled at assembly time (experiment matrix E1+)."""

    def __init__(self, top_k: int = 3, examples_per_cat: int = 1):
        self.top_k = top_k
        self.examples_per_cat = examples_per_cat

    def contribute(self, ctx: MutationContext) -> str | None:
        report = ctx.parent.report
        if report is None or not report.structured_feedback:
            return None
        feedback = StructuredFeedback.from_json(report.structured_feedback)
        body = feedback.render(self.top_k, self.examples_per_cat)
        if not body:
            return None
        return "# Parent failure analysis\n" + body
