"""Bounded multi-perspective deliberation for olympiad proofs.

The strategy is intentionally independent of any model provider or benchmark
harness.  A caller supplies one ``ask`` function; this module owns the
algorithmic control flow and returns the final proof plus an auditable trace.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass


Ask = Callable[[str, str], str]


@dataclass(frozen=True)
class ProofCall:
    """One labeled model call made by the strategy."""

    stage: str
    prompt: str
    response: str


@dataclass(frozen=True)
class ProofAudit:
    """Normalized result of the final adversarial audit."""

    verdict: str
    issues: tuple[str, ...]
    raw: str

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"


@dataclass(frozen=True)
class ProofRun:
    """Final proof and the complete bounded deliberation trace."""

    final_solution: str
    audit: ProofAudit
    calls: tuple[ProofCall, ...]


_BASE_RULES = """
Write a complete olympiad-standard proof. Every claim needed by a later step
must be justified. For classification problems, prove necessity, list every
candidate, and verify sufficiency. Treat monotonicity, equality cases,
divisibility, limiting arguments, and degenerate cases explicitly whenever
they occur. A correct answer without a proof earns no credit. Do not use a
computer search or cite an unnamed theorem as a substitute for proof.
""".strip()


class ProofStrategy:
    """Generate, synthesize, and run a bounded actor--critic proof loop.

    The normal path uses seven calls.  A failed audit triggers at most two
    repair rounds, each followed by fresh verification.  A final constrained
    rewrite is used only when the second repair still fails, keeping the hard
    maximum at thirteen calls.
    """

    def __init__(self, ask: Ask):
        if not callable(ask):
            raise TypeError("ask must be callable")
        self._ask = ask
        self._calls: list[ProofCall] = []

    def solve(self, problem: str) -> ProofRun:
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError("problem must be non-empty")
        self._calls = []

        candidate_a = self._call(
            "candidate_a",
            f"""You are the primary IMO solver.

{_BASE_RULES}

Privately build a proof-obligation ledger, but do not print it. Then write one
self-contained solution of at most 1200 words. Prefer a short chain of proved
implications over plausibility, experimentation, or exploratory commentary.

PROBLEM
{problem}
""",
        )
        candidate_b = self._call(
            "candidate_b",
            f"""You are an independent IMO solver. Solve the problem from
scratch without seeing another solver's work.

{_BASE_RULES}

Actively look for the failure modes common to this problem type and, when
possible, use a different central lemma or representation from the most
obvious approach. Return a self-contained proof of at most 1200 words, not
advice, scratch work, or a narrative about how to find one.

PROBLEM
{problem}
""",
        )
        candidate_c = self._call(
            "candidate_c",
            f"""You are a third independent olympiad mathematician. Build a
solution from first principles, emphasizing structural lemmas, extremal or
invariant arguments, and exact equality or boundary cases.

{_BASE_RULES}

Do not merely sketch an approach. Return a self-contained proof of at most
1200 words. Before finalizing, privately try to construct a counterexample to
every major claim and repair claims that fail. Do not print that exploration.

PROBLEM
{problem}
""",
        )
        review = self._call(
            "cross_review",
            f"""You are a skeptical IMO proof editor. Compare three proposed
solutions against the problem. Do not reward agreement between them; verify
the mathematics yourself.

For each candidate, list its proof obligations and mark each obligation as
proved, repairable, or fatal. Check every quantifier, case split, equality
condition, algebraic transformation, and uniqueness claim. Then give a
concrete synthesis plan containing only steps you believe can be made fully
rigorous. If all candidates are wrong, derive the missing key lemma yourself.
Be concise: report only defects that can change correctness and the exact
steps of the recommended proof.

PROBLEM
{problem}

CANDIDATE A
{candidate_a}

CANDIDATE B
{candidate_b}

CANDIDATE C
{candidate_c}
""",
        )
        synthesis = self._call(
            "synthesis",
            f"""You are the final IMO contestant. Produce the best correct
solution using the independent drafts and the editor's mathematical review.

{_BASE_RULES}

Resolve every issue raised by the editor. Do not mention candidates, editors,
audits, prompts, or uncertainty. Output only a polished self-contained proof.
If the drafts share an unsupported claim, replace it with a valid derivation
rather than repeating it. The proof must be at most 1400 words and must reach
an explicit conclusion before the output limit. Do not include scratch work,
a proof-obligation ledger, a summary, or alternative attempts.

PROBLEM
{problem}

DRAFT A
{candidate_a}

DRAFT B
{candidate_b}

DRAFT C
{candidate_c}

EDITOR REVIEW
{review}
""",
        )
        final_solution = synthesis
        audit = self._audit(problem, final_solution, "initial")

        for round_number in (1, 2):
            if audit.passed:
                break
            final_solution = self._repair(
                problem,
                final_solution,
                audit,
                round_number,
            )
            if round_number == 1:
                audit = self._audit(problem, final_solution, "repair_1")
            else:
                audit = self._final_audit(problem, final_solution)

        if not audit.passed:
            final_solution = self._call(
                "final_rewrite",
                f"""Produce a final replacement proof under a strict space
budget. The previous proof failed independent mathematical verification.
Use the reported issues only as leads and verify the mathematics yourself.
If its central approach is unsalvageable, replace it with a different proof.

{_BASE_RULES}

Output only one self-contained proof of at most 1200 words. No scratch work,
summaries, reviewer discussion, or unfinished alternatives. Every lemma used
must be proved, and the output must reach an explicit conclusion.

PROBLEM
{problem}

PREVIOUS PROOF
{final_solution}

FAILED AUDIT
{audit.raw}
""",
            )

        return ProofRun(
            final_solution=final_solution,
            audit=audit,
            calls=tuple(self._calls),
        )

    def _audit(self, problem: str, solution: str, label: str) -> ProofAudit:
        logic_raw = self._call(
            f"{label}_logic_audit",
            f"""Audit the proposed proof as an adversarial formal-logic
specialist. Try to construct a counterexample to every major inference. Check
implication directions, quantifiers, hidden assumptions, circular reasoning,
algebra, and whether every invoked lemma is actually proved and applicable.

Return only JSON in this schema:
{{"verdict":"pass" or "repair","issues":["specific fatal issue", ...]}}
Use "repair" for any incorrect or unjustified major step. Keep at most five
issues, each precise enough for a solver to act on. Do not object to concise
but valid standard algebra.

PROBLEM
{problem}

PROPOSED PROOF
{solution}
""",
        )
        completeness_raw = self._call(
            f"{label}_completeness_audit",
            f"""Independently audit the proposed proof as a strict IMO grader.
Focus only on mathematical completeness: missing cases, degenerate inputs,
boundary and equality conditions, existence, uniqueness, sufficiency, output
truncation, and claims that are plausible but unproved. Verify the argument
yourself rather than trusting its stated conclusion.

Return only JSON in this schema:
{{"verdict":"pass" or "repair","issues":["specific fatal issue", ...]}}
Keep at most five issues. Return "pass" only if the proof is complete enough
to receive full credit.

PROBLEM
{problem}

PROPOSED PROOF
{solution}
""",
        )
        return _combine_audits(
            _parse_audit(logic_raw),
            _parse_audit(completeness_raw),
        )

    def _final_audit(self, problem: str, solution: str) -> ProofAudit:
        raw = self._call(
            "repair_2_final_audit",
            f"""Act as the final adversarial IMO certifier. Independently
check both logical validity and completeness of the proposed proof. Try to
falsify its central lemma, check all cases and boundary conditions, and detect
truncation or an unfinished conclusion.

Return only JSON in this schema:
{{"verdict":"pass" or "repair","issues":["specific fatal issue", ...]}}
Return "pass" only if the proof merits full credit. Otherwise list at most
five precise issues that a final rewrite must resolve.

PROBLEM
{problem}

PROPOSED PROOF
{solution}
""",
        )
        return _parse_audit(raw)

    def _repair(
        self,
        problem: str,
        solution: str,
        audit: ProofAudit,
        round_number: int,
    ) -> str:
        return self._call(
            f"repair_{round_number}",
            f"""Rewrite the proposed solution into a complete and correct IMO
proof. The independent audit is evidence, not authority: verify each issue,
fix every genuine flaw, and reject any mistaken criticism. If the central
approach cannot close all proof obligations, replace it instead of extending
an invalid argument.

{_BASE_RULES}

Output only one polished self-contained proof of at most 1400 words. Do not
mention the audit or revision process. Do not print scratch work, speculative
claims, numerical experiments, or unfinished alternatives. Reach an explicit
conclusion before the output limit.

PROBLEM
{problem}

PROPOSED PROOF
{solution}

AUDIT
{audit.raw}
""",
        )

    def _call(self, stage: str, prompt: str) -> str:
        response = self._ask(stage, prompt)
        if not isinstance(response, str) or not response.strip():
            raise RuntimeError(f"model returned an empty {stage} response")
        response = response.strip()
        self._calls.append(ProofCall(stage, prompt, response))
        return response


def _parse_audit(raw: str) -> ProofAudit:
    """Parse one tiny JSON contract; malformed audits fail closed."""

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match is None:
        return ProofAudit("repair", ("audit output was not JSON",), raw)
    try:
        value = json.loads(match.group())
    except json.JSONDecodeError:
        return ProofAudit("repair", ("audit JSON was malformed",), raw)

    verdict = value.get("verdict")
    issues = value.get("issues")
    if verdict not in {"pass", "repair"} or not isinstance(issues, list):
        return ProofAudit("repair", ("audit schema was invalid",), raw)
    normalized = tuple(
        issue.strip()
        for issue in issues
        if isinstance(issue, str) and issue.strip()
    )
    if verdict == "pass" and normalized:
        verdict = "repair"
    return ProofAudit(verdict, normalized, raw)


def _combine_audits(*audits: ProofAudit) -> ProofAudit:
    issues = tuple(
        f"audit_{index}: {issue}"
        for index, audit in enumerate(audits, start=1)
        for issue in audit.issues
    )
    verdict = "pass" if all(audit.passed for audit in audits) else "repair"
    raw = json.dumps(
        {
            "verdict": verdict,
            "audits": [
                {
                    "verdict": audit.verdict,
                    "issues": list(audit.issues),
                }
                for audit in audits
            ],
        },
        ensure_ascii=False,
    )
    return ProofAudit(verdict, issues, raw)
