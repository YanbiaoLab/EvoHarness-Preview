import pytest

from experiments.imo_proof.strategy import ProofStrategy


def test_pass_path_uses_three_candidates_review_synthesis_and_two_audits():
    responses = {
        "candidate_a": "proof A",
        "candidate_b": "proof B",
        "candidate_c": "proof C",
        "cross_review": "C is strongest; repair its last lemma.",
        "synthesis": "final rigorous proof",
        "initial_logic_audit": '{"verdict":"pass","issues":[]}',
        "initial_completeness_audit": '{"verdict":"pass","issues":[]}',
    }
    seen = []

    def ask(stage, prompt):
        seen.append((stage, prompt))
        return responses[stage]

    run = ProofStrategy(ask).solve("Prove P.")

    assert run.final_solution == "final rigorous proof"
    assert run.audit.passed
    assert [call.stage for call in run.calls] == list(responses)
    assert "proof A" in seen[3][1]
    assert "proof B" in seen[3][1]
    assert "proof C" in seen[3][1]
    assert "C is strongest" in seen[4][1]


def test_failed_audit_triggers_repair_and_fresh_double_audit():
    outputs = iter(
        [
            "proof A",
            "proof B",
            "proof C",
            "review",
            "draft final",
            '{"verdict":"repair","issues":["missing edge case"]}',
            '{"verdict":"pass","issues":[]}',
            "repaired final proof",
            '{"verdict":"pass","issues":[]}',
            '{"verdict":"pass","issues":[]}',
        ]
    )

    run = ProofStrategy(lambda _stage, _prompt: next(outputs)).solve(
        "Prove P."
    )

    assert run.final_solution == "repaired final proof"
    assert run.audit.passed
    assert len(run.calls) == 10
    assert run.calls[7].stage == "repair_1"
    assert "missing edge case" in run.calls[7].prompt
    assert run.calls[-1].stage == "repair_1_completeness_audit"


@pytest.mark.parametrize(
    "audit",
    [
        "not json",
        "{broken}",
        '{"verdict":"pass","issues":["still broken"]}',
        '{"verdict":"unknown","issues":[]}',
    ],
)
def test_malformed_or_inconsistent_audit_fails_closed(audit):
    outputs = iter(
        [
            "A",
            "B",
            "C",
            "review",
            "draft",
            audit,
            '{"verdict":"pass","issues":[]}',
            "repair 1",
            '{"verdict":"pass","issues":[]}',
            '{"verdict":"pass","issues":[]}',
        ]
    )

    run = ProofStrategy(lambda _stage, _prompt: next(outputs)).solve(
        "Prove P."
    )

    assert run.final_solution == "repair 1"
    assert run.audit.passed
    assert len(run.calls) == 10


def test_two_failed_repairs_use_final_audit_and_bounded_rewrite():
    outputs = iter(
        [
            "A",
            "B",
            "C",
            "review",
            "draft",
            '{"verdict":"repair","issues":["gap 1"]}',
            '{"verdict":"pass","issues":[]}',
            "repair 1",
            '{"verdict":"repair","issues":["gap 2"]}',
            '{"verdict":"pass","issues":[]}',
            "repair 2",
            '{"verdict":"repair","issues":["gap 3"]}',
            "final rewrite",
        ]
    )

    run = ProofStrategy(lambda _stage, _prompt: next(outputs)).solve(
        "Prove P."
    )

    assert run.final_solution == "final rewrite"
    assert not run.audit.passed
    assert run.audit.issues == ("gap 3",)
    assert len(run.calls) == 13
    assert run.calls[-2].stage == "repair_2_final_audit"
    assert run.calls[-1].stage == "final_rewrite"


def test_rejects_empty_problem_and_empty_model_response():
    with pytest.raises(ValueError, match="problem"):
        ProofStrategy(lambda _stage, _prompt: "answer").solve(" ")

    with pytest.raises(RuntimeError, match="candidate_a"):
        ProofStrategy(lambda _stage, _prompt: "").solve("Prove P.")
