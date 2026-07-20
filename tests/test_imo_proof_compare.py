import json

import pytest

from experiments.imo_proof.compare import compare_reports, compare_runs


def write_report(path, *, total, score):
    path.write_text(
        json.dumps({"total": total, "points_percentage": score})
    )


def write_manifest(path, *, selection="first 10 rows", model="m"):
    path.write_text(
        json.dumps(
            {
                "dataset": {
                    "sha256": "dataset-hash",
                    "selection": selection,
                    "num_samples": 10,
                },
                "model": {
                    "name": model,
                    "enable_thinking": False,
                    "temperature": 0.0,
                    "max_tokens": 4096,
                },
                "grader": {
                    "model": model,
                    "source_sha256": "grader-hash",
                },
                "algorithm": {"name": "algo"},
            }
        )
    )


def write_gradings(directory, labels):
    gradings = directory / "gradings"
    gradings.mkdir()
    rows = ["Problem ID,prediction"]
    rows.extend(
        f"PB-{index:03d},{label}"
        for index, label in enumerate(labels, start=1)
    )
    (gradings / "predictions.csv").write_text("\n".join(rows) + "\n")
    solver_traces = directory / "agent_evals"
    solver_traces.mkdir()
    for index in range(1, len(labels) + 1):
        (solver_traces / f"chat_history_PB-{index:03d}.md").write_text(
            "Input: solve\nOutput: proof\n"
        )


def test_compare_reports_requires_strict_score_improvement(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    write_report(baseline, total=10, score=23 / 70)
    write_report(candidate, total=10, score=35 / 70)

    result = compare_reports(baseline, candidate)

    assert result["candidate_wins"] is True
    assert result["absolute_improvement"] == pytest.approx(12 / 70)


def test_compare_reports_rejects_different_sample_counts(tmp_path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    write_report(baseline, total=10, score=0.3)
    write_report(candidate, total=60, score=0.4)

    with pytest.raises(ValueError, match="same number"):
        compare_reports(baseline, candidate)


def test_compare_runs_verifies_protocol_before_scores(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    for directory, score in ((baseline, 0.3), (candidate, 0.5)):
        write_report(directory / "report.json", total=10, score=score)
        write_manifest(directory / "manifest.json")
    write_gradings(baseline, ["incorrect"] * 10)
    write_gradings(candidate, ["correct"] * 10)

    result = compare_runs(baseline, candidate)

    assert result["candidate_wins"] is True
    assert result["protocol_verified"] is True
    assert result["paired_analysis"]["candidate_problem_wins"] == 10
    assert result["paired_analysis"]["paired_bootstrap_95_ci"] == [1.0, 1.0]
    assert result["model_calls"]["candidate_to_baseline_ratio"] == 1.0


def test_compare_runs_rejects_model_or_dataset_drift(tmp_path):
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    for directory in (baseline, candidate):
        write_report(directory / "report.json", total=10, score=0.5)
    write_manifest(baseline / "manifest.json")
    write_manifest(
        candidate / "manifest.json",
        selection="last 10 rows",
        model="other-model",
    )

    with pytest.raises(
        ValueError,
        match="dataset_selection, grader_model, model",
    ):
        compare_runs(baseline, candidate)
