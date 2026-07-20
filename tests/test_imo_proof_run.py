import csv

import pytest

from experiments.imo_proof.run import (
    _clear_invalid_grader_predictions,
    _validate_artifacts,
)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("Problem ID", "prediction"),
        )
        writer.writeheader()
        writer.writerows(rows)


def make_artifacts(tmp_path, *, calls=(7, 8), labels=None):
    ids = ["PB-001", "PB-002"]
    write_csv(
        tmp_path / "predictions.csv",
        [
            {"Problem ID": problem_id, "prediction": f"proof {index}"}
            for index, problem_id in enumerate(ids)
        ],
    )
    labels = labels or ("correct", "partial")
    write_csv(
        tmp_path / "gradings" / "predictions.csv",
        [
            {"Problem ID": problem_id, "prediction": label}
            for problem_id, label in zip(ids, labels)
        ],
    )
    for problem_id, count in zip(ids, calls):
        solver = tmp_path / "agent_evals" / f"chat_history_{problem_id}.md"
        solver.parent.mkdir(parents=True, exist_ok=True)
        solver.write_text("\n".join(["Input: prompt", "Output: proof"] * count))
        grader = (
            tmp_path
            / "gradings"
            / "agent_evals"
            / f"chat_history_{problem_id}.md"
        )
        grader.parent.mkdir(parents=True, exist_ok=True)
        grader.write_text("Input: grade\nOutput: correct\n")


def test_validate_artifacts_accepts_complete_bounded_run(tmp_path):
    make_artifacts(tmp_path)

    calls = _validate_artifacts(
        tmp_path,
        expected_samples=2,
        min_calls=7,
        max_calls=8,
    )

    assert calls["total"] == 15
    assert calls["per_problem"] == {"PB-001": 7, "PB-002": 8}


def test_validate_artifacts_rejects_incomplete_or_invalid_grading(tmp_path):
    make_artifacts(tmp_path, labels=("correct", "None"))

    with pytest.raises(RuntimeError, match="invalid grader label"):
        _validate_artifacts(
            tmp_path,
            expected_samples=2,
            min_calls=7,
            max_calls=8,
        )


def test_validate_artifacts_rejects_call_budget_drift(tmp_path):
    make_artifacts(tmp_path, calls=(7, 9))

    with pytest.raises(RuntimeError, match="declared bounds"):
        _validate_artifacts(
            tmp_path,
            expected_samples=2,
            min_calls=7,
            max_calls=8,
        )


def test_clear_invalid_grader_predictions_only_clears_bad_labels(tmp_path):
    path = tmp_path / "predictions.csv"
    write_csv(
        path,
        [
            {"Problem ID": "PB-001", "prediction": "correct"},
            {"Problem ID": "PB-002", "prediction": "None"},
            {"Problem ID": "PB-003", "prediction": ""},
        ],
    )

    assert _clear_invalid_grader_predictions(path) == 2

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["prediction"] for row in rows] == ["correct", "", ""]
