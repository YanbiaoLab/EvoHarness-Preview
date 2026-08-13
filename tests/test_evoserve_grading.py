import pytest

from evoharness.core import EvalReport
from evoharness.serve import Grade, coerce_grade


def test_float_shortcut_fills_defaults():
    report = coerce_grade(0.7)
    assert report["fitness"] == 0.7
    assert report["passed"] is True
    assert report["schema_version"] == 1


def test_dict_partial_requires_fitness():
    report = coerce_grade({"fitness": 0.5, "fault": "timeout"})
    assert report["fault"] == "timeout"
    with pytest.raises(ValueError, match="fitness"):
        coerce_grade({"passed": True})


def test_unknown_key_rejected_loudly():
    with pytest.raises(ValueError, match="score"):
        coerce_grade({"fitness": 0.5, "score": 0.5})


def test_nan_rejected_at_source():
    with pytest.raises(ValueError, match="finite"):
        coerce_grade(float("nan"))


def test_bool_rejected():
    with pytest.raises(TypeError):
        coerce_grade(True)


def test_wire_round_trip_into_framework_report():
    """The fuse: serve output must parse as an core EvalReport."""
    wire = coerce_grade(
        Grade(
            fitness=0.42,
            passed=False,
            fault="3 carry errors",
            structured_feedback={"items": [], "error_histogram": {"carry": 3}},
            eval_cost_usd=0.01,
        )
    )
    report = EvalReport.from_json(wire)
    assert report.fitness == 0.42
    assert report.passed is False
    assert report.structured_feedback == wire["structured_feedback"]


def test_trustworthy_units_round_trip_and_cannot_exceed_execution():
    wire = coerce_grade(
        Grade(
            fitness=0.5,
            n_units=4,
            trustworthy_units=3,
        )
    )
    report = EvalReport.from_json(wire)
    assert (report.n_units, report.trustworthy_units) == (4, 3)
    with pytest.raises(ValueError, match="trustworthy_units"):
        coerce_grade(
            Grade(fitness=0.5, n_units=2, trustworthy_units=3)
        )


def test_grade_fields_mirror_eval_report():
    """Deliberate duplication (see docs/eval_protocol.md): Grade mirrors
    EvalReport field-for-field; this test is the drift alarm."""
    from dataclasses import fields

    grade_fields = {f.name for f in fields(Grade)}
    report_fields = {f.name for f in fields(EvalReport)}
    assert grade_fields == report_fields
