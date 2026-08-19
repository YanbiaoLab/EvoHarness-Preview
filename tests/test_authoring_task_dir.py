"""Loading an authored task directory.

The scoring rule itself belongs to whoever owns the domain, so nothing here
judges it. What these pin is the mechanical contract: identity travels with
contents, declared paths cannot escape, reference material reaches the agent,
and the loaded grader actually scores through the path the run uses.
"""

import json

import pytest

from evoharness.authoring import (
    CommandCheck,
    TaskDirError,
    load_task_from_dir,
)
from evoharness.core.population import Candidate

GRADE = '''
from pathlib import Path


def grade(candidate_dir, ctx):
    namespace = {}
    exec((Path(candidate_dir) / "main.py").read_text(), namespace)
    solve = namespace["solve"]
    hits = 0
    for n in range(1, 21):
        try:
            hits += solve(n) == n * 2
        except Exception:
            pass
    return {"fitness": hits / 20, "n_units": 20, "trustworthy_units": 20}
'''

SEED = "def solve(n):\n    return n\n"


def build_task_dir(root, *, declaration_overrides=None, grade_source=GRADE):
    root.mkdir(parents=True, exist_ok=True)
    (root / "grade.py").write_text(grade_source, encoding="utf-8")
    (root / "seed").mkdir(exist_ok=True)
    (root / "seed" / "main.py").write_text(SEED, encoding="utf-8")
    (root / "prompt.md").write_text(
        "Make solve(n) return twice its input.", encoding="utf-8"
    )
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs" / "brief.md").write_text(
        "Doubling is multiplication by two.", encoding="utf-8"
    )

    declaration = {
        "task_id": "authored_double",
        "version": "v1",
        "prompt": "prompt.md",
        "knowledge": ["docs"],
        "criterion": {"name": "accuracy", "direction": "maximize"},
        "measurement": {
            "name": "double-probe",
            "version": "v1",
            "universe_hash": "probe-1-to-20",
            "planned_units": 20,
        },
        "seed": {"dir": "seed", "main_file": "main.py"},
        "grade": {"module": "grade.py", "entry": "grade"},
    }
    declaration.update(declaration_overrides or {})
    (root / "task.json").write_text(
        json.dumps(declaration, indent=2), encoding="utf-8"
    )
    return root


def test_a_directory_loads_into_a_task(tmp_path):
    task = load_task_from_dir(build_task_dir(tmp_path / "task"))

    assert task.spec.task_id == "authored_double"
    assert task.initial_workspace.main_text() == SEED
    assert json.loads(task.spec.metadata_json)["authored"] is True


def test_the_loaded_grader_scores_through_the_production_path(tmp_path):
    task = load_task_from_dir(build_task_dir(tmp_path / "task"))
    seed = Candidate(
        id="seed0",
        code=task.initial_workspace.main_text(),
        generation=0,
        parent_id=None,
        island_idx=0,
        operator="seed",
    )
    seed.workspace = task.initial_workspace

    report = task.grader.grade(seed, tmp_path / "work")

    # The loader wraps the authored function before handing it to
    # WorkspaceGradeFnGrader, and nothing else exercises that wrapper. A
    # wrapper that normalizes the grade itself makes every candidate fault
    # while every other test stays green.
    assert report.passed, report.stderr_log
    assert report.fault is None
    assert 0.0 <= report.fitness <= 1.0


def test_reference_material_reaches_the_task(tmp_path):
    task = load_task_from_dir(build_task_dir(tmp_path / "task"))

    # The evolving agent reads this as its research brief; a task whose
    # knowledge silently failed to load would look identical to one that never
    # declared any.
    assert "multiplication by two" in task.research_brief


def test_missing_reference_material_is_refused_not_ignored(tmp_path):
    root = build_task_dir(
        tmp_path / "task", declaration_overrides={"knowledge": ["docs/gone.md"]}
    )
    with pytest.raises(TaskDirError, match="no such file"):
        load_task_from_dir(root)


def test_declared_preflight_checks_become_validators(tmp_path):
    root = build_task_dir(
        tmp_path / "task",
        declaration_overrides={
            "preflight": [
                {"name": "imports", "argv": ["python", "-c", "import main"]}
            ]
        },
    )
    task = load_task_from_dir(root)

    assert [v.name for v in task.preflight_validators] == ["imports"]
    assert task.preflight_validators[0].argv == ("python", "-c", "import main")


def test_a_preflight_check_fails_a_broken_candidate(tmp_path):
    from evoharness.core.preflight import PreflightContext

    check = CommandCheck(name="imports", argv=("python", "-c", "import main"))
    workdir = tmp_path / "candidate"
    workdir.mkdir()
    (workdir / "main.py").write_text("def solve(\n", encoding="utf-8")

    parent = Candidate(
        id="p", code="", generation=0, parent_id=None,
        island_idx=0, operator="seed",
    )
    result = check.validate(
        PreflightContext(parent=parent, operator="revise", workdir=workdir)
    )

    assert not result.ok
    issue = result.issues[0]
    assert issue.repairable is True
    # The interpreter's own message is what makes the failure actionable;
    # dropping it would leave the model to guess what "exit 1" meant.
    assert "SyntaxError" in issue.stderr


def test_a_missing_command_is_not_the_candidates_fault(tmp_path):
    from evoharness.core.preflight import PreflightContext

    check = CommandCheck(name="ghost", argv=("definitely-not-a-command",))
    parent = Candidate(
        id="p", code="", generation=0, parent_id=None,
        island_idx=0, operator="seed",
    )

    result = check.validate(
        PreflightContext(parent=parent, operator="revise", workdir=tmp_path)
    )

    # A broken task declaration must not be handed to the model as something
    # to repair: it would burn every repair round on an unfixable problem.
    assert not result.ok
    assert result.issues[0].repairable is False


def test_the_same_task_at_two_paths_is_the_same_task(tmp_path):
    here = load_task_from_dir(build_task_dir(tmp_path / "here"))
    moved = load_task_from_dir(build_task_dir(tmp_path / "elsewhere"))

    # Identity has to travel. Recording the absolute directory instead would
    # put the checkout location into TaskSpec.hash, and two identical tasks
    # would compare as different experiments.
    assert here.spec.hash == moved.spec.hash


def test_editing_the_grade_function_is_a_different_task(tmp_path):
    before = load_task_from_dir(build_task_dir(tmp_path / "before"))
    after = load_task_from_dir(
        build_task_dir(
            tmp_path / "after",
            grade_source=GRADE.replace("range(1, 21)", "range(1, 31)"),
        )
    )

    assert before.spec.hash != after.spec.hash


def test_editing_reference_material_is_a_different_task(tmp_path):
    before_root = build_task_dir(tmp_path / "before")
    after_root = build_task_dir(tmp_path / "after")
    (after_root / "docs" / "brief.md").write_text("Different.", encoding="utf-8")

    # The brief is in the agent's prompt every generation, so changing it
    # changes what was searched even though no code moved.
    assert (
        load_task_from_dir(before_root).spec.hash
        != load_task_from_dir(after_root).spec.hash
    )


def test_planned_units_must_be_stated_not_defaulted(tmp_path):
    root = build_task_dir(
        tmp_path / "task",
        declaration_overrides={
            "measurement": {
                "name": "double-probe",
                "version": "v1",
                "universe_hash": "probe-1-to-20",
            }
        },
    )
    with pytest.raises(TaskDirError, match="planned_units"):
        load_task_from_dir(root)


def test_a_declared_path_cannot_escape_the_task_directory(tmp_path):
    root = build_task_dir(
        tmp_path / "task", declaration_overrides={"seed": {"dir": "../.."}}
    )
    with pytest.raises(TaskDirError, match="escapes"):
        load_task_from_dir(root)


def test_a_knowledge_folder_is_read_whole_and_labelled(tmp_path):
    root = build_task_dir(tmp_path / "task")
    (root / "docs" / "api.md").write_text("call solve(n)", encoding="utf-8")
    (root / "docs" / "nested").mkdir()
    (root / "docs" / "nested" / "note.md").write_text("deep", encoding="utf-8")

    brief = load_task_from_dir(root).research_brief

    assert "multiplication by two" in brief and "call solve(n)" in brief
    assert "deep" in brief
    # Every chunk carries its path: a folder becomes many files, and prose
    # concatenated without boundaries leaves the agent unable to tell an API
    # reference from a worked example.
    assert "# docs/api.md" in brief
    assert "# docs/nested/note.md" in brief


def test_binary_and_vcs_files_inside_a_knowledge_folder_are_skipped(tmp_path):
    root = build_task_dir(tmp_path / "task")
    (root / "docs" / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\xff\xfe")
    (root / "docs" / ".git").mkdir()
    (root / "docs" / ".git" / "HEAD").write_text("ref: x", encoding="utf-8")

    brief = load_task_from_dir(root).research_brief

    # A knowledge folder is often a checkout: it legitimately contains images
    # and git bookkeeping beside its prose, and neither belongs in a prompt.
    assert "PNG" not in brief and "ref: x" not in brief


def test_an_external_knowledge_tree_is_part_of_the_task_identity(tmp_path):
    external = tmp_path / "shared_kb"
    external.mkdir()
    (external / "facts.md").write_text("first", encoding="utf-8")
    root = build_task_dir(
        tmp_path / "task",
        declaration_overrides={"knowledge": ["../shared_kb"]},
    )

    before = load_task_from_dir(root).spec.hash
    (external / "facts.md").write_text("second", encoding="utf-8")
    after = load_task_from_dir(root).spec.hash

    # The agent reads this tree every generation, so it must not be able to
    # change while the task hash stands still. It works because TaskSpec
    # carries the knowledge TEXT, not because the external path is tracked —
    # which is why nothing hashes the directory itself.
    assert before != after


def test_oversized_knowledge_is_refused_with_the_worst_offenders_named(
    tmp_path,
):
    root = build_task_dir(
        tmp_path / "task",
        declaration_overrides={"knowledge_max_bytes": 1024},
    )
    (root / "docs" / "huge.md").write_text("x" * 5000, encoding="utf-8")

    with pytest.raises(TaskDirError, match="huge.md"):
        load_task_from_dir(root)
