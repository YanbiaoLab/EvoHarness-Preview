import numpy as np
import pytest

from conftest import make_candidate
from evoharness.evocore import (
    MutationContext,
    PatchEngine,
    PromptBuilder,
    SearchConfig,
    apply_rewrite,
    editable_ranges,
    sample_operator,
    validate_edit_markers,
)
from evoharness.evocore.operators import parse_change_header

CODE = """import math

# EDIT-REGION-BEGIN
def solve(x):
    result = x * 2
    return result
# EDIT-REGION-END

def io_contract(x):
    return solve(x)
"""


def patch(search: str, replace: str) -> str:
    return (
        "TITLE: t\nSUMMARY: s\n"
        f"<<<<<<< ORIGINAL\n{search}\n=======\n{replace}\n>>>>>>> UPDATED\n"
    )


def test_editable_ranges_and_validation():
    ranges = editable_ranges(CODE)
    assert len(ranges) == 1
    start, end = ranges[0]
    assert "def solve" in CODE[start:end]
    assert validate_edit_markers(CODE) is None
    assert "missing" in validate_edit_markers("x = 1\n")
    assert "unbalanced" in validate_edit_markers(
        "# EDIT-REGION-BEGIN\n# EDIT-REGION-BEGIN\n# EDIT-REGION-END\n"
    )


def test_patch_applies_inside_region():
    out = PatchEngine().apply(CODE, patch("    result = x * 2", "    result = x * 3"))
    assert out.ok and out.n_applied == 1
    assert "x * 3" in out.new_code
    assert validate_edit_markers(out.new_code) is None


def test_patch_sequential_blocks_see_earlier_edits():
    text = patch("    result = x * 2", "    result = x * 3") + patch(
        "    result = x * 3", "    result = x * 4"
    )
    out = PatchEngine().apply(CODE, text)
    assert out.ok and out.n_applied == 2
    assert "x * 4" in out.new_code


def test_patch_indent_tolerance():
    # ORIGINAL block with wrong (missing) indentation is still matched.
    out = PatchEngine().apply(CODE, patch("result = x * 2", "result = x * 5"))
    assert out.ok
    assert "    result = x * 5" in out.new_code  # re-indented on apply


def test_patch_rejects_outside_region():
    out = PatchEngine().apply(
        CODE, patch("def io_contract(x):", "def io_contract(x):  # hacked")
    )
    assert not out.ok
    assert "outside the editable region" in out.error


def test_patch_not_found_reports_closest_match():
    out = PatchEngine().apply(
        CODE, patch("    result = x * 99", "    result = x * 100")
    )
    assert not out.ok
    assert "not found" in out.error
    assert "ORIGINAL block" in out.error


def test_patch_malformed_output():
    out = PatchEngine().apply(CODE, "TITLE: t\nno patch markers at all")
    assert not out.ok and "no patch blocks" in out.error
    out = PatchEngine().apply(CODE, patch("   ", "x"))
    assert not out.ok and "empty ORIGINAL" in out.error


def test_patch_cannot_remove_markers():
    out = PatchEngine().apply(
        CODE,
        patch(
            "def solve(x):\n    result = x * 2\n    return result",
            "# EDIT-REGION-END\ndef solve(x):\n    return x",
        ),
    )
    # marker lines inside patch blocks are stripped -> markers stay balanced
    assert out.ok
    assert validate_edit_markers(out.new_code) is None


def test_apply_rewrite_requires_fence_and_markers():
    good = "TITLE: t\nSUMMARY: s\n```python\n" + CODE + "```"
    out = apply_rewrite(CODE, good, "python")
    assert out.ok
    assert apply_rewrite(CODE, "no fence here", "python").error
    no_markers = "```python\nx = 1\n```"
    assert "EDIT-REGION" in apply_rewrite(CODE, no_markers, "python").error


def test_parse_change_header():
    title, summary = parse_change_header("TITLE: Faster loop\nSUMMARY: unroll\n")
    assert title == "Faster loop" and summary == "unroll"
    assert parse_change_header("nothing") == ("", "")


def test_sample_operator_excludes_recombine_without_inspirations():
    cfg = SearchConfig()
    rng = np.random.default_rng(3)
    ops = {sample_operator(cfg, has_inspirations=False, rng=rng) for _ in range(200)}
    assert "recombine" not in ops
    ops = {sample_operator(cfg, has_inspirations=True, rng=rng) for _ in range(500)}
    assert ops == {"revise", "rewrite", "recombine"}


def test_prompt_builder_contributor_injection_and_recombine_partner():
    class Injector:
        def contribute(self, ctx):
            return "# Learned Hints\nprefer smaller steps"

    parent = make_candidate("p", 1.0, code=CODE)
    peer = make_candidate("q", 2.0, code=CODE)
    builder = PromptBuilder(
        "maximize solve()", contributors=[Injector()],
        rng=np.random.default_rng(0),
    )
    ctx = MutationContext(
        parent=parent, archive_inspirations=[peer], top_k_inspirations=[],
        operator="recombine", generation=3,
    )
    system, user = builder.build(ctx)
    assert "Learned Hints" in system
    assert "maximize solve()" in system
    assert "Partner program" in user
    assert CODE.strip() in user


def test_prompt_builder_repair_carries_error_logs():
    failing = make_candidate("f", 0.0, passed=False, code=CODE)
    failing.report.fault = "ZeroDivisionError"
    failing.report.stderr_log = "Traceback ... ZeroDivisionError"
    builder = PromptBuilder("task")
    system, user = builder.build_repair(failing)
    assert "FAILED" in system
    assert "ZeroDivisionError" in user


def _multifile_candidate(tmp_path, passed=True):
    from evoharness.evocore.population import Candidate, EvalReport
    from evoharness.evocore.workspace import GitWorkspace

    (tmp_path / "main.py").write_text("def solve(x):\n    return arch.f(x)\n")
    (tmp_path / "arch.py").write_text("def f(x):\n    return x\n")
    ws = GitWorkspace.from_directory(tmp_path, main_file="main.py")
    return Candidate(
        id="mf",
        code=ws.serialize(),
        generation=1,
        parent_id=None,
        island_idx=0,
        operator="rewrite",
        workspace_kind="git",
        report=EvalReport(fitness=0.0, passed=passed),
    )


@pytest.mark.parametrize("operator", ["rewrite", "recombine"])
def test_multifile_prompts_do_not_ask_for_edit_region_markers(
    tmp_path, operator
):
    """Two contradictory formats in one prompt is a rejected proposal.

    rewrite, recombine and repair all ended in _REWRITE_RULES, which asks for
    ONE fenced block with EDIT-REGION markers preserved, while
    _MULTIFILE_FORMAT in the same prompt asks for one block PER FILE. The
    modmul genome has no markers in any file, so complying with the first
    rule was impossible and complying with the second got the answer thrown
    away.
    """
    parent = _multifile_candidate(tmp_path)
    ctx = MutationContext(
        parent=parent,
        archive_inspirations=[make_candidate("p", 0.5)],
        top_k_inspirations=[],
        operator=operator,
        generation=1,
    )
    system, user = PromptBuilder("task").build(ctx)
    assert "EDIT-REGION" not in system
    assert "### FILE:" in user


def test_multifile_repair_prompt_asks_for_file_blocks(tmp_path):
    """repair got neither half of the fix: no file-block format at all."""
    failing = _multifile_candidate(tmp_path, passed=False)
    failing.report.fault = "ZeroDivisionError"
    system, user = PromptBuilder("task").build_repair(failing)
    assert "EDIT-REGION" not in system
    assert "### FILE:" in user
    assert "FAILED" in system
