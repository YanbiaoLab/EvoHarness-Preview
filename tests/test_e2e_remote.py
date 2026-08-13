"""End-to-end: real serve over real HTTP, real RemoteGrader, recipe
assembly. The only fake left is the LLM transport (the demo task is offline
by design). This is the two sides of docs/eval_protocol.md talking for real.
"""

import threading

import pytest

import recipes
from evoharness.core import EvalReport, LLMClient, PopulationConfig, SearchConfig
from evoharness.core.population import Candidate
from evoharness.core.remote import RemoteEvalConfig, RemoteGrader
from evoharness.guard import AntiHackScanner
from evoharness.evoplus import StructuredFeedback
from evoharness.evoplus.config import PlusConfig
from evoharness.serve import EvalService
from evoharness.serve.http import serve
from recipes.common import RecipeContext
from tasks import get_task
from tasks.demo_counter_remote import grade_fn


# -- composition glue (recipes-layer role, see docs/eval_protocol.md §7) ------

def make_pregate(scanner: AntiHackScanner):
    """Wrap the L0 whole-workspace static scan as a RemoteGrader pregate."""

    def pregate(ws) -> EvalReport | None:
        findings = scanner.scan_files(ws.texts())
        if findings:
            f = findings[0]
            return EvalReport(
                fitness=0.0, passed=False, stage_reached=0,
                fault=f"L0 {f.rule}@{f.path}:{f.lineno}: {f.detail}",
            )
        return None

    return pregate


@pytest.fixture()
def remote(tmp_path):
    """A live eval service + a RemoteGrader pointed at it."""
    server_dir = tmp_path / "server"
    server_dir.mkdir()
    svc = EvalService(
        grade_fn, task_version="demo-v1", eval_set_version="items-q0-q4",
        base_dir=server_dir,
    )
    srv = serve(svc, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cfg = RemoteEvalConfig(
        base_url=f"http://127.0.0.1:{srv.server_address[1]}",
        poll_interval_s=0.01, poll_backoff_cap_s=0.05, job_timeout_s=10.0,
    )
    grader = RemoteGrader(cfg, pregate=make_pregate(AntiHackScanner()))
    yield svc, grader
    srv.shutdown()


def make_cand(code):
    return Candidate(id="cand-1", code=code, generation=1, parent_id=None,
                     island_idx=0, operator="rewrite")


def test_single_candidate_full_roundtrip(remote, tmp_path):
    svc, grader = remote
    workdir = tmp_path / "client"
    workdir.mkdir()
    report = grader.grade(make_cand('SOLVED = ["q0", "q1"]'), tmp_path / "client")

    assert report.fitness == pytest.approx(0.4)
    assert report.visible_metrics["solved"] == 2
    # The wire dict parses back into the framework-side feedback object (C1's
    # food): the JSON shape is the shared truth between the two sides.
    fb = StructuredFeedback.from_json(report.structured_feedback)
    assert fb.error_histogram == {"missing-q2": 1, "missing-q3": 1, "missing-q4": 1}
    assert (workdir / "remote_reply.json").exists()


def test_metadata_pins_eval_versions(remote, tmp_path):
    _, grader = remote
    cand = make_cand('SOLVED = ["q0"]')
    grader.grade(cand, tmp_path)
    assert cand.metadata["task_version"] == "demo-v1"
    assert cand.metadata["eval_set_version"] == "items-q0-q4"


def test_l0_pregate_blocks_banned_code_locally(remote, tmp_path):
    svc, grader = remote
    report = grader.grade(make_cand("import socket\nSOLVED = []"), tmp_path)
    assert report.passed is False
    assert report.stage_reached == 0
    assert "L0" in report.fault
    assert len(svc._jobs) == 0          # never crossed the wire


def test_evolution_loop_over_remote_grader(remote, tmp_path):
    """The crown: a real E1 evolution run where every fitness signal — and the
    C1 structured feedback it injects into prompts — arrives over HTTP."""
    _, grader = remote
    task = get_task("demo_counter")     # framework-side bundle: seed + fake LLM
    ctx = RecipeContext(
        search=SearchConfig(
            num_generations=8,
            operators=["rewrite"],
            operator_probs=[1.0],
            # demo mock transports emit duplicate programs by design;
            # the novelty gate would legitimately reject them all.
            novelty_enabled=False,
            seed=3,
            task_sys_msg=task.task_sys_msg,
        ),
        population=PopulationConfig(num_islands=1),
        plus=PlusConfig(),              # C1 is enabled by CHOOSING recipe e1
        grader=grader,                  # the ONLY line that differs from local
        llm=LLMClient(transport=task.transport, sleep=lambda s: None),
        run_dir=tmp_path / "run",
    )
    loop = recipes.get_recipe("e1").build(ctx)
    run_report = loop.run(task.initial_code)

    assert run_report.generations_completed == 8
    assert run_report.best_fitness > 0.2            # improved over the seed's 1/5
    assert run_report.total_eval_cost > 0           # remote eval_cost_usd flowed back
    best = loop.store.best()
    assert best.metadata["task_version"] == "demo-v1"   # §6 provenance on candidates


def test_l0_pregate_scans_side_files(remote, tmp_path):
    """M2.5 vaccine: a git candidate with a CLEAN main file and the banned
    import hidden in a side file must still be blocked locally — pre-upgrade,
    the scanner only ever saw main_text() and this cheat sailed through."""
    from evoharness.core.workspace import GitWorkspace

    svc, grader = remote
    ws = GitWorkspace(base_files={
        "main.py": "import helper\nSOLVED = helper.solve()\n",  # spotless
        "helper.py": "import socket\ndef solve():\n    return []\n",  # cheat
    })
    cand = Candidate(id="cand-git", code=ws.serialize(), generation=1,
                     parent_id=None, island_idx=0, operator="rewrite",
                     workspace_kind="git")
    report = grader.grade(cand, tmp_path)
    assert report.passed is False
    assert "banned-import" in report.fault
    assert "helper.py" in report.fault  # the fault names the culprit FILE
    assert len(svc._jobs) == 0  # blocked locally, never crossed the wire
