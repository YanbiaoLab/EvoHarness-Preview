"""Batched dispatch (SearchConfig.eval_batch_size): concurrency proof and
bookkeeping parity, on the offline demo task."""

import threading

import recipes
from evoharness.evocore import LLMClient, PopulationConfig, SearchConfig
from evoharness.evoplus.config import PlusConfig
from recipes.common import RecipeContext
from tasks import get_task


class BarrierGrader:
    """Releases only when `parties` grade() calls are IN FLIGHT together
    (the seed grades alone and is exempt). A serial loop deadlocks here,
    so a pass is a proof of concurrency, not a hint of it."""

    def __init__(self, inner, parties: int):
        self.inner = inner
        self.barrier = threading.Barrier(parties)

    def grade(self, cand, workdir):
        if cand.operator != "seed":
            self.barrier.wait(timeout=10)
        return self.inner.grade(cand, workdir)


def make_ctx(tmp_path, task, grader, batch):
    return RecipeContext(
        search=SearchConfig(
            num_generations=3, operators=["rewrite"], operator_probs=[1.0],
            seed=3, eval_batch_size=batch, task_sys_msg=task.task_sys_msg,
            # demo mock transports emit duplicate programs by design;
            # the novelty gate would legitimately reject them all.
            novelty_enabled=False,
        ),
        population=PopulationConfig(num_islands=1),
        plus=PlusConfig(),
        grader=grader,
        llm=LLMClient(transport=task.transport, sleep=lambda s: None),
        run_dir=tmp_path,
    )


def test_batch_generation_grades_concurrently(tmp_path):
    task = get_task("demo_counter")
    grader = BarrierGrader(task.grader, parties=2)
    loop = recipes.get_recipe("e0").build(make_ctx(tmp_path, task, grader, batch=2))
    report = loop.run(task.initial_code)
    assert report.generations_completed == 3
    assert report.evaluations == 1 + 3 * 2        # seed + gens * batch
    assert report.best_fitness is not None
    assert report.stopped_reason == "completed"


def test_batch_size_one_is_serial_parity(tmp_path):
    task = get_task("demo_counter")
    loop = recipes.get_recipe("e0").build(
        make_ctx(tmp_path, task, task.grader, batch=1)
    )
    report = loop.run(task.initial_code)
    assert report.evaluations == 1 + 3            # unchanged serial semantics
