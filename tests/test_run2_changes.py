"""Run-2 framework changes: repair throttling + heterogeneous island seeding."""

import recipes
from evoharness.evocore import LLMClient, PopulationConfig, SearchConfig
from evoharness.evoplus.config import PlusConfig
from recipes.common import RecipeContext
from tasks import get_task


class FailOnceGrader:
    """First non-seed candidate fails; everything else passes."""

    def __init__(self, inner):
        self.inner = inner
        self.failed_once = False

    def grade(self, cand, workdir):
        report = self.inner.grade(cand, workdir)
        if cand.operator != "seed" and not self.failed_once:
            self.failed_once = True
            report.passed = False
            report.fault = "boom"
        return report


def make_ctx(tmp_path, task, grader, islands=1, repair_probability=1.0):
    return RecipeContext(
        search=SearchConfig(
            num_generations=4, operators=["rewrite"], operator_probs=[1.0],
            seed=3, repair_probability=repair_probability,
            # demo mock transports emit duplicate programs by design;
            # the novelty gate would legitimately reject them all.
            novelty_enabled=False,
            task_sys_msg=task.task_sys_msg,
        ),
        population=PopulationConfig(num_islands=islands),
        plus=PlusConfig(),
        grader=grader,
        llm=LLMClient(transport=task.transport, sleep=lambda s: None),
        run_dir=tmp_path,
    )


def ops_history(report):
    return [h.get("operator") for h in report.history if "operator" in h]


def test_repair_probability_zero_never_repairs(tmp_path):
    task = get_task("demo_counter")
    loop = recipes.get_recipe("e0").build(
        make_ctx(tmp_path, task, FailOnceGrader(task.grader),
                 repair_probability=0.0)
    )
    report = loop.run(task.initial_code)
    assert "repair" not in ops_history(report)


def test_repair_probability_one_repairs(tmp_path):
    task = get_task("demo_counter")
    loop = recipes.get_recipe("e0").build(
        make_ctx(tmp_path, task, FailOnceGrader(task.grader),
                 repair_probability=1.0)
    )
    report = loop.run(task.initial_code)
    assert "repair" in ops_history(report)        # parity behaviour intact


def test_workspace_extra_seed_keeps_its_kind(tmp_path):
    """Multi-file island seeds: `code` holds a serialized workspace, so
    `workspace_kind` must say so — otherwise the store rebuilds the genome as
    a single file and the seed's side files vanish."""
    from evoharness.evocore.workspace import GitWorkspace

    task = get_task("s8_multifile")
    loop = recipes.get_recipe("e0").build(
        make_ctx(tmp_path, task, task.grader, islands=2)
    )
    variant = GitWorkspace(
        base_files={**task.initial_workspace.base_files,
                    "metadata.py": 'VERSION = "1"\n'},
    )
    loop.run(
        task.initial_code,
        extra_seeds=[variant],
        initial_workspace=task.initial_workspace,
    )

    seeded = [c for c in loop.store.all_candidates() if c.island_idx == 1
              and c.operator == "seed" and c.generation == 0]
    assert seeded, "variant did not reach island 1"
    variant_cand = seeded[-1]
    assert variant_cand.workspace_kind == "git"
    assert variant_cand.workspace.texts()["metadata.py"] == 'VERSION = "1"\n'


def test_string_extra_seed_is_lifted_into_the_primary_workspace(tmp_path):
    """A bare main-file string still works in multi-file mode: it is lifted
    into the primary genome so the side files come along."""
    task = get_task("s8_multifile")
    loop = recipes.get_recipe("e0").build(
        make_ctx(tmp_path, task, task.grader, islands=2)
    )
    loop.run(
        task.initial_code,
        extra_seeds=["# variant main\n" + task.initial_code],
        initial_workspace=task.initial_workspace,
    )

    seeded = [c for c in loop.store.all_candidates() if c.island_idx == 1
              and c.operator == "seed" and c.generation == 0]
    assert seeded
    texts = seeded[-1].workspace.texts()
    assert texts["main.py"].startswith("# variant main")
    assert "math_ops.py" in texts            # side files survived the lift


def test_extra_seeds_land_on_their_own_islands(tmp_path):
    task = get_task("demo_counter")
    loop = recipes.get_recipe("e0").build(
        make_ctx(tmp_path, task, task.grader, islands=2)
    )
    variant = task.initial_code.replace('"q0"', '"q0", "q1"')
    report = loop.run(task.initial_code, extra_seeds=[variant])
    assert report.evaluations >= 2 + 4            # primary + variant + gens
    island1 = loop.store.island_view(1)
    codes = [c.code for c in island1.passed_candidates]
    assert any('"q1"' in c for c in codes)        # variant seeded island 1
    assert loop.store.best().fitness >= 0.4       # variant (2/5) is the best seed
