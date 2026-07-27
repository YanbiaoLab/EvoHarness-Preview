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


def test_an_island_with_its_own_seed_does_not_also_get_the_primary(tmp_path):
    """Heterogeneous seeding has to survive contact with selection.

    The primary used to be copied onto every island before the extra seeds
    landed, so an island held its native AND a copy of the primary. Selection
    is fitness-weighted, so wherever the primary outscored the native the
    island was a primary island in all but name. Run modmul_r7 seeded three
    architectures and evolved one: primary 0.846 against natives 0.218 and
    0.070, and no test noticed because none asserted on it.
    """
    task = get_task("demo_counter")
    loop = recipes.get_recipe("e0").build(
        make_ctx(tmp_path, task, task.grader, islands=3)
    )
    variant_a = task.initial_code.replace('"q0"', '"q0", "q1"')
    variant_b = task.initial_code.replace('"q0"', '"q0", "q1", "q2"')
    loop.run(task.initial_code, extra_seeds=[variant_a, variant_b])

    seeds_per_island = {
        idx: [
            c for c in loop.store.island_view(idx).candidates
            if c.operator == "seed"
        ]
        for idx in range(3)
    }
    assert [len(v) for v in seeds_per_island.values()] == [1, 1, 1]
    assert '"q1"' in seeds_per_island[1][0].code
    assert '"q2"' in seeds_per_island[2][0].code


def test_islands_without_a_seed_of_their_own_still_get_a_parent(tmp_path):
    """The blanket copy existed for a reason; keep that reason working.

    An island with no passed candidate is skipped by _pick_island for the
    whole run, so fewer extra seeds than islands must still leave every
    island able to produce a proposal.
    """
    task = get_task("demo_counter")
    loop = recipes.get_recipe("e0").build(
        make_ctx(tmp_path, task, task.grader, islands=4)
    )
    loop.run(task.initial_code, extra_seeds=[])

    for idx in range(4):
        assert loop.store.island_view(idx).passed_candidates, (
            f"island {idx} has no parent"
        )


def test_children_of_a_seed_copy_inherit_from_the_original(tmp_path):
    """A seed copy has an id but nothing was ever published under it.

    The copy is a database row: it reuses the original's report and is never
    handed to the grader, so a domain that keys per-candidate state by id --
    trained weights above all -- finds nothing for it. Run modmul_r7 reported
    "parent-published-no-weights" for 25 of its 26 offspring for exactly this
    reason, which means that run's whole fitness column measures cold starts
    rather than mutations.
    """
    task = get_task("demo_counter")
    loop = recipes.get_recipe("e0").build(
        make_ctx(tmp_path, task, task.grader, islands=2)
    )
    loop.run(task.initial_code, extra_seeds=[])

    original = next(
        c for c in loop.store.island_view(0).candidates if c.operator == "seed"
    )
    copy = next(
        c for c in loop.store.island_view(1).candidates if c.operator == "seed"
    )
    assert copy.id != original.id
    assert copy.metadata.get("seed_copy_of") == original.id

    children = [
        c for c in loop.store.all_candidates() if c.parent_id == copy.id
    ]
    assert children, "island 1 produced no offspring to check"
    for child in children:
        # parent_id still records the copy -- island bookkeeping stays honest.
        assert child.parent_id == copy.id
        assert child.metadata.get("lineage_parent_id") == original.id


def test_grade_context_redirects_lineage_parent_to_the_original(tmp_path):
    """The redirect has to survive all the way to the grader's context."""
    from evoharness.evocore.population import Candidate, EvalReport
    from evoharness.evoserve.grading import GradeContext
    from evoharness.task import WorkspaceGradeFnGrader

    seen: dict = {}

    def fake_grade(candidate_dir, ctx: GradeContext):
        seen["parent_id"] = ctx.parent_id
        return EvalReport(fitness=1.0, passed=True).to_json()

    grader = WorkspaceGradeFnGrader(fake_grade, lineage_dir=tmp_path / "lin")
    child = Candidate(
        id="child", code="# x\n", generation=1,
        parent_id="the-copy", island_idx=1, operator="rewrite",
        metadata={"lineage_parent_id": "the-original"},
    )
    grader.grade(child, tmp_path / "work")
    assert seen["parent_id"] == "the-original"

    plain = Candidate(
        id="child2", code="# y\n", generation=1,
        parent_id="an-ordinary-parent", island_idx=0, operator="rewrite",
    )
    grader.grade(plain, tmp_path / "work2")
    assert seen["parent_id"] == "an-ordinary-parent"
