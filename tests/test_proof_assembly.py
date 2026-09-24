"""3d:根目标的凭据是组装后整体编译,不是逐节点的绿灯。

图说「证明了」是一句关于记账的话。这一层是把那句话兑现的地方——**在这之前,
fixture 从来没有被完整编译过一次**。

草图校验挡的是分解本身错;这一道挡的是**装配漂移**:引理是在不同时刻、不同
run 里分别证出来的,凑齐时 Mathlib 版本、兄弟签名、`sorry` 的填法都可能已经动过。
"""

import shutil

import pytest

from evoharness.proof.assembly import (
    AmbiguousRoute,
    AssemblyError,
    assemble,
    certify,
    verify,
)
from evoharness.proof.controller import FixedDecompositions, ProofController
from evoharness.proof.graph import DecompositionStatus, GoalStatus, Outcome
from evoharness.proof.sketch import LeanSketchValidator, Validation, render
from evoharness.proof.solver import StubSolver
from evoharness.proof.store import ProofGraphStore

from proof_fixture import (
    ALTERNATE_ROOT_SKETCH,
    FIXTURE_PROOFS,
    FIXTURE_SIGNATURE,
    FIXTURE_SKETCH,
    LAZY_SKETCH,
    NESTED_CHILD_SKETCH,
    NESTED_PREAMBLE,
    NESTED_PROOFS,
    NESTED_ROOT_SKETCH,
)

ROOT_IDENTITY = "text:fixture_main"

pytestmark = pytest.mark.skipif(
    shutil.which("lean") is None,
    reason="组装的凭据来自 Lean;没有 lean 就没有什么可验的",
)


def accept_all(goal, sketch):
    return Validation(ok=True)


@pytest.fixture
def store(tmp_path):
    graph = ProofGraphStore(tmp_path / "graph.db")
    yield graph
    graph.close()


def solve_fixture(store, *, sketch=FIXTURE_SKETCH, validate=None):
    """跑完整条链:根目标直接证失败 → 分解 → 三个子目标各自证完。"""

    root = store.upsert_goal(ROOT_IDENTITY, FIXTURE_SIGNATURE)
    store.add_root(root.id, "fixture")
    solver = StubSolver(
        script={
            ROOT_IDENTITY: [Outcome.TASK_FAILED],
            **{identity: [Outcome.PROVED] for identity in FIXTURE_PROOFS},
        },
        proofs=FIXTURE_PROOFS,
    )
    controller = ProofController(
        store,
        solver,
        decompositions=FixedDecompositions({ROOT_IDENTITY: sketch}),
        validate_sketch=validate or accept_all,
    )
    return root, controller.solve(root.id, budget=100)


# --- 端到端 ------------------------------------------------------------------

def test_the_assembled_fixture_actually_compiles(store):
    """P-3 的主验收:子目标全绿,**而且**拼起来送给 Lean 也绿。"""

    root, report = solve_fixture(store)
    assert report.root_proved is True

    result = verify(store, root.id)
    assert result.ok is True, result.reason
    assert result.forbidden_axioms == frozenset()
    assert "sorry" not in result.text


def test_the_axiom_report_is_what_certifies_the_root(store):
    """1.0 不来自状态机,来自 Lean 说它只依赖那三条公理。"""

    root, _ = solve_fixture(store)
    result = verify(store, root.id)

    assert result.axioms <= frozenset(
        {"propext", "Quot.sound", "Classical.choice"}
    )
    assert "sorryAx" not in result.axioms


def test_assembly_splices_the_proved_bodies_into_the_sketch(store):
    """组装走的是 `sketch.render`——和校验器验过的那份文件同一个函数,
    区别只在引理体是 `sorry` 还是真证明。两个拼装器会漂,而漂了看不见。"""

    root, _ = solve_fixture(store)
    text = assemble(store, root.id)

    for body in FIXTURE_PROOFS.values():
        assert body in text
    assert "sorry" not in text


# --- 装配漂移会被抓住 --------------------------------------------------------

def test_a_lemma_body_that_does_not_typecheck_is_caught_at_assembly(store):
    """每个节点都被记成绿的,但拼起来编译不过——这正是逐节点绿灯挡不住的那类。"""

    root = store.upsert_goal(ROOT_IDENTITY, FIXTURE_SIGNATURE)
    store.add_root(root.id, "fixture")
    drifted = dict(FIXTURE_PROOFS)
    # 一条引理的证明体换成一个类型对不上的项:单独看它「有」证明,拼进去不成立。
    drifted["text:lemma_assoc"] = "Nat.mul_zero a"
    solver = StubSolver(
        script={
            ROOT_IDENTITY: [Outcome.TASK_FAILED],
            **{identity: [Outcome.PROVED] for identity in drifted},
        },
        proofs=drifted,
    )
    ProofController(
        store,
        solver,
        decompositions=FixedDecompositions({ROOT_IDENTITY: FIXTURE_SKETCH}),
        validate_sketch=accept_all,
    ).solve(root.id, budget=100)

    assert store.goal(root.id).status is GoalStatus.PROVED  # 图相信它成了

    result = verify(store, root.id)
    assert result.ok is False
    assert "did not survive being put together" in result.reason


def test_a_proved_goal_without_proof_text_is_a_wiring_fault_not_a_finding(store):
    """「已证但没有东西可组装」是这一层本该杜绝的状态,所以它抛异常而不是返回
    ok=False——后者会说成「这个证明是错的」,那是强得多的断言。"""

    root = store.upsert_goal(ROOT_IDENTITY, FIXTURE_SIGNATURE)
    store.add_root(root.id, "fixture")
    store._conn.execute(
        "INSERT INTO attempts (id, goal_id, outcome, proof_text, cost,"
        " created_at) VALUES ('att_x', ?, 'proved', NULL, 0, 1.0)",
        (root.id,),
    )
    store._conn.commit()
    store.propagate(root.id, max_capability_attempts=3)

    with pytest.raises(AssemblyError, match="no proof text"):
        assemble(store, root.id)


# --- 草图校验(真 Lean) ------------------------------------------------------

def test_the_real_validator_accepts_the_fixture_sketch(store):
    root = store.upsert_goal(ROOT_IDENTITY, FIXTURE_SIGNATURE)
    verdict = LeanSketchValidator()(store.goal(root.id), FIXTURE_SKETCH)

    assert verdict.ok is True
    assert "3 lemmas" in verdict.reason


def test_a_sketch_that_leaves_sorry_in_the_parent_is_rejected(store):
    """它编译得好好的。只有 `sorry` 位置检查能看出它只是把目标挪了个地方。"""

    root = store.upsert_goal(ROOT_IDENTITY, FIXTURE_SIGNATURE)
    verdict = LeanSketchValidator()(store.goal(root.id), LAZY_SKETCH)

    assert verdict.ok is False
    assert "moves the goal rather than reducing it" in verdict.reason


def test_a_sketch_for_a_different_proposition_is_rejected(store):
    """草图声称关掉的命题必须就是这个目标,否则组装的是别的东西。"""

    root = store.upsert_goal(ROOT_IDENTITY, "theorem other : True")
    verdict = LeanSketchValidator()(store.goal(root.id), FIXTURE_SKETCH)

    assert verdict.ok is False
    assert "different proposition" in verdict.reason


def test_rendering_records_where_each_declaration_landed(store):
    """行号是排版时记下的,不是回头解析出来的——我们自己排的文件,
    再去猜它落在哪一行等于凭空制造一个不确定性。"""

    rendered = render(FIXTURE_SKETCH)
    lines = rendered.text.splitlines()

    for name, line in rendered.declaration_lines.items():
        assert lines[line - 1].startswith(f"theorem {name} ")


# --- 认证进图 ------------------------------------------------------------------

def test_certify_records_what_verify_found(store):
    """`verify` 只回答;`certify` 把回答写进图,`status` 才分得出「路线闭合」
    和「成品编译通过」。"""

    root, _ = solve_fixture(store)
    assert store.latest_certification(root.id) is None

    result, certification = certify(store, root.id)

    assert result.ok is True
    assert certification.goal_id == root.id
    read = store.latest_certification(root.id)
    assert read is not None
    assert read.ok is True
    assert read.axioms == result.axioms
    assert read.text_sha256 != ""


def test_a_failed_assembly_is_certified_as_failed(store):
    """组装编不过是关于图的真发现,必须留痕;只记成功会把它显示成「尚未认证」。"""

    root = store.upsert_goal(ROOT_IDENTITY, FIXTURE_SIGNATURE)
    store.add_root(root.id, "fixture")
    drifted = dict(FIXTURE_PROOFS)
    drifted["text:lemma_assoc"] = "Nat.mul_zero a"
    ProofController(
        store,
        StubSolver(
            script={
                ROOT_IDENTITY: [Outcome.TASK_FAILED],
                **{identity: [Outcome.PROVED] for identity in drifted},
            },
            proofs=drifted,
        ),
        decompositions=FixedDecompositions({ROOT_IDENTITY: FIXTURE_SKETCH}),
        validate_sketch=accept_all,
    ).solve(root.id, budget=100)

    result, certification = certify(store, root.id)

    assert result.ok is False
    assert certification.ok is False
    assert store.latest_certification(root.id).ok is False


# --- 两层图:拼接 -------------------------------------------------------------

def solve_nested(store):
    """根 → lemma_distrib → lemma_distrib_step,两层都要真的走一遍。

    在这个 fixture 出现之前,所有图都是一层的:叶子经 `_direct_proof` 只贡献
    body,`assemble` 里那条把子树整段拼进来的分支**一次都没有执行过**——不在
    任何一次真实会话里,也不在任何一个用例里。
    """

    root = store.upsert_goal(ROOT_IDENTITY, FIXTURE_SIGNATURE)
    store.add_root(root.id, "fixture")
    solver = StubSolver(
        script={
            ROOT_IDENTITY: [Outcome.TASK_FAILED],
            "text:lemma_distrib": [Outcome.TASK_FAILED],
            **{identity: [Outcome.PROVED] for identity in NESTED_PROOFS},
        },
        proofs=NESTED_PROOFS,
    )
    controller = ProofController(
        store,
        solver,
        decompositions=FixedDecompositions({
            ROOT_IDENTITY: NESTED_ROOT_SKETCH,
            "text:lemma_distrib": NESTED_CHILD_SKETCH,
        }),
        validate_sketch=accept_all,
    )
    return root, controller.solve(root.id, budget=100)


def test_a_two_level_graph_assembles_into_a_file_lean_accepts(store):
    """承重项。两层之前必炸,而炸法有三种,Lean 只报得出它撞到的第一种。"""

    root, report = solve_nested(store)
    assert report.root_proved is True

    result = verify(store, root.id)
    assert result.ok is True, result.reason
    assert "sorry" not in result.text


def test_the_import_is_hoisted_and_emitted_once(store):
    """`import` 只在文件开头合法。原来每个 sketch 各渲染一次 preamble,
    子树一进来第二个 import 就落到了文件中部。"""

    root, _ = solve_nested(store)
    lines = assemble(store, root.id).splitlines()

    assert lines[0] == NESTED_PREAMBLE
    assert [line for line in lines if line.startswith("import")] == [
        NESTED_PREAMBLE
    ]


def test_a_spliced_subtree_declares_its_lemma_exactly_once(store):
    """子树已经声明了 `lemma_distrib`,父层就不能再声明一次。

    原来的写法两处都发:子树发一份真证明,父层再发一份、体是
    `lemma_distrib_assembled`——**一个没有任何地方定义的名字**。Lean 停在
    重复的 import 上,所以这一条一直躲在后面。
    """

    root, _ = solve_nested(store)
    text = assemble(store, root.id)

    declarations = [
        line for line in text.splitlines()
        if line.startswith("theorem lemma_distrib ")
    ]
    assert len(declarations) == 1
    assert "_assembled" not in text


# --- 两层图:选路线 -----------------------------------------------------------

def add_alternate_route(store, root_id):
    """给根目标再加一条已完成的扁平路线,复用已经证出来的 `lemma_distrib`。

    这正是 PB-Basic-008 的形状:模型看出第一条路线会撞上拼接缺陷,另提一条、
    Lean 也接受了——然后组装器照旧去拿最老的那条。
    """

    distrib = store.goal_by_identity("text:lemma_distrib")
    decomposition = store.add_decomposition(
        root_id,
        [(spec.identity, spec.signature) for spec in ALTERNATE_ROOT_SKETCH.subgoals],
        sketch=ALTERNATE_ROOT_SKETCH,
        status=DecompositionStatus.ACCEPTED,
    )
    store.set_decomposition_status(
        decomposition.id, DecompositionStatus.COMPLETED
    )
    assert distrib is not None
    return decomposition


def test_two_completed_routes_and_no_choice_is_a_question_not_a_pick(store):
    """承重项。原来取 `completed[0]`,也就是**最早**的那条——而「早」和
    「这份证明是不是你要的」毫无关系,更糟的是选择过程完全不可见。"""

    root, _ = solve_nested(store)
    alternate = add_alternate_route(store, root.id)

    with pytest.raises(AmbiguousRoute) as caught:
        assemble(store, root.id)

    named = {route_id for route_id, _ in caught.value.candidates}
    assert alternate.id in named
    assert len(named) == 2
    # 报错本身要能回答:被拒的一方通常是个只有一轮机会的模型,
    # 光给 id 会逼它再查一次图才知道那两条路线分别是什么。
    assert "lemma_distrib" in str(caught.value)


def test_naming_a_route_is_what_decides_the_file(store):
    """选路线是「值得试哪条」,不是「哪条为真」——Lean 仍然要重编译整篇。"""

    root, _ = solve_nested(store)
    alternate = add_alternate_route(store, root.id)

    text = assemble(store, root.id, routes=[alternate.id])

    # 扁平那条只声明 lemma_distrib,另外两个合取项在 parent_body 里就地收掉。
    assert "theorem lemma_assoc " not in text
    assert "theorem lemma_distrib " in text
    result = verify(store, root.id, routes=[alternate.id])
    assert result.ok is True, result.reason


def test_pinning_a_route_further_down_the_tree_is_normal_use(store):
    """子目标也可能有多条路线,所以 `routes` 是可重复的:只钉住顶上那个,
    调用方就答不了下面那一层给它的拒绝。"""

    root, _ = solve_nested(store)
    child = store.goal_by_identity("text:lemma_distrib")
    child_route = store.decompositions_of(child.id)[0]

    text = assemble(store, root.id, routes=[child_route.id])
    assert "lemma_distrib_step" in text


def test_a_named_route_that_never_applied_is_refused_not_ignored(store):
    """默默忽略是最坏的结局:调用方以为自己选过了,却拿到了另一份文件,
    而且面前没有任何东西提示这件事。"""

    root, _ = solve_nested(store)
    alternate = add_alternate_route(store, root.id)
    child = store.goal_by_identity("text:lemma_distrib")

    # 根的路线对「只组装 lemma_distrib」这棵子树不适用。
    with pytest.raises(AssemblyError) as caught:
        assemble(store, child.id, routes=[alternate.id])
    assert alternate.id in str(caught.value)

    with pytest.raises(KeyError):
        assemble(store, root.id, routes=["dec_does_not_exist"])


def test_a_route_that_did_not_complete_cannot_be_assembled_through(store):
    """只有子目标全证完的路线才拼得出文件;`accepted` 只说明 Lean 认可这条路,
    不说明有人走完了。"""

    root, _ = solve_nested(store)
    half = store.add_decomposition(
        root.id,
        [(spec.identity, spec.signature) for spec in ALTERNATE_ROOT_SKETCH.subgoals],
        sketch=ALTERNATE_ROOT_SKETCH,
        status=DecompositionStatus.ACCEPTED,
    )

    with pytest.raises(AssemblyError) as caught:
        assemble(store, root.id, routes=[half.id])
    assert "not completed" in str(caught.value)


def test_the_certification_records_which_route_was_compiled(store):
    """`text_sha256` 承诺「文件可从图复现」。选路一旦交给调用方,同一份图
    能渲染出好几份文件,不记路线的哈希读起来像出处、其实不是。"""

    root, _ = solve_nested(store)
    alternate = add_alternate_route(store, root.id)

    _, certification = certify(store, root.id, routes=[alternate.id])

    assert certification.decomposition_id == alternate.id
    assert store.latest_certification(root.id).decomposition_id == alternate.id


def test_a_directly_proved_goal_certifies_with_no_route_not_an_unknown_one(store):
    """空串和 null 是两件事:前者说「没有路线可选」,后者说「这条记录写在
    开始记录路线之前」。合成一个值就再也分不开了。"""

    root, _ = solve_fixture(store)
    leaf = store.goal_by_identity("text:lemma_assoc")

    _, certification = certify(store, leaf.id)

    assert certification.decomposition_id == ""
    assert certification.decomposition_id is not None


# -- the board's header comes first, however the goal was closed --------------


def test_a_directly_proved_goal_is_assembled_under_the_boards_header(tmp_path):
    """A goal the solver closed in one go has no sketch to carry the header, and its
    file used to come out with none: a Mathlib proof failed to parse once assembled."""

    from evoharness.proof.assembly import assemble
    from evoharness.proof.graph import Outcome
    from evoharness.proof.store import ProofGraphStore

    store = ProofGraphStore(tmp_path / "graph.db")
    goal = store.upsert_goal("id:odd", "theorem odd (n : ℕ) : ∑ i ∈ Finset.range n, (2 * i + 1) = n ^ 2")
    store.record_attempt(goal.id, Outcome.PROVED, proof_text="by\n  induction n <;> simp_all [Finset.sum_range_succ]; ring")
    store.propagate(goal.id, max_capability_attempts=3)
    text = assemble(store, goal.id, preamble="import Mathlib\nopen Nat")
    assert text.startswith("import Mathlib\nopen Nat\n\ntheorem odd")
    assert assemble(store, goal.id).startswith("theorem odd"), "no header given, none added"
    store.close()


def test_a_header_carried_by_a_sketch_is_not_written_twice(tmp_path):
    from evoharness.proof.assembly import assemble
    from evoharness.proof.graph import DecompositionStatus, Outcome
    from evoharness.proof.sketch import Sketch, SubgoalSpec
    from evoharness.proof.store import ProofGraphStore

    store = ProofGraphStore(tmp_path / "graph.db")
    root = store.upsert_goal("id:root", "theorem root : True")
    sketch = Sketch(parent_name="root", parent_signature="theorem root : True", parent_body="l1",
                    subgoals=(SubgoalSpec("l1", "id:l1", "theorem l1 : True"),),
                    preamble="import Mathlib")
    d = store.add_decomposition(root.id, [("id:l1", "theorem l1 : True")], sketch=sketch)
    store.set_decomposition_status(d.id, DecompositionStatus.ACCEPTED)
    store.record_attempt(d.subgoal_ids[0], Outcome.PROVED, proof_text="trivial")
    store.propagate(d.subgoal_ids[0], max_capability_attempts=3)
    text = assemble(store, root.id, preamble="import Mathlib")
    assert text.count("import Mathlib") == 1 and text.startswith("import Mathlib\n")
    store.close()


def test_the_cli_assembles_under_the_boards_header(tmp_path, capsys, monkeypatch):
    import json

    from evoharness.proof import cli
    from evoharness.proof.graph import Outcome
    from evoharness.proof.store import ProofGraphStore

    monkeypatch.delenv(cli.WORK_ENV, raising=False)
    monkeypatch.delenv(cli.PROJECT_ENV, raising=False)
    seen = {}

    class Runner:
        def compile(self, text):
            seen["text"] = text
            return 0, "'odd' depends on axioms: [propext]"

    monkeypatch.setattr(cli, "_runner", lambda args: Runner())
    assert cli.main(["--work", str(tmp_path), "open", "--preamble", "import Mathlib",
                     "--statement", "theorem odd : True"]) == 0
    goal_id = json.loads(capsys.readouterr().out)["opened"]["goal_id"]
    store = ProofGraphStore(tmp_path / "graph.db")
    store.record_attempt(goal_id, Outcome.PROVED, proof_text="trivial")
    store.propagate(goal_id, max_capability_attempts=3)
    store.close()
    assert cli.main(["--work", str(tmp_path), "assemble", "--goal", goal_id]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert seen["text"].startswith("import Mathlib\n\ntheorem odd : True := trivial")
