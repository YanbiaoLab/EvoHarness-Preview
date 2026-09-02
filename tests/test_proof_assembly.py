"""3d:根目标的凭据是组装后整体编译,不是逐节点的绿灯。

图说「证明了」是一句关于记账的话。这一层是把那句话兑现的地方——**在这之前,
fixture 从来没有被完整编译过一次**。

草图校验挡的是分解本身错;这一道挡的是**装配漂移**:引理是在不同时刻、不同
run 里分别证出来的,凑齐时 Mathlib 版本、兄弟签名、`sorry` 的填法都可能已经动过。
"""

import shutil

import pytest

from evoharness.proof.assembly import AssemblyError, assemble, certify, verify
from evoharness.proof.controller import FixedDecompositions, ProofController
from evoharness.proof.graph import GoalStatus, Outcome
from evoharness.proof.sketch import LeanSketchValidator, Validation, render
from evoharness.proof.solver import StubSolver
from evoharness.proof.store import ProofGraphStore

from proof_fixture import (
    FIXTURE_PROOFS,
    FIXTURE_SIGNATURE,
    FIXTURE_SKETCH,
    LAZY_SKETCH,
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
