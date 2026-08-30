"""身份哈希:合并该发生的,以及**绝不能**发生的那种合并。

这是整层里唯一一条能直接产出假结论的路径——合并两个目标等于说「A 的证明也
证明了 B」,它们要不是同一个,一个未证目标就被标成已证了。所以失败方向的用例
比成功用例更重要。
"""

import shutil

import pytest

from evoharness.proof import identity as identity_module
from evoharness.proof.identity import (
    ExactTextHasher,
    LeanExprHasher,
    is_unresolved,
)
from evoharness.proof.sketch import SketchUnavailable

ASSOC_ABC = "theorem h (a b c : Nat) : (a + b) + c = a + (b + c)"
ASSOC_XYZ = "theorem h (x y z : Nat) : (x + y) + z = x + (y + z)"
DISTRIB = "theorem h (a b c : Nat) : (a + b) * c = a * c + b * c"

needs_lean = pytest.mark.skipif(
    shutil.which("lean") is None, reason="alpha 等价要问 Lean"
)


# --- 保守的默认 --------------------------------------------------------------

def test_the_default_hasher_merges_only_identical_text():
    """漏合并只损失效率,错合并损失正确性。默认站在会漏的那一边。"""

    hasher = ExactTextHasher()
    same, spaced, other = hasher.hash_many(
        [ASSOC_ABC, "  ".join(ASSOC_ABC.split()), ASSOC_XYZ]
    )

    assert same == spaced          # 只是空白不同,是同一条
    assert same != other           # alpha 变体,保守起见不合并


# --- alpha 等价 --------------------------------------------------------------

@needs_lean
def test_alpha_variants_get_one_identity():
    """绑定变量换个名字不该在图上变成两个节点,否则记忆化白做。"""

    abc, xyz, distrib = LeanExprHasher().hash_many(
        [ASSOC_ABC, ASSOC_XYZ, DISTRIB]
    )

    assert abc == xyz
    assert abc != distrib


@needs_lean
def test_hypotheses_are_part_of_the_statement():
    """两条结论相同、前提不同的引理不是同一条引理。

    这条钉的是 `_proposition`:绑定必须挪到冒号左边,否则被 elaborate 的只是
    结论,而带假设和不带假设的会撞成一个。
    """

    with_hypothesis = "theorem h (a : Nat) (ha : a = 0) : a + a = 0"
    without = "theorem h (a : Nat) : a + a = 0"
    first, second = LeanExprHasher().hash_many([with_hypothesis, without])

    assert first != second


# --- 失败方向(承重) --------------------------------------------------------

@needs_lean
def test_a_statement_lean_cannot_elaborate_does_not_borrow_anyone_else_s_key():
    """elaborate 不了的命题必须拿到一个独有的 key,不能落到别人身上。"""

    good, broken = LeanExprHasher().hash_many(
        [ASSOC_ABC, "theorem h (a : Nat) : this_symbol_does_not_exist a"]
    )

    assert is_unresolved(broken)
    assert broken != good


@needs_lean
def test_two_unelaborable_statements_do_not_merge_with_each_other():
    """两个都算不出身份的目标,尤其不能因为「都失败了」而被认成同一个。"""

    first, second = LeanExprHasher().hash_many(
        ["theorem h : nope_one", "theorem h : nope_two"]
    )

    assert is_unresolved(first) and is_unresolved(second)
    assert first != second


def test_a_dead_toolchain_merges_nothing(monkeypatch):
    """Lean 起不来的时候,整批都拿到独有 key——宁可一次复用都不做。"""

    def explode(*args, **kwargs):
        raise SketchUnavailable("no lean on PATH")

    monkeypatch.setattr(identity_module, "compile_lean", explode)
    keys = LeanExprHasher().hash_many([ASSOC_ABC, ASSOC_XYZ, ASSOC_ABC])

    assert all(is_unresolved(key) for key in keys)
    assert len(set(keys)) == 3


# --- 批量 --------------------------------------------------------------------

@needs_lean
def test_the_whole_batch_costs_one_lean_process():
    """`import Lean` 要几秒。每个目标问一次会让图不可用,所以接口是批量的。"""

    calls = {"n": 0}
    real = identity_module.compile_lean

    def counted(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    import unittest.mock

    with unittest.mock.patch.object(identity_module, "compile_lean", counted):
        keys = LeanExprHasher().hash_many([ASSOC_ABC, ASSOC_XYZ, DISTRIB])

    assert calls["n"] == 1
    assert len(keys) == 3


def test_an_empty_batch_does_not_start_lean(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("should not have run lean")

    monkeypatch.setattr(identity_module, "compile_lean", explode)
    assert LeanExprHasher().hash_many([]) == []
