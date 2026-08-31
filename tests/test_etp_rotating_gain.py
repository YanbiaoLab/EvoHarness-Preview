"""轮转增益档:每代换一批没见过的题,让「背下某一行」下一代就不给分。

这套机制有两个容易写错、写错了又看不出来的地方,都在这里钉住:

  1. 轮转行**不能进参照物**。一旦进了,下一代它们就成了守卫行,解不出判回归 ——
     一批本来只是「这代随机抽到」的题,变成了永久义务。
  2. 参照物的 ref_fitness **不能把轮转档算进去**。参照物在那一档天然是 0
     (那些行它压根没测过),算进去会把它系统性拉低,于是"打赢参照物"变成白送。
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("ETP_SCORING", "official")

from experiments.etp_stage2 import grade as G  # noqa: E402


def test_pool_excludes_scoring_set_overlap():
    """按 (eq1_id, eq2_id) 排重,不按 id —— 两个数据集 id 命名体系不同。"""
    pool = G._wb3500_pool()
    if not pool:
        pytest.skip("wrong_book_3500.jsonl 不在")
    taken = {(r.get("eq1_id"), r.get("eq2_id")) for r in G.fitness_set()}
    assert not [r for r in pool if (r["eq1_id"], r["eq2_id"]) in taken]


def test_pool_uses_rendered_equations():
    """深层引擎只认 ◇,喂 * 会静默失败 —— 不报错,只是一道都解不出。"""
    pool = G._wb3500_pool()
    if not pool:
        pytest.skip("wrong_book_3500.jsonl 不在")
    for row in pool[:200]:
        assert "*" not in row["equation1"] + row["equation2"]
        assert "◇" in row["equation1"] + row["equation2"]


def test_sample_is_reproducible_within_a_generation():
    """同代所有候选必须看到同一批题,否则同代之间不可比。"""
    if not G._wb3500_pool():
        pytest.skip("wrong_book_3500.jsonl 不在")
    a = [r["id"] for r in G.rotating_set(7)]
    b = [r["id"] for r in G.rotating_set(7)]
    assert a == b and len(a) == G.ROTATING_ROWS


def test_sample_rotates_across_generations():
    """跨代必须换题 —— 不换的话背题又开始划算了。"""
    if not G._wb3500_pool():
        pytest.skip("wrong_book_3500.jsonl 不在")
    a = {r["id"] for r in G.rotating_set(1)}
    b = {r["id"] for r in G.rotating_set(2)}
    # 3372 行里抽 60,期望重叠约 1 行。放宽到 1/4 也远低于"没换题"。
    assert len(a & b) < len(a) // 4


def test_rotating_rows_all_land_in_the_gain_segment():
    """轮转行不在参照物里 → 一定被划进增益段 → 永远不产生回归。"""
    if not G._wb3500_pool():
        pytest.skip("wrong_book_3500.jsonl 不在")
    rotating = G.rotating_set(3)
    reference = {r["id"]: True for r in G.fitness_set()}   # 固定集全解出
    guard = [r for r in rotating if reference.get(r["id"])]
    assert guard == []


def test_rotating_tier_is_dropped_when_disabled(monkeypatch):
    """ROTATING_ROWS=0 时退回七档旧口径,一字不差。"""
    monkeypatch.setattr(G, "ROTATING_ROWS", 0)
    assert G.rotating_set(1) == []
    rows = G.fitness_set()
    tiers = G.tier_scores(rows, [True] * len(rows), None)
    assert "wb_rotating" not in tiers
    assert len(tiers) == 7


def test_rotating_tier_carries_one_eighth_of_fitness():
    """八档等权:轮转档的权重必须是 1/8,而不是被题数稀释掉。

    这是整套机制的要害 —— 固定集上剩余空间只有 0.0851(七档口径),轮转档
    一档就值 0.125。育种信号的主项从此是「解出没见过的题」。
    """
    if not G._wb3500_pool():
        pytest.skip("wrong_book_3500.jsonl 不在")
    rows = G.fitness_set() + G.rotating_set(1)
    all_solved = G.tier_scores(rows, [True] * len(rows), None)
    assert len(all_solved) == 8
    assert G.combine(all_solved) == pytest.approx(1.0)

    # 轮转档全灭、其余全解出 → 正好掉 1/8
    vector = [r["stratum"] != "wb_rotating" for r in rows]
    partial = G.tier_scores(rows, vector, None)
    assert G.combine(partial) == pytest.approx(7 / 8)
