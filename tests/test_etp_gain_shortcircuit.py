"""增益段的短路只在悬崖罚分口径下成立。

背景(run18 种子,2026-08-27):种子对着自己的参照物有 18 处回归 —— 跑间抖动,
不是能力退步。短路于是触发,`gain_plan=skipped_regressed`,60 行轮转增益档
一次都没跑,`wb_rotating` 显示 0/60。

**每个候选都会有几处回归**,所以在分级口径下这个短路会让整套防过拟合机制永远
测不到,而指标上只表现为"轮转档拿了 0 分",看不出是没跑。
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("ETP_SCORING", "official")


def _reload(penalty: str):
    import importlib

    os.environ["ETP_REGRESSION_PENALTY"] = penalty
    from experiments.etp_stage2 import grade

    return importlib.reload(grade)


@pytest.fixture(autouse=True)
def _restore():
    yield
    _reload("cliff")


def test_cliff_mode_keeps_the_short_circuit():
    """悬崖口径下掉一行就判 0,增益段跑了也改不了结论 —— 短路是对的。"""
    g = _reload("cliff")
    assert g.REGRESSION_PENALTY is None


def test_graded_mode_disables_the_short_circuit():
    g = _reload("1.0")
    assert g.REGRESSION_PENALTY == 1.0


@pytest.mark.parametrize(
    "penalty,regressed,expect_skip",
    [
        ("cliff", True, True),     # 悬崖 + 有回归 → 短路
        ("cliff", False, False),
        ("1.0", True, False),      # 分级 + 有回归 → **必须继续跑增益段**
        ("1.0", False, False),
    ],
)
def test_short_circuit_condition(penalty, regressed, expect_skip):
    """把 grade_workspace 里那个条件单独钉住。

    直接跑 grade_workspace 要判题器和几十分钟,这里只验条件本身 —— 它是一个
    两项的布尔式,而错的那一版在指标上不可见,正是需要被测试固定的形状。
    """
    g = _reload(penalty)
    assert bool(regressed and g.REGRESSION_PENALTY is None) is expect_skip


def test_graded_mode_still_prices_the_regression():
    """不短路不等于不罚:分级口径下回归照样按档权重扣两次。"""
    g = _reload("1.0")
    tiers = {n: g.TierScore(n, 100, 50, ["x"], []) for n in ("a", "b")}
    cost = sum(len(t.regressions) / (len(tiers) * t.total) for t in tiers.values())
    assert cost == pytest.approx(2 * (1 / (2 * 100)))
