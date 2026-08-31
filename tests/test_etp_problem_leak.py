"""题面泄漏扫描:基因组里不许出现评分集的题面原文。

为什么这道闸值得单独测:gemma 谱系的提交产物压缩后 398 KB,离 500 KB 上限还有
88 KB 余量,而评分集上的剩余空间只有 59 行。「把这 59 行的题面和答案背下来」拿到
的分,和「想出一个通用机制」一模一样,却便宜得多 —— 进化会挑便宜的那条,除非
这条路是关死的。

扫描按**规范化后的整份源码**做子串查找,所以要覆盖的绕过手法有三类:换运算符、
拆成多段拼接、藏进非 Python 的段。
"""

from __future__ import annotations

import json

import pytest

from experiments.etp_stage2 import rung0


EQ = "x = y ◇ (y ◇ (x ◇ y))"          # 长度 13(规范化后),过 _EQ_MIN_CHARS


@pytest.fixture()
def datasets(tmp_path):
    """一份最小评分集,只有一道题。"""
    d = tmp_path / "datasets"
    d.mkdir()
    (d / "wb_frontier.jsonl").write_text(
        json.dumps({"id": "wb_frontier_0000", "equation1": EQ,
                    "equation2": "x ◇ y = x ◇ (z ◇ ((z ◇ y) ◇ z))"},
                   ensure_ascii=False) + "\n",
        encoding="utf-8")
    return d


def _workspace(tmp_path, **files):
    ws = tmp_path / "ws"
    ws.mkdir(parents=True)
    for name, body in files.items():
        (ws / name).write_text(body, encoding="utf-8")
    (ws / "layout.json").write_text(
        json.dumps({"order": list(files)}), encoding="utf-8")
    return ws


def _leaks(ws, datasets):
    return [f for f in rung0.scan_workspace(ws, datasets)
            if f["kind"] == "scoring_equation_literal"]


def test_clean_genome_has_no_leak(tmp_path, datasets):
    ws = _workspace(tmp_path, **{"a.py": "def solve(eq1, eq2):\n    return None\n"})
    assert _leaks(ws, datasets) == []


def test_verbatim_equation_is_caught(tmp_path, datasets):
    ws = _workspace(tmp_path, **{"a.py": f"_C = {EQ!r}\n"})
    found = _leaks(ws, datasets)
    assert len(found) == 1
    assert found[0]["severity"] == "fail"


def test_operator_swap_does_not_evade(tmp_path, datasets):
    # `*` 和 `◇` 在渲染里混用,换一个符号不该成为绕过手段。
    ws = _workspace(tmp_path, **{"a.py": f"_C = {EQ.replace('◇', '*')!r}\n"})
    assert len(_leaks(ws, datasets)) == 1


def test_split_concatenation_does_not_evade(tmp_path, datasets):
    # 拆成两段拼接,规范化后仍是同一串。
    ws = _workspace(tmp_path, **{"a.py": f"_C = ({EQ[:9]!r}\n     {EQ[9:]!r})\n"})
    assert len(_leaks(ws, datasets)) == 1


def test_non_python_segment_is_scanned(tmp_path, datasets):
    # 非 Python 段 ast.parse 会抛 SyntaxError;泄漏扫描必须照做不误。
    ws = _workspace(tmp_path, **{"meta.json": json.dumps({"x": EQ},
                                                         ensure_ascii=False)})
    found = _leaks(ws, datasets)
    assert len(found) == 1
    assert found[0]["file"] == "meta.json"


def test_short_equations_are_not_candidates(tmp_path):
    """短题面不参与匹配 —— `x=y` 出现在任何解析器的测试串里都很正常。"""
    d = tmp_path / "datasets"
    d.mkdir()
    (d / "core.jsonl").write_text(
        json.dumps({"id": "core_0000", "equation1": "x = y",
                    "equation2": "y = x"}, ensure_ascii=False) + "\n",
        encoding="utf-8")
    assert rung0.scoring_equations(d) == set()
    ws = _workspace(tmp_path, **{"a.py": "_C = 'x = y'\n"})
    assert _leaks(ws, d) == []


def test_fingerprint_is_stable_across_files(tmp_path, datasets):
    """同一条泄漏换个文件藏,指纹不变 —— 否则挪一下就能冒充「新增」逃过基线差。"""
    a = _workspace(tmp_path / "one", **{"a.py": f"_C = {EQ!r}\n"})
    b = _workspace(tmp_path / "two", **{"zzz.py": f"_C = {EQ!r}\n"})
    assert _leaks(a, datasets)[0]["fingerprint"] == _leaks(b, datasets)[0]["fingerprint"]
