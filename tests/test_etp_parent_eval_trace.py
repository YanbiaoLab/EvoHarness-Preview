"""ETP 的逐行评测轨迹:落冷 store → 脱敏出口 → agent 按需取。

为什么这条链值得单独钉住:它的每一环都**早就写好了**,而整条链断在没人接线上,
断法是完全静默的。

  · `structured_feedback.items` —— 每个候选 100 行,一直在算、一直在存
  · `FileArtifactStore` / `InspectParentEvalTool` / `AllowlistSanitizer`
    —— 框架里齐全,连 `failed_digest`(一次调用拿回全部失败题)都实现了
  · 唯一没接的是 ETP 的 grader 没有 store、任务没有把工具交出去

后果是 Austin 线连跑十几代、八个候选,agent 每一代拿到的只有 `solved: 28`
这一个汇总数字,而它需要的「哪 72 行、各自止步于哪个阶段」就躺在磁盘上。
断了不会报错、不会掉分,只会让进化变成盲搜。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from evoharness.core.agent.tools import InspectParentEvalTool
from evoharness.core.agent.tools.base import AgentToolError
from evoharness.core.artifacts import FileArtifactStore
from evoharness.core.llm import LLMToolCall

ROOT = Path(__file__).resolve().parents[1]


def _sanitizer():
    """从任务包里拿真的那一个 —— 测试自己造一个白名单等于什么都没验。"""
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
         "from experiments.etp_stage2 import task as T\n"
         "import json; print(json.dumps({\n"
         "  'item': sorted(T._TRACE_SANITIZER.item_allow['optimizer']),\n"
         "  'summary': sorted(T._TRACE_SANITIZER.summary_allow['optimizer']),\n"
         "}))" % (str(ROOT), str(ROOT / "experiments"))],
        capture_output=True, text=True, cwd=ROOT,
        env=os.environ | {"ETP_PROFILE": "austin", "PYTHONPATH": str(ROOT)},
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


class _Ctx:
    """AgentToolContext 里工具只用到 parent.report.artifacts_ref。"""

    class _Report:
        def __init__(self, ref):
            self.artifacts_ref = ref

    class _Parent:
        def __init__(self, ref):
            self.report = _Ctx._Report(ref)

    def __init__(self, ref):
        self.parent = _Ctx._Parent(ref)


def _blob():
    """一份贴近真实的轨迹:两行解出、三行没解出,失败原因各不相同。

    敏感字段是**故意**放进去的 —— 出口是 fail-closed 的白名单,不喂脏数据就
    验不出它在过滤。
    """
    return {
        "summary": {
            "solved": 2, "total": 5, "fitness": 0.4,
            "backend": "official", "solver_timeout_s": 600,
            "judge_url": "http://10.220.69.172:8900",     # 不该出去
            "run_dir": "/Users/someone/etp-work/run_austin",  # 不该出去
        },
        "items": [
            {"item_id": "austin_01", "passed": True, "expected": "false",
             "error_category": "", "stage": "infinite_v13", "seconds": 35},
            {"item_id": "austin_02", "passed": True, "expected": "false",
             "error_category": "", "stage": "infinite_v13", "seconds": 41},
            {"item_id": "austin_03", "passed": False, "expected": "false",
             "error_category": "unsolved", "stop_reason": "COUNTERMODEL_NOT_FOUND",
             "trace": "infinite_v13 EXHAUSTED after 58s", "seconds": 600,
             "sandbox_id": "default--math-distill-ggjzl"},   # 不该出去
            {"item_id": "austin_04", "passed": False, "expected": "false",
             "error_category": "budget_exhausted", "stop_reason": "HARD_TIMEOUT",
             "seconds": 600, "judge_url": "http://10.220.69.172:8900"},  # 不该出去
            {"item_id": "austin_05", "passed": False, "expected": "false",
             "error_category": "unsolved", "stop_reason": "RSS_BUDGET_REACHED",
             "trace": "killed at 600 MiB", "seconds": 120},
        ],
    }


@pytest.fixture
def tool_and_ref(tmp_path):
    allow = _sanitizer()
    from evoharness.core.sanitize import AllowlistSanitizer
    san = AllowlistSanitizer(
        item_allow={"optimizer": frozenset(allow["item"])},
        summary_allow={"optimizer": frozenset(allow["summary"])},
    )
    store = FileArtifactStore(tmp_path / "artifacts")
    ref = store.put("cand_a", _blob())
    return InspectParentEvalTool(store, san, audience="optimizer"), ref.encode()


def _call(tool, ref, action, **kw):
    args = {"action": action, "item_id": None, "query": None} | kw
    res = tool.invoke(LLMToolCall(call_id="c1", name="inspect_parent_eval",
                                  arguments=args), _Ctx(ref))
    return json.loads(res.content)


def test_failed_digest_returns_every_failure_in_one_call(tool_and_ref):
    """`failed_digest` 是这个工具的主用法:一次拿回全部失败题。

    存在的理由是实测的:轮次受限的 agent 会把整场会话花在一题一题读上
    (smoke run 复盘)。而 Austin 线上 72 行未解出、每代 160 轮 —— 一题一轮
    读完就没有轮次改代码了。
    """
    tool, ref = tool_and_ref
    got = _call(tool, ref, "failed_digest")
    assert got["failed_count"] == 3
    ids = {d["item_id"] for d in got["failed_items"]}
    assert ids == {"austin_03", "austin_04", "austin_05"}
    # 这三个 stop_reason 指向三种完全不同的改法,而汇总数字里它们长得一模一样。
    assert {d["stop_reason"] for d in got["failed_items"]} == {
        "COUNTERMODEL_NOT_FOUND", "HARD_TIMEOUT", "RSS_BUDGET_REACHED"}


def test_egress_drops_sandbox_ids_and_internal_urls(tool_and_ref):
    """出口是 fail-closed 白名单。内网拓扑不进提示词。

    放行一次的代价不是一次泄漏:轨迹会进每一轮的上下文,而上下文每轮重发。
    """
    tool, ref = tool_and_ref
    blob = json.dumps(_call(tool, ref, "failed_digest"))
    assert "10.220.69.172" not in blob
    assert "default--math-distill" not in blob
    assert "/Users/" not in blob
    summary = _call(tool, ref, "summary")["summary"]
    assert "judge_url" not in summary and "run_dir" not in summary
    # 但该出去的必须出去 —— 全删也能通过上面三条断言。
    assert summary["solved"] == 2 and summary["total"] == 5


def test_search_finds_rows_by_trace_text(tool_and_ref):
    """按轨迹文本搜 —— 「哪些行撞了内存墙」这种问题的唯一入口。

    匹配是**整条记录的大小写不敏感子串**,不是只搜 trace 字段。所以查
    "EXHAUSTED" 会连 `error_category: budget_exhausted` 一起命中 —— 这是对的
    行为(那一行确实也「耗尽」了),但意味着查询词要挑得能区分。下面用
    stop_reason 的原文,那是每行唯一的。
    """
    tool, ref = tool_and_ref
    assert _call(tool, ref, "search", query="COUNTERMODEL_NOT_FOUND")["matches"] == ["austin_03"]
    assert _call(tool, ref, "search", query="RSS_BUDGET_REACHED")["matches"] == ["austin_05"]
    assert _call(tool, ref, "search", query="600 MiB")["matches"] == ["austin_05"]
    # 宽泛的词会多命中,这一条把那件事本身钉住,免得以后有人当 bug 去"修"。
    assert _call(tool, ref, "search", query="exhausted")["matches"] == [
        "austin_03", "austin_04"]


def test_missing_trace_is_a_clear_error_not_a_silent_empty(tool_and_ref):
    """父本没有轨迹时要**报错**,不能返回空。

    返回空的话 agent 会读成「上一代没有失败」,那比没有这个工具更坏。
    """
    tool, _ = tool_and_ref
    with pytest.raises(AgentToolError) as exc:
        _call(tool, None, "summary")
    assert "no-parent-trace" in str(exc.value.code if hasattr(exc.value, "code") else exc.value)


def test_grade_writes_the_trace_and_task_hands_over_the_tool(tmp_path):
    """整条链:grader 落盘 → 任务交出工具 → 工具读得到。

    分开测三段都会绿而链子是断的 —— 这正是它断了十几代没人发现的原因。
    """
    script = (
        "import sys, json\n"
        "sys.path.insert(0, %r); sys.path.insert(0, %r)\n"
        "from experiments.etp_stage2 import grade as G, task as T\n"
        "store = G._artifact_store()\n"
        "tools = [t.definition.name for t in T._agent_tools()]\n"
        "print(json.dumps({'store': store is not None, 'tools': tools}))"
        % (str(ROOT), str(ROOT / "experiments"))
    )
    env = os.environ | {"ETP_PROFILE": "austin", "PYTHONPATH": str(ROOT),
                        "ETP_RUN_DIR": str(tmp_path),
                        "ETP_SEED_DIR": str(ROOT / "experiments/etp_stage2/seeds/v97fix")}
    out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, cwd=ROOT, env=env)
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout.strip().splitlines()[-1])
    assert got["store"] is True
    assert got["tools"] == ["inspect_parent_eval"]

    # 没有 run dir 就不给工具:一个必然报 no-parent-trace 的工具只会浪费轮次。
    env2 = {k: v for k, v in env.items() if k != "ETP_RUN_DIR"}
    out2 = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          text=True, cwd=ROOT, env=env2)
    assert out2.returncode == 0, out2.stderr
    got2 = json.loads(out2.stdout.strip().splitlines()[-1])
    assert got2 == {"store": False, "tools": []}
