"""读回一次尝试:板子只记一个词,而「为什么」在编译器那句话里。

这个视图是被一次真实会话逼出来的。PB-Advanced-006 那 8 小时里,模型用 25 次
`glob`/`read`/`grep` 自己扒了 `.evo/runs/*/attempt_*/run/` ——包括拿手写正则去
grep 求解器的 `events.jsonl`。**需求是真的,而它满足需求的方式是解析没人承诺过
形状的内部文件**:那些文件一变,这条路静默失效,而模型不会知道自己瞎了。

两条规矩定了这个视图的形状:

- **窄是构造出来的,不是删出来的**(和 `readout/peer.py` 同一条理由);
- **被切断的跑没有得出任何结论**,所以 `conclusive` 必须在载荷里,不在文档里。
"""

import json
import sqlite3
from pathlib import Path

import pytest

from evoharness.proof.graph import Outcome
from evoharness.proof.inspect import InspectError, attempt_view
from evoharness.proof.store import ProofGraphStore

GOAL = ("id:leaf", "theorem leaf (a : Nat) : a + 0 = a")


@pytest.fixture
def store(tmp_path):
    graph = ProofGraphStore(tmp_path / "graph.db")
    yield graph
    graph.close()


def make_run(run_dir: Path, *, report: dict, metadata: dict, code: str,
             generation: int = 1) -> None:
    """一个只有 `candidates` 表的最小 run.db,字段名跟真库一致。"""

    database = run_dir / "run"
    database.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(database / "run.db")
    conn.execute(
        "CREATE TABLE candidates (id TEXT, code TEXT, generation INTEGER,"
        " operator TEXT, change_title TEXT, change_summary TEXT,"
        " report TEXT, metadata TEXT)"
    )
    conn.execute(
        "INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?)",
        ("seed", json.dumps({"base_files": {"subgoal.lean": "-- seed"}}), 0,
         "seed", "initial program", "", json.dumps({"passed": True,
         "fitness": 0.3, "notes": ""}), json.dumps({})),
    )
    conn.execute(
        "INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?)",
        ("later", json.dumps({"base_files": {"subgoal.lean": code}}),
         generation, "rewrite", "Zero-set descent", "tried halving",
         json.dumps(report), json.dumps(metadata)),
    )
    conn.commit()
    conn.close()


FAILED_REPORT = {
    "passed": False,
    "fitness": 0.0,
    "fault": "the file does not compile",
    "notes": (
        "/var/folders/pb/x/tmpabc/sketch.lean:16:29: error: omega could not "
        "prove the goal:\n"
        "/var/folders/pb/x/tmpabc/sketch.lean:43:35: error: Type mismatch"
    ),
    "hidden_metrics": {"secret_score": 0.99},
    "stdout_log": "SHOULD NOT LEAK",
    "stderr_log": "SHOULD NOT LEAK EITHER",
}

EFFORT = {
    "turns": 22, "tool_calls": 26, "repair_rounds": 0,
    "agent_elapsed_s": 818.4, "prompt_tokens": 232465,
    "completion_tokens": 19869, "termination": "completed", "cost_usd": 0.0,
}


def failed_attempt(store, tmp_path, outcome=Outcome.TASK_FAILED):
    goal = store.upsert_goal(*GOAL)
    run_dir = tmp_path / "runs" / goal.id / "attempt_0000"
    make_run(run_dir, report=FAILED_REPORT, metadata=EFFORT,
             code="import Mathlib\ntheorem leaf : True := trivial\n")
    store.record_attempt(goal.id, outcome, run_dir=str(run_dir),
                         note="stopped_reason=completed best_fitness=0.3")
    return goal


# --- 它到底说了什么 ----------------------------------------------------------

def test_the_compiler_errors_are_what_comes_back(store, tmp_path):
    """求解器的 summary 是它**以为**自己在干什么;Lean 的报错才是发生了什么。"""

    goal = failed_attempt(store, tmp_path)
    view = attempt_view(store, goal.id)

    assert view["conclusive"] is True
    latest = view["tried"][0]
    assert latest["lean_errors"] == [
        "16:29: error: omega could not prove the goal:",
        "43:35: error: Type mismatch",
    ]
    assert latest["fault"] == "the file does not compile"
    assert view["effort"]["turns"] == 22
    assert view["effort"]["prompt_tokens"] == 232465


def test_the_temp_path_is_stripped_but_the_position_is_kept(store, tmp_path):
    """那个临时目录早就不在了,而行列号和 `code` 对得上,是唯一能用的部分。"""

    goal = failed_attempt(store, tmp_path)
    errors = attempt_view(store, goal.id)["tried"][0]["lean_errors"]

    assert not any("/var/folders" in line for line in errors)
    assert all(line[0].isdigit() for line in errors)


def test_the_view_is_built_not_filtered(store, tmp_path):
    """承重项。拿完整报告删字段今天也能过,而下一个新字段默认外泄。"""

    goal = failed_attempt(store, tmp_path)
    text = json.dumps(attempt_view(store, goal.id), ensure_ascii=False)

    assert "SHOULD NOT LEAK" not in text
    assert "hidden_metrics" not in text
    assert "secret_score" not in text
    assert "stdout_log" not in text


def test_the_code_comes_only_when_asked_and_only_for_the_latest(store, tmp_path):
    goal = failed_attempt(store, tmp_path)

    assert "code" not in attempt_view(store, goal.id)["tried"][0]

    tried = attempt_view(store, goal.id, include_code=True)["tried"]
    assert "theorem leaf" in tried[0]["code"]
    assert all("code" not in entry for entry in tried[1:])


# --- 被切断的跑不是结论 ------------------------------------------------------

@pytest.mark.parametrize(
    "outcome",
    [Outcome.INTERRUPTED, Outcome.INFRA_FAILED, Outcome.BUDGET_EXHAUSTED],
)
def test_a_run_that_was_stopped_says_so_in_the_payload(store, tmp_path, outcome):
    """承重项。被切断的那次留下的是它**恰好停在哪**,不是它得出了什么。

    和一次 `task-failed` 并排平铺,那堆残骸读起来就是发现,而读的人会在上面
    编出一个连贯的故事——这一条写在载荷里,不写在文档里。
    """

    goal = failed_attempt(store, tmp_path, outcome=outcome)
    view = attempt_view(store, goal.id)

    assert view["conclusive"] is False
    assert "not a verdict" in view["caveat"]
    assert outcome.value in view["caveat"]


def test_an_attempt_with_no_directory_says_that_rather_than_nothing(store):
    """空的 `tried` 读起来像「它什么都没试」。那是两件事。"""

    goal = store.upsert_goal(*GOAL)
    store.record_attempt(goal.id, Outcome.INTERRUPTED, note="lease expired")

    view = attempt_view(store, goal.id)

    assert view["tried"] == []
    assert "nothing of what it tried can be read back" in view["caveat"]


def test_naming_an_attempt_that_is_not_on_this_goal_is_refused(store, tmp_path):
    goal = failed_attempt(store, tmp_path)

    with pytest.raises(InspectError):
        attempt_view(store, goal.id, "att_does_not_exist")

    other = store.upsert_goal("id:other", "theorem other : True")
    with pytest.raises(InspectError):
        attempt_view(store, other.id)
