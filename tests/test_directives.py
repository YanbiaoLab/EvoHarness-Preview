"""HITL v1: directive book, guidance TTL, lineage veto, prompt injection."""

import json

import pytest

import recipes
from conftest import make_candidate
from evoharness.core import LLMClient, MutationContext, PopulationConfig, PopulationStore
from evoharness.evoplus import (
    DirectiveBook,
    HumanDirectiveContributor,
    LineageVetoPolicy,
    append_directive,
)
from tasks import get_task
from test_recipes import _ctx


def _write(path, directives, version=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": version, "directives": directives}))


def test_append_directive_assigns_ids_and_versions(tmp_path):
    ctl = tmp_path / "directives.json"
    d1 = append_directive(ctl, {"kind": "guidance", "text": "try caching"})
    d2 = append_directive(ctl, {"kind": "lineage_veto", "candidate_ids": ["x"]})
    assert (d1["id"], d2["id"]) == ("d1", "d2")
    data = json.loads(ctl.read_text())
    assert data["version"] == 2 and len(data["directives"]) == 2
    with pytest.raises(ValueError, match="unknown directive kind"):
        append_directive(ctl, {"kind": "bogus"})


def test_book_reloads_and_expires_guidance(tmp_path):
    ctl = tmp_path / "directives.json"
    book = DirectiveBook(ctl)
    assert book.active(1) == []  # missing file is fine
    _write(ctl, [{"id": "d1", "kind": "guidance", "text": "go", "ttl_generations": 3}])
    assert len(book.active(5)) == 1  # first seen at gen 5
    assert len(book.active(7)) == 1  # 5+3 exclusive
    assert book.active(8) == []      # expired
    # file update reloads (new directive visible)
    _write(ctl, [
        {"id": "d1", "kind": "guidance", "text": "go", "ttl_generations": 3},
        {"id": "d2", "kind": "guidance", "text": "stop tuning temperature"},
    ], version=2)
    texts = [d.text for d in book.active(9)]
    assert texts == ["stop tuning temperature"]  # d1 expired, d2 alive


def test_contributor_injects_guidance(tmp_path):
    ctl = tmp_path / "directives.json"
    book = DirectiveBook(ctl)
    contrib = HumanDirectiveContributor(book)
    ctx = MutationContext(make_candidate("p", 1.0), [], [], "revise", 4)
    assert contrib.contribute(ctx) is None
    _write(ctl, [{"id": "d1", "kind": "guidance", "text": "use inverted index"}])
    section = contrib.contribute(ctx)
    assert section.startswith("# Reviewer directives")
    assert "use inverted index" in section


def test_lineage_veto_covers_descendants(tmp_path):
    store = PopulationStore(PopulationConfig())
    root = make_candidate("root", 1.0)
    child = make_candidate("child", 2.0, parent_id="root", generation=2)
    grand = make_candidate("grand", 3.0, parent_id="child", generation=3)
    other = make_candidate("other", 1.5)
    for c in (root, child, grand, other):
        store.insert(c)
    ctl = tmp_path / "directives.json"
    _write(ctl, [{"id": "d1", "kind": "lineage_veto", "candidate_ids": ["child"]}])
    policy = LineageVetoPolicy(DirectiveBook(ctl), store)
    assert policy.weight_multiplier(root) == 1.0   # ancestor unaffected
    assert policy.weight_multiplier(child) == 0.0
    assert policy.weight_multiplier(grand) == 0.0  # descendant vetoed
    assert policy.weight_multiplier(other) == 1.0


def test_directives_flow_into_prompts_via_assemble(tmp_path):
    task = get_task("demo_counter")
    captured = []
    inner = task.transport

    def spy(messages, model, **kw):
        captured.append(messages[0].content)
        return inner(messages=messages, model=model, **kw)

    task.transport = spy  # must be set BEFORE build: the proposer holds llm
    _write(tmp_path / "control" / "directives.json",
           [{"id": "d1", "kind": "guidance", "text": "HITL-MARKER: solve q3 first"}])
    ctx = _ctx(tmp_path, task, generations=3)
    loop = recipes.get_recipe("e0").build(ctx)
    loop.run(task.initial_code)
    assert captured, "spy transport never called"
    assert any("HITL-MARKER" in s for s in captured)
    assert "directive_book" in ctx.extras
