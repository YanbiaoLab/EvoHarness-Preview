import json
"""Tier-1 frozen research brief: contributor + all-groups injection."""

from evoharness.core import LLMClient, MutationContext
from evoharness.evoplus import StaticBriefContributor

import recipes
from conftest import make_candidate
from tasks import get_task
from test_recipes import _ctx


def _mctx():
    return MutationContext(make_candidate("p", 1.0), [], [], "revise", 1)


def test_contributor_injects_header_and_text():
    section = StaticBriefContributor("Use an inverted index.").contribute(_mctx())
    assert section.startswith("# Domain research brief")
    assert "Use an inverted index." in section
    assert "may be incomplete" in section  # honesty caveat travels with it


def test_contributor_empty_and_truncation():
    assert StaticBriefContributor("   ").contribute(_mctx()) is None
    long = StaticBriefContributor("x" * 10_000, max_bytes=100)
    assert len(long.text.encode()) <= 100


def test_contributor_from_file(tmp_path):
    path = tmp_path / "research_brief.md"
    path.write_text("Frozen finding: prefer beam width 3.")
    section = StaticBriefContributor.from_file(path).contribute(_mctx())
    assert "beam width 3" in section


def test_brief_shared_by_all_groups_and_ordered_first(tmp_path):
    """The brief appears in EVERY group's prompts (task infrastructure, not
    an ablation), and precedes recipe contributor sections."""
    for name in ("e0", "e1", "e3r"):
        task = get_task("demo_counter")
        captured = []
        inner = task.transport

        def spy(messages, model, _inner=inner, _cap=captured, **kw):
            _cap.append(messages[0].content)
            return _inner(messages=messages, model=model, **kw)

        task.transport = spy  # before build: the proposer holds the llm ref
        ctx = _ctx(tmp_path / name, task, generations=4)
        ctx.research_brief = "BRIEF-MARKER: use set lookups."
        loop = recipes.get_recipe(name).build(ctx)
        loop.run(task.initial_code)

        assert captured, f"{name}: spy transport never called"
        assert all("BRIEF-MARKER" in s for s in captured), name
        with_feedback = [s for s in captured if "Parent failure analysis" in s]
        for system in with_feedback:  # brief section precedes C1 section
            assert system.index("BRIEF-MARKER") < system.index(
                "Parent failure analysis"
            )


def test_oversized_brief_warns_instead_of_dropping_silently(tmp_path, caplog):
    """A brief is the one prompt section carrying human-chosen evidence.

    Truncating it silently drops whichever fact was written last, and the loss is
    invisible from both ends: the file on disk still reads complete, and the model
    never sees that anything is missing.
    """
    import logging

    from conftest import make_candidate
    from evoharness.core.interfaces import MutationContext
    from evoharness.evoplus import IslandBriefContributor

    path = tmp_path / "island_briefs.json"
    tail = "THE-LAST-FACT"
    path.write_text(json.dumps({"0": "x" * 400 + tail}), encoding="utf-8")

    led = IslandBriefContributor(path, max_bytes=200)
    ctx = MutationContext(parent=make_candidate("p", 1.0, island=0),
                          archive_inspirations=[], top_k_inspirations=[],
                          operator="revise", generation=1)
    with caplog.at_level(logging.WARNING):
        out = led.contribute(ctx)
    assert tail not in out, "precondition: the cap must actually bite here"
    assert any("truncated" in r.message for r in caplog.records), \
        "a dropped brief tail must not be silent"
