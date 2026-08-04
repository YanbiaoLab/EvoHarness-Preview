"""Task registry: name -> TaskBundle factory."""

from __future__ import annotations


def _equational_stub():
    raise NotImplementedError(
        "equational task needs [HUMAN-INPUT]: eval samples + gold labels, "
        "the hand-tuned harness as initial program, and an LLM API key "
        "(plan phase B)"
    )


def _add_experiments_to_path():
    # Task packages live under experiments/ and are imported by bare name.
    import sys
    from pathlib import Path

    experiments = str(Path(__file__).resolve().parents[1] / "experiments")
    if experiments not in sys.path:
        sys.path.insert(0, experiments)


def _modmul():
    _add_experiments_to_path()

    from modmul.task import make_task  # lazy: pulls torch/modchallenge

    return make_task()


def _genesis_reward():
    _add_experiments_to_path()

    from genesis_reward.task import make_task  # lazy: reads HyperAgents paths

    return make_task()


def _polyglot_py():
    _add_experiments_to_path()

    from polyglot_py.task import make_task  # lazy: reads the exercise corpus

    return make_task()


def get_task(name: str):
    from . import demo_counter, s8_multifile

    registry = {
        "demo_counter": demo_counter.make_task,
        "s8_multifile": s8_multifile.make_task,
        "equational": _equational_stub,
        "modmul": _modmul,
        "genesis_reward": _genesis_reward,
        "polyglot_py": _polyglot_py,
    }
    try:
        return registry[name]()
    except KeyError:
        raise ValueError(
            f"unknown task {name!r}; available: {sorted(registry)}"
        ) from None
