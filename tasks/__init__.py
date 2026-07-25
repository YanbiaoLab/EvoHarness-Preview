"""Task registry: name -> TaskBundle factory."""

from __future__ import annotations


def _equational_stub():
    raise NotImplementedError(
        "equational task needs [HUMAN-INPUT]: eval samples + gold labels, "
        "the hand-tuned harness as initial program, and an LLM API key "
        "(plan phase B)"
    )


def _modmul():
    # Task packages live under experiments/ and are imported by bare name.
    import sys
    from pathlib import Path

    experiments = str(Path(__file__).resolve().parents[1] / "experiments")
    if experiments not in sys.path:
        sys.path.insert(0, experiments)

    from modmul.task import make_task  # lazy: pulls torch/modchallenge

    return make_task()


def get_task(name: str):
    from . import demo_counter, s8_multifile

    registry = {
        "demo_counter": demo_counter.make_task,
        "s8_multifile": s8_multifile.make_task,
        "equational": _equational_stub,
        "modmul": _modmul,
    }
    try:
        return registry[name]()
    except KeyError:
        raise ValueError(
            f"unknown task {name!r}; available: {sorted(registry)}"
        ) from None
