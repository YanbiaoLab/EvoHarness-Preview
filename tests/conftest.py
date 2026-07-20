import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evoharness.evocore import Candidate, EvalReport  # noqa: E402


def make_candidate(
    cid: str,
    fitness: float,
    passed: bool = True,
    children: int = 0,
    generation: int = 1,
    island: int = 0,
    in_archive: bool = True,
    embedding=None,
    parent_id=None,
    code: str = "",
) -> Candidate:
    return Candidate(
        id=cid,
        code=code or f"# program {cid}\n",
        generation=generation,
        parent_id=parent_id,
        island_idx=island,
        operator="rewrite",
        children_count=children,
        in_archive=in_archive,
        embedding=embedding,
        report=EvalReport(fitness=fitness, passed=passed),
        timestamp=float(generation),
    )
