"""Turn IMO-LeanProofBench rows into the shape the proof layer consumes.

    uv run python tasks/lean_bench/extract.py \
        --csv ~/.../superhuman/imobench/lean_proof_bench_v2.csv \
        --ids PB-Basic-001 PB-Basic-002 PB-Basic-003

The CSV's `Lean Statement` is a whole file: imports, a comment carrying the
informal problem and its answer, then `theorem NAME ... := by sorry`. The graph
wants those pieces apart -- a preamble it can re-emit above assembled proofs,
and a signature it can hash, render and hand to a solver.

**The informal comment is dropped.** It states the answer in prose, and every
one of these problems is "determine all f such that ...": leaving it in the
seed would hand the solver the result and the run would measure nothing. That
is the same leak `tests/test_etp_problem_leak.py` exists to catch a version of.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

#: `theorem NAME` / `lemma NAME`, at the start of a line.
_DECL_RE = re.compile(r"^(theorem|lemma)\s+(\S+)", re.MULTILINE)
_OPEN, _CLOSE = "([{⦃", ")]}⦄"


def split_statement(text: str) -> dict:
    """(preamble, name, signature) from one `Lean Statement` cell."""

    match = _DECL_RE.search(text)
    if not match:
        raise ValueError("no theorem/lemma declaration found")

    preamble = _preamble(text[: match.start()])

    body = text[match.start():]
    cut = _top_level_assign(body)
    if cut is None:
        raise ValueError("declaration has no `:=`")
    return {
        "name": match.group(2),
        "preamble": preamble,
        "signature": body[:cut].rstrip(),
    }


def _preamble(head: str) -> str:
    """Everything before the declaration except the informal comment.

    An allowlist of `import` and `set_option` is NOT enough, and the way it
    fails is quiet: PB-Advanced-001 carries `open scoped Classical`, without
    which its `Finset.filter` cannot synthesize `DecidablePred` and the
    statement does not compile. A seed that does not compile is not `passed`,
    so the island has no parent and the run emits no proposals -- which reads
    exactly like a model that could not solve the problem.

    So: drop `/- ... -/` blocks and `--` lines, keep whatever else is there.
    """

    kept: list[str] = []
    in_block = False
    for line in head.splitlines():
        stripped = line.strip()
        if in_block:
            if "-/" in stripped:
                in_block = False
            continue
        if stripped.startswith("/-"):
            in_block = "-/" not in stripped
            continue
        if not stripped or stripped.startswith("--"):
            continue
        kept.append(line.rstrip())
    return "\n".join(kept).strip()


def _top_level_assign(body: str) -> int | None:
    """Index of the `:=` that ends the signature.

    Depth-aware: `(f : ℤ → ℤ)` and set-builder braces both contain colons, and
    `{f | ...}` can contain `:=` inside a nested term in principle. Splitting on
    the first `:=` found would cut a statement in half and leave something Lean
    cannot elaborate -- which resolves to "unresolved" and silently stops the
    problem from ever merging or being hashed.
    """

    depth = 0
    for index in range(len(body) - 1):
        char = body[index]
        if char in _OPEN:
            depth += 1
        elif char in _CLOSE:
            depth -= 1
        elif char == ":" and body[index + 1] == "=" and depth == 0:
            return index
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument(
        "--out", default=str(Path(__file__).parent / "problems.json")
    )
    args = parser.parse_args()

    rows = {r["Problem ID"]: r for r in csv.DictReader(open(args.csv))}
    problems = []
    for problem_id in args.ids:
        row = rows[problem_id]
        parts = split_statement(row["Lean Statement"])
        problems.append({
            "id": problem_id,
            "level": row["Level"],
            "category": row["Category"],
            "source": row["Source"],
            **parts,
        })
    Path(args.out).write_text(
        json.dumps(problems, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(problems)} problems to {args.out}")
    for problem in problems:
        print(f"  {problem['id']:18s} {problem['level']:12s} {problem['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
