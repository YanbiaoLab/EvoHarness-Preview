"""Score a candidate on the digit-power rule over a wide probe range.

The probe range is the load-bearing choice. A grader that samples only small
`n` accepts a candidate that memorised a handful of answers, and the memorised
candidate then wins the search while implementing nothing. `calibration/memorised`
exists to make that failure a load error rather than a result nobody questions.
"""

from pathlib import Path

# Wide enough that memorising is not a strategy, small enough to grade in
# milliseconds. Squares of four-digit inputs keep the arithmetic exact.
PROBE_RANGE = range(1, 501)


def _expected(n: int) -> int:
    return sum(int(digit) for digit in str(n * n))


def _load(candidate_dir: Path):
    namespace: dict = {}
    source = (Path(candidate_dir) / "main.py").read_text(encoding="utf-8")
    # The candidate may split its work across files, so the genome directory is
    # importable from inside its own module while it executes.
    import sys

    sys.path.insert(0, str(candidate_dir))
    try:
        exec(compile(source, "main.py", "exec"), namespace)
    finally:
        sys.path.remove(str(candidate_dir))
    solve = namespace.get("solve")
    if not callable(solve):
        raise AttributeError("main.py defines no callable solve(n)")
    return solve


def grade(candidate_dir, ctx):
    solve = _load(candidate_dir)

    passed = 0
    first_failures: list[str] = []
    for n in PROBE_RANGE:
        try:
            ok = solve(n) == _expected(n)
        except Exception:
            # A candidate's own exception is a wrong answer, not a grader
            # fault: letting it escape would turn "this genome is broken" into
            # "evaluation is broken", and the two get handled very differently.
            ok = False
        if ok:
            passed += 1
        elif len(first_failures) < 5:
            first_failures.append(f"solve({n}) != {_expected(n)}")

    total = len(PROBE_RANGE)
    return {
        "fitness": passed / total,
        "visible_metrics": {"passed": passed, "total": total},
        "notes": "; ".join(first_failures),
        # Reported so coverage is a measured fact rather than an assumption:
        # every probe ran, so every probe is usable evidence.
        "n_units": total,
        "trustworthy_units": total,
    }
