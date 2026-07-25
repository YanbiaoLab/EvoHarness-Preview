"""Progress probe that works before and after the first checkpoint."""
import json, sys
from pathlib import Path

for name in sys.argv[1:] or ["e5s_verify"]:
    d = Path("results") / name
    log = Path("results") / f"{name}.log"
    ck = d / "checkpoint.json"
    if ck.exists():
        c = json.loads(ck.read_text())
        rr = c["run_report"]
        line = (f"{name}: gen {c['generation']} | evals {rr['evaluations']}"
                f" | failed {rr['proposals_failed']} | best {rr['best_fitness']}"
                f" | {rr['stopped_reason']}")
        for k, v in c.get("components", {}).items():
            if "Reflector" in k:
                pad = v.get("scratchpad") or ""
                line += f" | scratchpad v{v.get('scratchpad_version')} ({len(pad)} chars)"
        print(line)
    else:
        n = sum(1 for x in log.read_text(errors="ignore").splitlines()
                if "evaluating" in x) if log.exists() else 0
        print(f"{name}: seeding — {n} problem evaluations so far (no checkpoint yet)")
    jl = d / "experience.jsonl"
    if jl.exists():
        rows = [json.loads(x) for x in jl.read_text().splitlines() if x.strip()]
        kinds = {}
        for r in rows:
            kinds[r.get("kind", "evaluated")] = kinds.get(r.get("kind", "evaluated"), 0) + 1
        print(f"    experience.jsonl: {kinds}")
