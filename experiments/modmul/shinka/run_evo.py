#!/usr/bin/env python3
"""Launch ShinkaEvolve on modmul. Run from this directory.

    python run_evo.py --config_path config.yaml
"""

import argparse
import os
from pathlib import Path

import yaml

from shinka.core import EvolutionConfig, ShinkaEvolveRunner
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig
from shinka.launch import local as shinka_local

HERE = Path(__file__).resolve().parent


def _patch_process_return_code() -> None:
    """Give ProcessWithLogging the attribute its own monitor reads.

    Upstream `shinka/launch/local.py` ends monitor() with
    `return_code = process.return_code`, but ProcessWithLogging delegates
    unknown attributes to subprocess.Popen, which spells it `returncode`. So
    every local evaluation raises AttributeError after the job finishes, the
    runner catches it as "evaluation failed", and stores the placeholder
    `{"score": 0.0}` — the scores are computed correctly and then thrown away.
    Observed on ShinkaEvolve 7939f6b: three candidates, three real
    metrics.json files, three 0.000 rows in the database.

    Patched here rather than in the vendored clone because third_party/ is
    gitignored, so an edit there survives exactly until the next checkout.
    """
    if not hasattr(shinka_local.ProcessWithLogging, "return_code"):
        shinka_local.ProcessWithLogging.return_code = property(
            lambda self: self.process.returncode
        )

# What the search is told. Everything here was measured, not guessed — handing
# it over costs one prompt and saves the search from rediscovering a week of
# work. The one thing deliberately withheld is the answer: which combination
# of radix and margin actually lands under the deadline is what the run is for.
SEARCH_TASK_SYS_MSG = """\
You are optimising a neural network that computes (a * b) mod p for a
competition. The network is a width-generic Horner cell: a fixed, feedback-free
loop feeds the operand's digits in one at a time, and one learned cell performs
every transition s' = (2^k*s + d*x) mod p over bit vectors.

READ THIS FIRST — THE BOTTLENECK IS SPEED, NOT ACCURACY.

The current program already scores 100% on tiers 1-8 and 92% on tier 9. It
scores 0% on tier 10 for one reason: inference runs out of wall clock before
tier 10 starts. Tiers 1-9 consume 182 seconds against a 142 second allowance,
and all ten tiers project to 418 seconds. About a 2.94x inference speedup is
needed. Making the cell more accurate while it stays this slow is worth almost
nothing; making it faster without losing much accuracy is worth everything.

Watch `over_budget_factor` in the metrics. It is the speedup still needed.
`inference_seconds_by_tier` shows where the time goes — tier 9 alone eats 83%
of the entire budget, because inference cost is dominated by the Horner step
count, which is operand_bits / RADIX_BITS, and tier 9 has 2048-bit operands
(tier 10 has 4096).

TWO LEVERS ARE ALREADY IDENTIFIED. You are not restricted to them, but they
are where the measured headroom is.

1. RADIX_BITS. The loop consumes RADIX_BITS bits per step, so step count is
   operand_bits / RADIX_BITS. Raising it buys speed AND slack, because fewer
   steps means fewer chances to make a mistake. Tier 9's 92% over 5120 steps
   implies a per-step error rate of 1.63e-5. Carrying that to tier 10:

     RADIX_BITS=1 (now)  10240 steps  projects 84.6%  needs <=1.03e-5  (1.6x short)
     RADIX_BITS=2         5120 steps  projects 92.0%  needs <=2.06e-5  (already there)
     RADIX_BITS=4         2560 steps  projects 95.9%  needs <=4.12e-5  (2.5x margin)

   The cost is real: at k=1 the intermediate 2s + d*x stays under 3p, at k=4 it
   reaches 32p, and the pre-commit sweep measured bit-accuracy 0.73 vs 0.80
   under an equal budget. Pay for it with training, not by giving up.

2. The state register has no headroom. The state is sized to exactly the
   prime's bit length everywhere, so there are no spare high bits to absorb an
   intermediate. A sibling experiment measured 27 errors at zero margin versus
   3 errors with 16 spare bits, and margin costs nothing at inference time.
   TRAP: rounding the width up to a multiple of 32 lands in a margin of 1 or 2
   bits, which is worse than none. Add a fixed margin instead of rounding.

RULES YOU CANNOT BREAK. The region outside the EVOLVE-BLOCK markers is the
compliance contract and you cannot edit it, but you can still violate the rules
from inside a block, so:

  * The network must reduce the full-width operands itself. Never compute
    a % p, or any equivalent, in Python. This is the core of the task and
    moving it outside the network disqualifies the submission.
  * No big-integer shortcuts, lookup tables, or hand-coded modular algorithms
    (schoolbook, long division, Barrett, Montgomery, CRT) on the inference
    path. Exact integer arithmetic is allowed ONLY inside sample_batch, which
    synthesises training labels.
  * The names RADIX_BITS, MAX_WIDTH, HornerCell and train must keep existing
    with their current meaning; the frozen contract calls them.
  * A weight-perturbation gate runs automatically: randomising the cell's
    weights must collapse accuracy. If it does not, the answer is not coming
    from the trained parameters and the candidate scores zero.

Do not remove the bidirectional scan. Carries travel low-to-high but the
reduction decision is set by the high bits and must reach every low bit; an
upward-only scan plateaus at bit-accuracy 0.80 and never moves.
"""


def resolve_models(endpoint: dict | None) -> list[str] | None:
    """Build ShinkaEvolve's `local/<model>@<url>?api_key_env=<ENV>` identifiers.

    The base URL and key live in the environment (`.env` at the repo root), so
    neither ends up in a committed config file. Returns None when the config
    names no endpoint, leaving ShinkaEvolve's own defaults in charge.
    """
    if not endpoint:
        return None
    base_env = endpoint.get("api_base_env", "EVOHARNESS_API_BASE")
    key_env = endpoint.get("api_key_env", "EVOHARNESS_API_KEY")
    base_url = os.environ.get(base_env)
    if not base_url:
        raise SystemExit(
            f"{base_env} is not set. Run `set -a; source .env; set +a` first."
        )
    if not os.environ.get(key_env):
        raise SystemExit(f"{key_env} is not set. Run `set -a; source .env; set +a`.")
    return [
        f"local/{model}@{base_url.rstrip('/')}?api_key_env={key_env}"
        for model in endpoint["models"]
    ]


def main(config_path: str) -> None:
    _patch_process_return_code()
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    config["evo_config"]["task_sys_msg"] = SEARCH_TASK_SYS_MSG
    models = resolve_models(config.get("llm_endpoint"))
    if models:
        config["evo_config"]["llm_models"] = models
        # The novelty judge, when enabled, must reach the same endpoint.
        config["evo_config"].setdefault("novelty_llm_models", models)
    evo_config = EvolutionConfig(**config["evo_config"])
    job_config = LocalJobConfig(
        eval_program_path=str(HERE / "evaluate.py"),
        # One candidate trains for up to 90 minutes and then runs inference
        # over ten tiers. A cap below that would not slow candidates down, it
        # would truncate them mid-training and score the wreckage.
        time=config.get("job_time", "03:00:00"),
        # Without this, `import torch` on the L40S image dies with
        # `libucc.so.1: undefined symbol: ucs_config_doc_nop`.
        activate_script=config.get("activate_script"),
        python_executable=config.get("python_executable"),
    )
    db_config = DatabaseConfig(**config["db_config"])

    runner = ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=config.get("max_evaluation_jobs"),
        max_proposal_jobs=config.get("max_proposal_jobs"),
        max_db_workers=config.get("max_db_workers"),
        debug=False,
        verbose=True,
    )
    runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default=str(HERE / "config.yaml"))
    main(parser.parse_args().config_path)
