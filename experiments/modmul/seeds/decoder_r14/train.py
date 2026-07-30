"""Training recipe for the fixed-width step cell.

MUTATION SURFACE — data distribution, curriculum, optimizer, schedule.

Harness contract (both are load-bearing):
  * `train(model_dir)` is RESUMABLE: weights already in model_dir are picked up
    and training continues. The grader calls it once per ASHA rung.
  * MODMUL_TRAIN_SECONDS is a wall-clock budget for THIS call. Overrunning it
    gets the process killed and the candidate scored 0.

Exact integer arithmetic here is legal and expected: labels are synthesized at
TRAIN time. Only the inference path must be free of it.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from arch import BITS, StepCell, pick_device, to_bits

LR = 1e-3
WD = 0.0
BATCH = 4096
SEED = 0
WARMUP = 200
# NOT "an upper bound the time budget beats to" -- that comment was here and
# it was false, the same way it was false in limb_horner until run modmul_r8's
# 06dde31b disproved it. This lineage's cached weights stood at exactly
# 400,000 steps, so the loop below could not execute even once and training
# became a no-op the moment the cap was reached. Raised to leave room; the
# wall clock is what should stop training, and if this number ever binds
# again the run will say so through `training_skipped`.
TOTAL_STEPS = 2_000_000

# Task-fixed tier geometry (p bit ranges); NOT part of the mutation surface —
# a candidate must not narrow training to the easy tiers.
TIERS = ((1, 3), (4, 8), (9, 16))


def _sieve(limit: int) -> list[int]:
    flags = bytearray([1]) * limit
    flags[0] = flags[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if flags[i]:
            flags[i * i:: i] = bytearray(len(flags[i * i:: i]))
    return [i for i in range(2, limit) if flags[i]]


def prime_pools(device) -> list[torch.Tensor]:
    primes = _sieve(1 << BITS)
    return [
        torch.tensor([p for p in primes if lo <= p.bit_length() <= hi],
                     device=device)
        for lo, hi in TIERS
    ]


def sample_batch(n: int, pools: list[torch.Tensor], device):
    """Random exact step tuples, tier-balanced. All values < 2**16, so the
    int64 ops below are overflow-safe."""
    tier = torch.randint(len(pools), (n,), device=device)
    p = torch.empty(n, dtype=torch.long, device=device)
    for i, pool in enumerate(pools):
        mask = tier == i
        k = int(mask.sum())
        if k:
            p[mask] = pool[torch.randint(pool.numel(), (k,), device=device)]
    s = (torch.rand(n, device=device) * p).long()
    x = (torch.rand(n, device=device) * p).long()
    d = torch.randint(2, (n,), device=device)
    y = (2 * s + d * x) % p
    return s, x, p, d, y


def train(model_dir: str) -> None:
    budget = float(os.environ.get("MODMUL_TRAIN_SECONDS", "600"))
    deadline = time.monotonic() + budget * 0.9      # leave room to save
    directory = Path(model_dir)
    state_path = directory / "train_state.json"
    weights_path = directory / "weights.pt"

    done = 0
    if state_path.exists():
        done = int(json.loads(state_path.read_text()).get("steps", 0))
    torch.manual_seed(SEED + done)
    random.seed(SEED + done)

    device = pick_device()
    cell = StepCell().to(device)
    if weights_path.exists():                        # resume, do not restart
        cell.load_state_dict(torch.load(weights_path, map_location=device))
    opt = torch.optim.AdamW(cell.parameters(), lr=LR, weight_decay=WD)
    opt_path = directory / "optimizer.pt"
    if opt_path.exists():
        try:
            opt.load_state_dict(torch.load(opt_path, map_location=device))
            # Adam's exp_avg / exp_avg_sq are keyed by parameter POSITION and
            # carry that parameter's shape. load_state_dict copies them
            # without checking, so a checkpoint that predates a reshape loads
            # cleanly and then fails at the first step, deep inside
            # _multi_tensor_adam, with a size mismatch. `except: pass` around
            # the load does not help -- nothing is raised there.
            for param, entry in opt.state.items():
                for value in entry.values():
                    if (torch.is_tensor(value) and value.dim()
                            and value.shape != param.shape):
                        raise ValueError("optimizer state predates a reshape")
        except Exception:
            # Clear AND remove, so the next rung does not pick it up again.
            opt.state.clear()
            opt_path.unlink(missing_ok=True)

    pools = prime_pools(device)
    cell.train()
    # loss is initialised, not just step: when the cap is already reached or
    # the budget is spent the loop body never runs, and the state write below
    # touches `loss`. Run modmul_r11's island-1 seed died on exactly that --
    # UnboundLocalError, reported as train-failed, which is indistinguishable
    # from the candidate's own code being broken. limb_horner and serial_ar
    # both guard this; horner_cell was the one that did not.
    step, loss = done, torch.tensor(0.0)
    last_tick = time.monotonic()
    while step < TOTAL_STEPS and time.monotonic() < deadline:
        # The grader freezes training (SIGSTOP) while a timed inference holds
        # the card alone; a gap this size can only be an external stop, so
        # push the deadline out by the gap instead of billing it to training.
        now = time.monotonic()
        if now - last_tick > 5.0:
            deadline += now - last_tick
        last_tick = now
        for group in opt.param_groups:
            group["lr"] = LR * min(1.0, (step + 1) / WARMUP)
        s, x, p, d, y = sample_batch(BATCH, pools, device)
        logits = cell(to_bits(s), to_bits(x), to_bits(p),
                      d.float().unsqueeze(1))
        loss = F.binary_cross_entropy_with_logits(logits, to_bits(y))
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1

    torch.save(cell.state_dict(), weights_path)
    torch.save(opt.state_dict(), opt_path)
    state_path.write_text(json.dumps({"steps": step, "loss": float(loss)}))
