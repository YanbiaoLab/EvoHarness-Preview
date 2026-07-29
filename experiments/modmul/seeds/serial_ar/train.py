"""Training recipe for the serial AR transformer.

MUTATION SURFACE — data distribution, curriculum, loss weights, optimizer.

Harness contract: `train(model_dir)` is RESUMABLE and must respect the
MODMUL_TRAIN_SECONDS wall-clock budget (see the horner_cell seed's train.py
for the same pattern). Exact integer arithmetic is legal here — labels are
synthesized at TRAIN time; only inference must be free of it.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from arch import BASE, W_ANS, W_IN, W_Q, W_RAW, SerialNet, pick_device

LR = 5e-4
WD = 0.1
AUX_RAW, AUX_Q = 1.0, 1.0        # loss weight on the intermediate segments
BATCH = 256
SEED = 0
WARMUP = 300
# The comment here used to read "upper bound; the time budget is what binds",
# and it was false in both of the other seeds. limb_horner silently stopped
# improving once it reached the cap, which run modmul_r8's 06dde31b found by
# reading the training_skipped diagnostic; horner_cell's cache sat exactly ON
# its cap, so training became a no-op and then a crash. This one has not bound
# yet -- 16,724 steps against 300,000 -- but lineages in this project already
# reach 740,958 and 1,200,000, so it would have.
TOTAL_STEPS = 2_000_000

# Task-fixed: (p_bits_lo, p_bits_hi, operand_bits) per tier. Deliberately NOT
# a mutation target — a candidate must not train only on the easy tiers.
TIERS = ((1, 3, 32), (4, 8, 48), (9, 16, 64))


def _sieve(limit: int) -> list[int]:
    flags = bytearray([1]) * limit
    flags[0] = flags[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if flags[i]:
            flags[i * i:: i] = bytearray(len(flags[i * i:: i]))
    return [i for i in range(2, limit) if flags[i]]


_PRIMES = _sieve(1 << 16)
PRIMES_BY_TIER = [
    [p for p in _PRIMES if lo <= p.bit_length() <= hi] for lo, hi, _ in TIERS
]


def digits_lsb(value: int, width: int) -> list[int]:
    """Python-int -> fixed-width LSB-first digits. Stays in Python ints on
    purpose: tier-3 products reach 128 bits and torch int64 wraps silently."""
    out = []
    for _ in range(width):
        out.append(value % BASE)
        value //= BASE
    return out


def synthesize(n: int, rng: random.Random, device) -> dict[str, torch.Tensor]:
    rows = {k: [] for k in ("x", "y", "p", "raw", "q", "ans")}
    for _ in range(n):
        t = rng.randrange(len(TIERS))
        p = rng.choice(PRIMES_BY_TIER[t])
        op_bits = TIERS[t][2]
        a, b = rng.getrandbits(op_bits), rng.getrandbits(op_bits)
        raw = a * b
        rows["x"].append(digits_lsb(a, W_IN))
        rows["y"].append(digits_lsb(b, W_IN))
        rows["p"].append(digits_lsb(p, W_IN))
        rows["raw"].append(digits_lsb(raw, W_RAW))
        rows["q"].append(digits_lsb(raw // p, W_Q))
        rows["ans"].append(digits_lsb(raw % p, W_ANS))
    return {k: torch.tensor(v, dtype=torch.long, device=device)
            for k, v in rows.items()}


def train(model_dir: str) -> None:
    budget = float(os.environ.get("MODMUL_TRAIN_SECONDS", "600"))
    deadline = time.monotonic() + budget * 0.9
    directory = Path(model_dir)
    state_path = directory / "train_state.json"
    weights_path = directory / "weights.pt"
    opt_path = directory / "optimizer.pt"

    done = 0
    if state_path.exists():
        done = int(json.loads(state_path.read_text()).get("steps", 0))
    torch.manual_seed(SEED + done)
    rng = random.Random(SEED + done)

    device = pick_device()
    net = SerialNet().to(device)
    if weights_path.exists():
        net.load_state_dict(torch.load(weights_path, map_location=device))
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
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

    lw = torch.tensor([AUX_RAW] * W_RAW + [AUX_Q] * W_Q + [1.0] * W_ANS,
                      dtype=torch.float32, device=device)
    net.train()
    step, loss = done, torch.tensor(0.0)
    while step < TOTAL_STEPS and time.monotonic() < deadline:
        for group in opt.param_groups:
            group["lr"] = LR * min(1.0, (step + 1) / WARMUP)
        batch = synthesize(BATCH, rng, device)
        logits = net.forward_teacher(batch)
        tgt = net.targets(batch)
        ce = F.cross_entropy(logits.reshape(-1, BASE), tgt.reshape(-1),
                             reduction="none").view(tgt.shape)
        loss = (ce * lw).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1

    torch.save(net.state_dict(), weights_path)
    torch.save(opt.state_dict(), opt_path)
    state_path.write_text(json.dumps({"steps": step, "loss": float(loss)}))
