"""Training recipe for the width-generic Horner cell.

MUTATION SURFACE — curriculum, data distribution, optimizer, schedule.

Three things here are deliberate, per the research brief:
  * WIDTH CURRICULUM with warm-start: the cell is trained at 8 bits, then
    progressively wider. Because the architecture is width-generic, the same
    weights keep improving instead of being re-learned per width. This is the
    mechanism that is supposed to reach tier 9-10 widths.
  * SPARSE / power-of-two-adjacent strata: the published failure mode of this
    family is drift on Fermat-like operands at the top of the trained width
    range. They are sampled explicitly rather than hoped for.
  * MODULI ARE NOT REQUIRED TO BE PRIME: the transition is defined for any
    modulus, so training on arbitrary moduli is strictly more data and needs
    no primality machinery.

Harness contract: `train(model_dir)` is RESUMABLE (picks up weights + step
count from model_dir) and must respect MODMUL_TRAIN_SECONDS.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from arch import RADIX_BITS, HornerCell, pick_device

LR = 1e-3
WD = 0.0
SEED = 0
WARMUP = 300
# The comment here used to read "upper bound; the time budget is what binds",
# and it was wrong. Run modmul_r8's candidate 06dde31b found it in two
# generations by reading the `training_skipped` diagnostic: an inherited
# lineage reaches two million steps while later ASHA rungs still have their
# whole wall-clock allowance, so the rungs report training_skipped and the
# candidate stops improving with compute left on the table. Raising the cap
# took tier 9 from 80% to 98% at unchanged wall clock, h90 8 -> 9. Not a
# ceiling that binds, then -- a ceiling that hid the one that does.
TOTAL_STEPS = 3_000_000

# Width curriculum. Each entry is a state width in bits; the sampler unlocks
# them progressively. Tier geometry for reference: t1 needs 2-3 (the fixed
# primes 2,3,5,7 — skipping these forfeits the easiest tier), t3 needs 16,
# t5 needs 64, t7 needs 256, t10 needs 2048. 2112 exists for one reason:
# inference buckets a prime's width up to the next multiple of 64 with at
# least 4 bits of headroom, so the very top of tier 10 (2045-2048 bit
# primes) runs in a 2112-bit register.
WIDTHS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512,
          768, 1024, 1536, 2048, 2112)
UNLOCK_EVERY = 2500                # steps before the next width joins the mix
NEWEST_SHARE = 0.5                 # probability mass on the newest width
SPARSE_SHARE = 0.15                # power-of-two-adjacent / sparse operands
# Share of samples whose PRIME is narrower than the state register (delta =
# width - prime_bits drawn from [4, 67]). Community measurement on a fork of
# this family (Hongyue Lei, mod-arith-k2, 2026-07): a cell trained only at
# delta=0 collapses to 2-30% accuracy at delta in {1,2}, is marginal at 3,
# fine at >=4, and identical across 16..64 -- and robustness tracks the
# TRAINING distribution, not the loop. Training only delta=0 is what killed
# all four width-padding candidates in run r12 gen 1: they padded to the
# next power of two for batching (the right idea) and the cell had never
# seen a padded register. This stratum is the key that unlocks padded
# batching; inference stays inside the [4, 67] band it covers.
DELTA_SHARE = 0.5
TOKEN_BUDGET = 32_768              # batch = TOKEN_BUDGET // width, clamped
MIN_BATCH, MAX_BATCH = 16, 512


def _bit_tensor(values: list[int], width: int, device) -> torch.Tensor:
    """list[int] -> (N, width) float bit tensor, LSB first. Byte-level so a
    2048-bit sample costs a memcpy instead of 2048 Python shifts."""
    nbytes = (width + 7) // 8
    blob = bytearray()
    for value in values:
        blob += value.to_bytes(nbytes, "little")
    packed = torch.frombuffer(blob, dtype=torch.uint8).view(len(values), nbytes)
    shifts = torch.arange(8, dtype=torch.uint8)
    bits = ((packed.unsqueeze(-1) >> shifts) & 1).view(len(values), -1)
    return bits[:, :width].float().to(device)


def _sparse_value(rng: random.Random, width: int) -> int:
    """Power-of-two-adjacent / very sparse values — the known drift trigger
    (the published failure mode of this family). Always in [0, 2^width)."""
    limit = 1 << width
    kind = rng.randrange(4)
    if kind == 0:                                   # a single set bit
        value = 1 << rng.randrange(width)
    elif kind == 1:                                 # all ones below a cut
        value = (1 << rng.randrange(1, width + 1)) - 1
    elif kind == 2:                                 # power of two +/- 1
        value = (1 << rng.randrange(width)) + rng.choice((-1, 0, 1))
    else:                                           # a couple of set bits
        value = (1 << rng.randrange(width)) | (1 << rng.randrange(width))
    return max(0, value) % limit


def sample_width(step: int, rng: random.Random) -> int:
    unlocked = min(len(WIDTHS), 1 + step // UNLOCK_EVERY)
    if unlocked > 1 and rng.random() < NEWEST_SHARE:
        return WIDTHS[unlocked - 1]
    return WIDTHS[rng.randrange(unlocked)]


def sample_batch(width: int, count: int, rng: random.Random, device):
    """Exact transition tuples for `s' = (2^k*s + d*x) mod p` at this width.

    Exact integer arithmetic is legal at TRAIN time — this synthesizes labels,
    it is not on the inference path.
    """
    radix = 1 << RADIX_BITS
    moduli, states, multiplicands, digits, targets = [], [], [], [], []
    for _ in range(count):
        if rng.random() < DELTA_SHARE:
            # Prime narrower than the register: the padded-batching case.
            bits = max(2, width - rng.randrange(4, 68))
        else:
            bits = width                                    # exact fill
        p = rng.getrandbits(bits) | (1 << (bits - 1))
        if p < 2:
            p = 2
        if rng.random() < SPARSE_SHARE:
            s = _sparse_value(rng, width) % p
            x = _sparse_value(rng, width) % p
        else:
            s = rng.randrange(p)
            x = rng.randrange(p)
        d = rng.randrange(radix)
        moduli.append(p)
        states.append(s)
        multiplicands.append(x)
        digits.append(d)
        targets.append((s * radix + d * x) % p)
    return (
        _bit_tensor(states, width, device),
        _bit_tensor(multiplicands, width, device),
        _bit_tensor(moduli, width, device),
        _bit_tensor(digits, RADIX_BITS, device),
        _bit_tensor(targets, width, device),
    )


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
    cell = HornerCell().to(device)
    if weights_path.exists():
        # Load per TENSOR, not all-or-nothing. RADIX_BITS only enters
        # in_features = 3k + 4, so raising it reshapes exactly one matrix
        # (embed's first Linear) — 896 of 91,841 parameters. A strict load
        # rejects the whole checkpoint over that 1% and the candidate starts
        # from noise, which makes the single highest-value mutation the most
        # expensive one to try. Keep every tensor whose name and shape still
        # match; leave the rest at their fresh initialisation.
        incoming = torch.load(weights_path, map_location=device)
        current = cell.state_dict()
        usable = {
            name: tensor
            for name, tensor in incoming.items()
            if name in current and current[name].shape == tensor.shape
        }
        current.update(usable)
        cell.load_state_dict(current)
        if len(usable) < len(current):
            kept = sum(t.numel() for t in usable.values())
            total = sum(t.numel() for t in current.values())
            print(f"[train] partial warm start: kept {len(usable)}/{len(current)} "
                  f"tensors, {kept}/{total} params ({kept / total:.1%})",
                  flush=True)
    opt = torch.optim.AdamW(cell.parameters(), lr=LR, weight_decay=WD)
    if opt_path.exists():
        try:
            opt.load_state_dict(torch.load(opt_path, map_location=device))
            # Adam's exp_avg / exp_avg_sq are keyed by parameter POSITION and
            # carry that parameter's shape. load_state_dict copies them
            # without checking, so a checkpoint that predates a reshape loads
            # cleanly and then fails at the first step, deep inside
            # _multi_tensor_adam:
            #     RuntimeError: The size of tensor a (7) must match the size
            #     of tensor b (16) at non-singleton dimension 1
            # 7 and 16 are in_features for RADIX_BITS 1 and 4.
            for param, entry in opt.state.items():
                for value in entry.values():
                    if (torch.is_tensor(value) and value.dim()
                            and value.shape != param.shape):
                        raise ValueError("optimizer state predates a reshape")
        except Exception:
            # Clear AND remove. The previous version only rebound the local
            # path, so the stale file stayed on disk: the first rung skipped
            # it, saved elsewhere, and the SECOND rung -- by which point the
            # weights matched and the guard no longer fired -- loaded it and
            # died. Every RADIX_BITS candidate in runs r7, r10 and r11 was
            # killed this way at R1, which is why the highest-value lever in
            # this task had never once been evaluated.
            opt.state.clear()
            opt_path.unlink(missing_ok=True)

    cell.train()
    step, loss = done, torch.tensor(0.0)
    last_tick = time.monotonic()
    while step < TOTAL_STEPS and time.monotonic() < deadline:
        # The grader freezes training (SIGSTOP) while a timed inference holds
        # the card alone. Frozen seconds are the harness's spend, not this
        # candidate's: a step normally takes well under a second, so a gap
        # this size can only be an external stop -- push the deadline out by
        # exactly that gap instead of silently billing it to training.
        now = time.monotonic()
        if now - last_tick > 5.0:
            deadline += now - last_tick
        last_tick = now
        for group in opt.param_groups:
            group["lr"] = LR * min(1.0, (step + 1) / WARMUP)
        width = sample_width(step, rng)
        count = max(MIN_BATCH, min(MAX_BATCH, TOKEN_BUDGET // width))
        s, x, p, digit, y = sample_batch(width, count, rng, device)
        logits = cell(s, x, p, digit)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cell.parameters(), 1.0)
        opt.step()
        step += 1

    torch.save(cell.state_dict(), weights_path)
    torch.save(opt.state_dict(), opt_path)
    state_path.write_text(json.dumps({
        "steps": step,
        "loss": float(loss),
        "max_width": WIDTHS[min(len(WIDTHS), 1 + step // UNLOCK_EVERY) - 1],
    }))
