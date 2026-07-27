"""Width-generic neural Horner cell — single-file genome for ShinkaEvolve.

This is `limb_horner` (seeds/limb_horner/{arch,model,train}.py) merged into one
file, because ShinkaEvolve evolves a single program. The merge is verbatim; the
only thing added is the EVOLVE-BLOCK partition, and that partition is the point
of the port:

    outside the blocks   the compliance contract — MANIFEST, EvolvedModel, the
                         three-pass schedule. Mutation cannot reach it, so
                         `a % p` cannot come back into Python no matter what
                         the search decides to try.
    inside block one     the architecture: radix, widths, the cell.
    inside block two     the training recipe: curriculum, sampling, optimizer.

Legality, spelled out because this family sits closest to the line:
  * `output_base` is 2 and the emitted digits ARE the network's output bits.
    Nothing converts, corrects or post-processes them — the harness decoder
    turns them into the answer. Garbage bits give a garbage answer, which is
    exactly what principle 1 ("the emitted digits must materially determine
    the answer") asks for.
  * The three passes eat the ORIGINAL a, b, p. No `a % p` anywhere: reducing
    the full-width operands is done BY THE NETWORK, which is the point.
  * preprocess_a/b convert their own argument to base-2^k digits and
    preprocess_p to bits — per-argument base conversion, explicitly allowed.
  * The forward path contains no arithmetic on those values at all: the third
    pass reads its digits by SLICING the bit vector the network produced.
  * The schedule (three passes, one digit per step) is a fixed, feedback-free
    encoder — permitted; every transition is the learned cell.
  * Exact integer arithmetic appears only inside `sample_batch`, which
    synthesises TRAINING LABELS. It is not on the inference path.

Harness contract: `train(model_dir)` is RESUMABLE (picks up weights + step
count from model_dir) and must respect MODMUL_TRAIN_SECONDS. The grader falls
back to `model.py` when a candidate has no `train.py`, so one file is enough.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from modchallenge.interface.base_model import ModularMultiplicationModel


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# EVOLVE-BLOCK-START
# ===========================================================================
# ARCHITECTURE
#
# The contract below this block imports four names from here by reference:
# RADIX_BITS, MAX_WIDTH, HornerCell, and a cell callable as
# cell(s, x, p, digit) -> (N, W) logits. Renaming or removing any of them
# breaks a part of the file you cannot see and the candidate scores zero.
# Everything else in this block is free.
#
# Why this shape, in one paragraph: the step `s' = (2^k*s + d*x) mod p` needs
# carry/borrow information to travel across the whole width of the state.
# Doing that with a dense layer over the whole state ties the parameters to
# one width. Doing it with a sequential loop over limbs costs O(width)
# sequential steps and blows the inference budget at tiers 9-10. So the carry
# travels through a LEARNED ASSOCIATIVE SCAN (Hillis-Steele, depth
# log2(width)) whose operator is SHARED across all levels and all positions.
# Nothing in the module knows the width:
#
#   * no position embeddings (they would not exist for unseen widths),
#   * one scan operator reused at every level (an unseen width just means
#     more levels of the same learned operator),
#   * per-position features are a fixed LOCAL WINDOW of (s, x, p).
#
# That is what lets a cell trained at 16-64 bits run at 2048 bits. Measured
# before this seed was committed: 5 minutes of training on widths 8/12/16
# only, then evaluated zero-shot on the transition —
#
#     width   8  16  24  32  64  128  256
#     exact  1.0 .99 .98 .97 .83  .43  .12
#
# THE SCAN MUST BE BIDIRECTIONAL — this cost a day to find, do not "simplify"
# it away. Carries travel LSB->MSB, but the mod-p reduction decision ("is the
# intermediate >= p?") is determined by the HIGH bits and has to reach every
# low bit. With an upward-only scan the cell plateaus at bit-accuracy 0.80 /
# exact 0.21 and never moves; adding the downward scan takes it to exact 1.00
# on the same budget.
# ===========================================================================

# Horner radix: the outer loop consumes RADIX_BITS bits of the operand per
# step, so inference costs operand_bits/RADIX_BITS steps. This is the single
# biggest inference-time lever at tiers 9-10 (4096-bit operands) AND a real
# trade-off: with k=1 the intermediate 2s + d*x is under 3p (the reduction is
# a 0/1/2 choice), with k=4 it is under 32p and measurably harder to learn
# (bit-accuracy 0.73 vs 0.80 under the same budget in the pre-commit sweep).
# k=1 is the proven setting; raising it is a legitimate, load-bearing mutation
# for the higher tiers — but pay for it with training.
RADIX_BITS = 1

D_MODEL = 64
HIDDEN = 128
ROUNDS = 3                 # learned refinement rounds per Horner step
MAX_WIDTH = 4096           # honest 0 beyond this


def window(t: torch.Tensor, span: int) -> torch.Tensor:
    """(N, W) -> (N, W, span+1) stack of t[i], t[i-1], ..., t[i-span].

    Index 0 is the LSB, so a shift toward higher indices is a multiplication
    by a power of two. Providing the window does NOT impose the shift — the
    cell decides what to do with the neighbours it can see.
    """
    parts = [t]
    for offset in range(1, span + 1):
        parts.append(F.pad(t, (offset, 0))[:, : t.shape[1]])
    return torch.stack(parts, dim=-1)


def mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 2):
        layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.GELU()]
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    return nn.Sequential(*layers)


class HornerCell(nn.Module):
    """One learned transition s' = (2^k*s + d*x) mod p over bit vectors."""

    def __init__(self):
        super().__init__()
        k = RADIX_BITS
        # local features per bit position: window of s and x over the radix
        # span, the two lowest bits of p at that position, and the digit.
        self.in_features = (k + 1) + (k + 1) + 2 + k
        self.embed = mlp([self.in_features, HIDDEN, D_MODEL])
        # ONE operator per direction, reused at every scan level — this is the
        # width-generalization hinge. Do not give either a level index.
        self.up = mlp([2 * D_MODEL, HIDDEN, D_MODEL])      # carries, LSB->MSB
        self.down = mlp([2 * D_MODEL, HIDDEN, D_MODEL])    # reduction, MSB->LSB
        self.mix = mlp([3 * D_MODEL, HIDDEN, D_MODEL])
        self.head = nn.Linear(D_MODEL, 1)

    def scan(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Bidirectional Hillis-Steele scan, depth ceil(log2(W)) each way.

        Upward alone is not enough — see the block header."""
        width = h.shape[1]
        up = down = h
        offset = 1
        while offset < width:
            lower = F.pad(up, (0, 0, offset, 0))[:, :width]
            higher = F.pad(down, (0, 0, 0, offset))[:, offset:]
            up = self.up(torch.cat([lower, up], dim=-1))
            down = self.down(torch.cat([higher, down], dim=-1))
            offset *= 2
        return up, down

    def forward(
        self,
        s: torch.Tensor,       # (N, W) bits, LSB first
        x: torch.Tensor,       # (N, W) bits
        p: torch.Tensor,       # (N, W) bits
        digit: torch.Tensor,   # (N, RADIX_BITS) bits of the operand digit
    ) -> torch.Tensor:         # (N, W) logits for the next state
        width = s.shape[1]
        feats = torch.cat(
            [
                window(s, RADIX_BITS),
                window(x, RADIX_BITS),
                window(p, 1),
                digit.unsqueeze(1).expand(-1, width, -1),
            ],
            dim=-1,
        )
        h = self.embed(feats)
        for _ in range(ROUNDS):
            upward, downward = self.scan(h)
            h = self.mix(torch.cat([h, upward, downward], dim=-1))
        return self.head(h).squeeze(-1)


# EVOLVE-BLOCK-END


# ===========================================================================
# COMPLIANCE CONTRACT — frozen, outside every evolve block.
#
# This is the surface the organisers judge. It states what the submission
# does and implements the three-pass schedule that eats the ORIGINAL a, b, p.
# Keeping it out of the mutable region is the whole reason this task is on
# ShinkaEvolve rather than on a whole-workspace mutator: no amount of search
# can put `a % p` back into Python from here.
# ===========================================================================

MANIFEST = {
    "entry_class": "model.EvolvedModel",
    "output_base": 2,
    "framework": "pytorch",
    "model_description": (
        "width-generic modulus-conditioned Horner cell (~100K params): per-bit "
        "local windows of (state, multiplicand, modulus) plus a learned "
        "associative scan (Hillis-Steele, depth log2(width), one shared "
        "operator) resolve carries, so the same weights run at any state "
        "width. Three passes with shared weights: reduce a, reduce b, multiply "
        "the residues. State width is sized to the prime's bit length at "
        "inference. Emits the state bits directly as base-2 digits."
    ),
    "training_description": (
        "trained at eval time from random init on exact transition tuples "
        "s' = (2^k*s + d*x) mod p over a width curriculum (8 -> 2048 bits, "
        "warm-started across widths) with a power-of-two-adjacent stratum; "
        "BCE on the target bits, AdamW, fixed seed 0. Labels come from exact "
        "integer arithmetic at TRAIN time only; the inference path has none."
    ),
}


class EvolvedModel(ModularMultiplicationModel):
    def load(self, model_dir: str) -> None:
        self.device = pick_device()
        self.cell = HornerCell().to(self.device)
        state = torch.load(Path(model_dir) / "weights.pt",
                           map_location=self.device)
        self.cell.load_state_dict(state)
        self.cell.eval()

    def max_batch_size(self) -> int:
        return 64

    # -- per-argument preprocessing (each hook sees only its own argument) --

    @staticmethod
    def _radix_digits(text: str) -> tuple[int, ...]:
        """Decimal string -> base-2^k digits, MSB first. Base conversion of
        one's own argument is explicitly permitted preprocessing."""
        value = int(text)
        if value == 0:
            return (0,)
        mask = (1 << RADIX_BITS) - 1
        digits = []
        while value:
            digits.append(value & mask)
            value >>= RADIX_BITS
        return tuple(reversed(digits))

    def preprocess_a(self, a: str):
        return self._radix_digits(a)

    def preprocess_b(self, b: str):
        return self._radix_digits(b)

    def preprocess_p(self, p: str):
        value = int(p)
        width = max(value.bit_length(), 2)
        return tuple((value >> i) & 1 for i in range(width)), width

    # -- inference ----------------------------------------------------------

    def _digit_bits(self, digit: int) -> list[float]:
        return [float((digit >> i) & 1) for i in range(RADIX_BITS)]

    @torch.no_grad()
    def _run_pass(
        self,
        digit_rows: torch.Tensor,     # (N, T, RADIX_BITS) MSB-first in T
        x_bits: torch.Tensor,         # (N, W)
        p_bits: torch.Tensor,         # (N, W)
    ) -> torch.Tensor:
        s = torch.zeros_like(p_bits)
        for t in range(digit_rows.shape[1]):
            logits = self.cell(s, x_bits, p_bits, digit_rows[:, t])
            s = (logits > 0).float()
        return s

    def _pack_digits(self, digit_lists: list[tuple[int, ...]]) -> torch.Tensor:
        """Left-pad with zero digits: a leading zero digit is an exact no-op
        for Horner (state stays 0), so padding cannot change an answer."""
        length = max(len(d) for d in digit_lists)
        rows = [
            [self._digit_bits(0)] * (length - len(digits))
            + [self._digit_bits(d) for d in digits]
            for digits in digit_lists
        ]
        return torch.tensor(rows, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def _solve_group(self, batch: list[tuple]) -> list[list[int]]:
        """Every item in `batch` shares one state width."""
        p_bits = torch.tensor(
            [list(p_enc[0]) for _, _, p_enc in batch],
            dtype=torch.float32, device=self.device,
        )
        width = p_bits.shape[1]
        ones = torch.zeros_like(p_bits)
        ones[:, 0] = 1.0

        a_digits = self._pack_digits([a for a, _, _ in batch])
        b_digits = self._pack_digits([b for _, b, _ in batch])
        ra = self._run_pass(a_digits, ones, p_bits)      # a mod p
        rb = self._run_pass(b_digits, ones, p_bits)      # b mod p

        # Digits of ra come from SLICING the network's own bit vector — no
        # arithmetic, and the residue never leaves tensor form.
        chunks = (width + RADIX_BITS - 1) // RADIX_BITS
        padded = torch.nn.functional.pad(
            ra, (0, chunks * RADIX_BITS - width)
        ).view(len(batch), chunks, RADIX_BITS)
        ra_digits = torch.flip(padded, dims=(1,))        # MSB-first in time

        out = self._run_pass(ra_digits, rb, p_bits)
        bits = out.to(torch.int64).tolist()
        return [list(reversed(row)) for row in bits]     # LSB -> MSB-first

    def predict_digits(self, a_enc, b_enc, p_enc) -> list[int]:
        return self.predict_digits_batch([(a_enc, b_enc, p_enc)])[0]

    def predict_digits_batch(self, inputs) -> list[list[int]]:
        results: list[list[int]] = [[0]] * len(inputs)
        groups: dict[int, list[int]] = defaultdict(list)
        for index, (_a, _b, p_enc) in enumerate(inputs):
            width = p_enc[1]
            if width > MAX_WIDTH:
                continue                      # honest 0 beyond the range
            groups[width].append(index)
        for indices in groups.values():
            batch = [inputs[i] for i in indices]
            for index, digits in zip(indices, self._solve_group(batch)):
                results[index] = digits
        return results


# EVOLVE-BLOCK-START
# ===========================================================================
# TRAINING RECIPE — curriculum, data distribution, optimizer, schedule.
#
# The harness calls `train(model_dir)` and nothing else from this block, so
# that name must survive. It is RESUMABLE by contract: continue from whatever
# weights are already in model_dir and respect MODMUL_TRAIN_SECONDS.
#
# Three things here are deliberate:
#   * WIDTH CURRICULUM with warm-start: the cell is trained at 8 bits, then
#     progressively wider. Because the architecture is width-generic, the same
#     weights keep improving instead of being re-learned per width. This is
#     the mechanism that is supposed to reach tier 9-10 widths.
#   * SPARSE / power-of-two-adjacent strata: the published failure mode of
#     this family is drift on Fermat-like operands at the top of the trained
#     width range. They are sampled explicitly rather than hoped for.
#   * MODULI ARE NOT REQUIRED TO BE PRIME: the transition is defined for any
#     modulus, so training on arbitrary moduli is strictly more data and needs
#     no primality machinery.
# ===========================================================================

LR = 1e-3
WD = 0.0
SEED = 0
WARMUP = 300
TOTAL_STEPS = 2_000_000            # upper bound; the time budget is what binds

# Width curriculum. Each entry is a state width in bits; the sampler unlocks
# them progressively. Tier geometry for reference: t1 needs 2-3 (the fixed
# primes 2,3,5,7 — skipping these forfeits the easiest tier), t3 needs 16,
# t5 needs 64, t7 needs 256, t10 needs 2048.
WIDTHS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512,
          768, 1024, 1536, 2048)
UNLOCK_EVERY = 2500                # steps before the next width joins the mix
NEWEST_SHARE = 0.5                 # probability mass on the newest width
SPARSE_SHARE = 0.15                # power-of-two-adjacent / sparse operands
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
        p = rng.getrandbits(width) | (1 << (width - 1))     # exact width
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
            # The optimizer state is keyed by parameter position, so it no
            # longer lines up once the shapes moved; start it fresh.
            opt_path = Path(str(opt_path) + ".stale")
    opt = torch.optim.AdamW(cell.parameters(), lr=LR, weight_decay=WD)
    if opt_path.exists():
        try:
            opt.load_state_dict(torch.load(opt_path, map_location=device))
        except Exception:
            pass

    cell.train()
    step, loss = done, torch.tensor(0.0)
    while step < TOTAL_STEPS and time.monotonic() < deadline:
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


# EVOLVE-BLOCK-END
