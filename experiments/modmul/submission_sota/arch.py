"""Architecture: width-GENERIC Horner cell (the tier-10 route).

MUTATION SURFACE — architecture. This is the family meant to climb past tier 3.

Why this shape, in one paragraph: the step `s' = (2^k*s + d*x) mod p` needs
carry/borrow information to travel across the whole width of the state. Doing
that with a dense layer over the whole state ties the parameters to one width
(the horner_cell family's ceiling, tier 3). Doing it with a sequential loop
over limbs costs O(width) sequential steps and blows the 5-minute inference
budget at tier 9-10. So the carry travels through a LEARNED ASSOCIATIVE SCAN
(Hillis-Steele, depth log2(width)) whose operator is SHARED across all levels
and all positions. Nothing in the module knows the width:

  * no position embeddings (they would not exist for unseen widths),
  * one scan operator reused at every level (an unseen width just means more
    levels of the same learned operator),
  * per-position features are a fixed LOCAL WINDOW of (s, x, p).

That is what lets a cell trained at 16-64 bits be run at 2048 bits. Measured
on a laptop before this seed was committed: 5 minutes of training on widths
8/12/16 only, then evaluated zero-shot on the transition —

    width   8  16  24  32  64  128  256
    exact  1.0 .99 .98 .97 .83  .43  .12

so the transfer is real, and the curriculum in train.py is there to push the
frontier out. Note what the Horner loop demands of this number: a 2048-bit
operand takes ~4096 steps, so end-to-end correctness needs per-step exactness
of about 1 - 1e-5. Getting from .99 to .99999 is the actual work.

THE SCAN MUST BE BIDIRECTIONAL — this cost a day to find, do not "simplify" it
away. Carries travel LSB->MSB, but the mod-p reduction decision ("is the
intermediate >= p?") is determined by the HIGH bits and has to reach every low
bit. With an upward-only scan the cell plateaus at bit-accuracy 0.80 /
exact 0.21 and never moves; adding the downward scan takes it to exact 1.00 on
the same budget.

The output projection intentionally has no scalar bias. A biased one-bit head
can make a randomized cell emit the same all-zero register at every position:
that has accidental accuracy on small moduli despite no learned arithmetic.
The learned position-dependent representation, rather than a global output
default, must determine every emitted state bit.

Legality: the schedule (which slot feeds which cell input, how many scan
levels) is hand-coded control flow. Every value-producing step is the learned
cell — no adder, no comparator, no conditional subtract is written down.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

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

# The widest state this model will attempt; wider primes get an honest 0.
#
# 2048 is the scored range: tier 10's primes are 1025-2048 bits and no scored
# tier goes above it. It is also where the width curriculum in train.py stops.
#
# It used to say 4096, and that costs the run everything. The DIAGNOSTIC tier
# spans the whole benchmark -- primes from 8 bits to 8192 -- and it is not
# scored, but it runs FIRST and it spends the same shared clock. Profiled on
# this seed: its ten problems at width 4096 take 219.7 seconds, 78% of that
# tier's whole cost, and the budget is 300 seconds for everything. Tier 0 then
# finishes at ~280-330s and tiers 1 through 10 never start. Measured h90: 0.
#
# What this trades, stated plainly: the model answers those ten problems
# CORRECTLY -- 10/10, generalising past the widths it was trained on -- and
# declining them gives up ten right answers that are worth no points, to buy
# tier 9 and tier 10, which are worth two levels of the ranking key. It is a
# deliberate allocation of a shared budget, not a correctness fix, and it
# belongs in the submission's model description rather than in a footnote.
MAX_WIDTH = 2048


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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

        # No global output bias: a randomized scalar bias otherwise creates a
        # parameter-insensitive constant-register fallback on small moduli.
        self.head = nn.Linear(D_MODEL, 1, bias=False)

    def scan(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Bidirectional Hillis-Steele scan, depth ceil(log2(W)) each way.

        Upward alone is not enough — see the module docstring.
        """
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
