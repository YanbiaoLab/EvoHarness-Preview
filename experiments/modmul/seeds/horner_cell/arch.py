"""Architecture: fixed-width modulus-conditioned step cell (tiers 1-3 family).

MUTATION SURFACE — this file is where architecture changes belong.

The contract this family keeps: the SCHEDULE (a fixed double-and-add loop over
the operand's bits) is hand-coded, every TRANSITION is the learned cell. Never
replace cell() with arithmetic — that turns the model into a circuit.

Known ceiling: BITS is fixed, so this family cannot represent p beyond 16 bits
(tier 4 starts at 17). Widening it is a legitimate mutation; making it
width-GENERIC (see the limb_horner seed) is the bigger prize.
"""

from __future__ import annotations

import torch
from torch import nn

BITS = 16                     # state/modulus width: p < 2**16 on tiers 1-3

HIDDEN = 512
DEPTH = 2


def to_bits(v: torch.Tensor, width: int = BITS) -> torch.Tensor:
    """(N,) small non-negative ints -> (N, width) float bit vectors (LSB first)."""
    ar = torch.arange(width, device=v.device)
    return ((v.unsqueeze(1) >> ar) & 1).float()


class StepCell(nn.Module):
    """Learns ONE exact transition s' = (2s + d*x) mod p on bit vectors."""

    def __init__(self):
        super().__init__()
        layers: list[nn.Module] = []
        width = 3 * BITS + 1
        for _ in range(DEPTH):
            layers += [nn.Linear(width, HIDDEN), nn.GELU()]
            width = HIDDEN
        layers.append(nn.Linear(width, BITS))
        self.net = nn.Sequential(*layers)

    def forward(self, s, x, p, d):        # s/x/p: (N, BITS), d: (N, 1)
        return self.net(torch.cat([s, x, p, d], dim=1))


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
