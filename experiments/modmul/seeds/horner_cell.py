"""Seed candidate: modulus-conditioned step cell on a FIXED double-and-add
schedule (the brief §2 pattern: "the arithmetic is learned, the schedule is
not"). Three passes share ONE cell: reduce a, reduce b, then multiply the
residues by scanning the bits of ra with x = rb.

CONTRACT NOTES FOR MUTATION:
- The schedule (bit loops below) is hand-coded control flow — LEGAL. The
  transition itself must stay LEARNED: never replace cell() with arithmetic.
- Train-time label synthesis uses exact arithmetic — legal. The predict path
  uses no %, //, or big-int ops; int()/str() on small model outputs is legal.
- preprocess_a/b/p: base conversion of one's own argument is explicitly legal.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from modchallenge.interface.base_model import ModularMultiplicationModel

MANIFEST = {
    "entry_class": "model.EvolvedModel",
    "output_base": 10,
    "model_description": "modulus-conditioned MLP step cell learning "
                         "s' = (2s + d*x) mod p over 16-bit states, driven by "
                         "a fixed double-and-add schedule; ~560K params",
    "training_description": "cell trained at eval time on random exact step "
                            "tuples (fixed seed 0), AdamW; schedule hand-coded, "
                            "arithmetic learned",
}

BITS = 16                     # state/modulus width: p < 2**16 on tiers 1-3

# EDIT-REGION-BEGIN  (mutation surface: cell architecture / training recipe)
HIDDEN = 512
DEPTH = 2
LR = 1e-3
WD = 0.0
TRAIN_STEPS = 3000
BATCH = 4096
SEED = 0
# EDIT-REGION-END

# Task-fixed tier geometry (p bit ranges); not part of the mutation surface.
TIERS = ((1, 3), (4, 8), (9, 16))


def _sieve(limit: int) -> list[int]:
    flags = bytearray([1]) * limit
    flags[0] = flags[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if flags[i]:
            flags[i * i :: i] = bytearray(len(flags[i * i :: i]))
    return [i for i in range(2, limit) if flags[i]]


_PRIMES = _sieve(1 << BITS)
PRIMES_BY_TIER = [
    [p for p in _PRIMES if lo <= p.bit_length() <= hi] for lo, hi in TIERS
]


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _to_bits(v: torch.Tensor) -> torch.Tensor:
    """(N,) small non-negative ints -> (N, BITS) float bit vectors (LSB first).
    Values here are < 2**16, far from int64 overflow."""
    ar = torch.arange(BITS, device=v.device)
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


def _sample_batch(n: int, device):
    """Random exact step tuples, tier-balanced. TRAIN-time arithmetic (legal).
    All values < 2**16, so torch int64 ops are overflow-safe here."""
    tier = torch.randint(len(_PRIME_TENSORS), (n,), device=device)
    p = torch.empty(n, dtype=torch.long, device=device)
    for i, plist in enumerate(_PRIME_TENSORS):
        mask = tier == i
        k = int(mask.sum())
        if k:
            p[mask] = plist[torch.randint(plist.numel(), (k,), device=device)]
    s = (torch.rand(n, device=device) * p).long()
    x = (torch.rand(n, device=device) * p).long()
    d = torch.randint(2, (n,), device=device)
    y = (2 * s + d * x) % p
    return s, x, p, d, y


_PRIME_TENSORS: list[torch.Tensor] = []


def train(model_dir: str) -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)
    device = _pick_device()
    global _PRIME_TENSORS
    _PRIME_TENSORS = [torch.tensor(pl, device=device) for pl in PRIMES_BY_TIER]
    cell = StepCell().to(device)
    opt = torch.optim.AdamW(cell.parameters(), lr=LR, weight_decay=WD)
    warmup = min(200, TRAIN_STEPS)
    cell.train()
    for step in range(TRAIN_STEPS):
        for pg in opt.param_groups:
            pg["lr"] = LR * min(1.0, (step + 1) / warmup)
        s, x, p, d, y = _sample_batch(BATCH, device)
        logits = cell(_to_bits(s), _to_bits(x), _to_bits(p),
                      d.float().unsqueeze(1))
        loss = F.binary_cross_entropy_with_logits(logits, _to_bits(y))
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.save(cell.state_dict(), Path(model_dir) / "weights.pt")


class EvolvedModel(ModularMultiplicationModel):
    def load(self, model_dir: str) -> None:
        self.device = _pick_device()
        self.cell = StepCell().to(self.device)
        state = torch.load(Path(model_dir) / "weights.pt",
                           map_location=self.device)
        self.cell.load_state_dict(state)
        self.cell.eval()

    # Base conversion of one's OWN argument: explicitly permitted preprocessing.
    def preprocess_a(self, a: str) -> int:
        return int(a)

    def preprocess_b(self, b: str) -> int:
        return int(b)

    def preprocess_p(self, p: str) -> int:
        return int(p)

    @torch.no_grad()
    def _chain(self, value: int, x_bits: torch.Tensor,
               p_bits: torch.Tensor) -> torch.Tensor:
        """Fixed double-and-add schedule over the bits of `value` (MSB->LSB);
        every transition is the LEARNED cell. Returns final state bits."""
        s = torch.zeros(1, BITS, device=self.device)
        width = max(value.bit_length(), 1)
        for i in range(width - 1, -1, -1):
            d = torch.tensor([[float((value >> i) & 1)]], device=self.device)
            logits = self.cell(s, x_bits, p_bits, d)
            s = (logits > 0).float()
        return s

    @staticmethod
    def _bits_to_int(bits: torch.Tensor) -> int:
        return int(sum(int(bits[0, i].item()) << i for i in range(BITS)))

    def predict_digits(self, a_enc, b_enc, p_enc) -> list[int]:
        p_bits = _to_bits(torch.tensor([p_enc], device=self.device))
        one = _to_bits(torch.tensor([1], device=self.device))
        ra = self._bits_to_int(self._chain(a_enc, one, p_bits))   # a mod p
        rb_bits = self._chain(b_enc, one, p_bits)                 # b mod p
        # (ra * rb) mod p: scan bits of ra with x = rb
        out = self._chain(ra, rb_bits, p_bits)
        value = self._bits_to_int(out)
        return [ord(c) - 48 for c in str(value)]                  # MSB-first
