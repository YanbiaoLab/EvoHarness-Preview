"""Inference contract for the width-generic Horner family.

CONTRACT FILE — architecture goes in arch.py, recipe in train.py.

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
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch

from arch import MAX_WIDTH, RADIX_BITS, HornerCell, pick_device
from modchallenge.interface.base_model import ModularMultiplicationModel

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
