"""Inference contract for the serial AR family.

CONTRACT FILE — architecture goes in arch.py, recipe in train.py.

Legality notes:
  * The predict path is purely neural: no %, //, or big-int arithmetic on the
    answer path. The intermediates (raw product, quotient) are generated
    INSIDE the model and never leave it — the harness decoder sees only the
    answer digits, which is what the rules require.
  * The official decoder expects MSB-first digits; the internal representation
    is LSB-first (it matches the carry structure). The flip happens ONLY at
    the predict_digits boundary.
  * preprocess_a/b/p each read only their own argument (isolation-checked).
"""

from __future__ import annotations

from pathlib import Path

import torch

from arch import W_ANS, W_IN, SerialNet, pick_device
from modchallenge.interface.base_model import ModularMultiplicationModel

MANIFEST = {
    "entry_class": "model.EvolvedModel",
    "output_base": 10,
    "framework": "pytorch",
    "model_description": "serial AR transformer emitting raw->quotient->answer "
                         "digit chains over fixed decimal windows; the answer "
                         "is conditioned on the generated intermediates",
    "training_description": "trained at eval time on synthesized tier-1..3 "
                            "data, fixed seed 0, AdamW + warmup; labels come "
                            "from exact integer arithmetic at TRAIN time only",
}


class EvolvedModel(ModularMultiplicationModel):
    def load(self, model_dir: str) -> None:
        self.device = pick_device()
        self.net = SerialNet().to(self.device)
        state = torch.load(Path(model_dir) / "weights.pt",
                           map_location=self.device)
        self.net.load_state_dict(state)
        self.net.eval()

    # Per-argument tokenisation ONLY (decimal string -> LSB digit tensor);
    # zero arithmetic, not even int() — chars to digits via ord.
    def _dig(self, s: str) -> torch.Tensor:
        d = [ord(c) - 48 for c in reversed(s)][:W_IN]
        d += [0] * (W_IN - len(d))
        return torch.tensor([d], dtype=torch.long, device=self.device)

    def max_batch_size(self) -> int:
        # The official timer is checked between batches, not per problem, so
        # batches are the unit of cost. The interface default is 1, which
        # runs an official tier's 100 problems as 100 sequential forward
        # passes; this seed never set it. 128 covers a full tier in one.
        return 128

    def preprocess_a(self, a: str):
        return self._dig(a)

    def preprocess_b(self, b: str):
        return self._dig(b)

    def preprocess_p(self, p: str):
        return self._dig(p)

    def predict_digits(self, a_enc, b_enc, p_enc) -> list[int]:
        out = self.net.greedy(a_enc, b_enc, p_enc)[0]
        ans_lsb = out[-W_ANS:].tolist()
        return list(reversed(ans_lsb))       # LSB -> MSB boundary flip
