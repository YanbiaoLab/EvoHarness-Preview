"""Seed candidate: serial autoregressive transformer (raw -> quotient -> answer).

Derived from the closed-genome winner family (modmul/baselines/closed_genome.py,
2026-07-03 verdict: serial computation path beats parallel heads).

CONTRACT NOTES FOR MUTATION:
- The predict path must be purely neural: no %, //, or big-int arithmetic on
  the answer path. Training-time label synthesis below MAY use exact integer
  arithmetic — the rules require the ANSWER to come from trained parameters.
- The official decoder expects MSB-first digits; internal representation is
  LSB-first (matches carry structure). The flip happens ONLY at the
  predict_digits boundary.
- preprocess_a/b/p may each read ONLY their own argument (isolation-checked).
"""

import random
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

from modchallenge.interface.base_model import ModularMultiplicationModel

MANIFEST = {
    "entry_class": "model.EvolvedModel",
    "output_base": 10,
    "model_description": "serial AR transformer emitting raw->quotient->answer "
                         "digit chains",
    "training_description": "trained at eval time on synthesized tier-1..3 data, "
                            "fixed seed 0, AdamW + warmup",
}

W_IN, W_RAW, W_Q, W_ANS = 20, 39, 39, 5
SEG_X, SEG_Y, SEG_P, SEG_RAW, SEG_Q, SEG_ANS = range(6)
IN_SEGS = ((SEG_X, W_IN), (SEG_Y, W_IN), (SEG_P, W_IN))
OUT_SEGS = ((SEG_RAW, W_RAW), (SEG_Q, W_Q), (SEG_ANS, W_ANS))
L_IN = 3 * W_IN
L_OUT = W_RAW + W_Q + W_ANS
TOTAL_LEN = L_IN + L_OUT


def _sieve(limit: int) -> list[int]:
    flags = bytearray([1]) * limit
    flags[0] = flags[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if flags[i]:
            flags[i * i :: i] = bytearray(len(flags[i * i :: i]))
    return [i for i in range(2, limit) if flags[i]]

# Task-fixed (deliberately OUTSIDE the edit region): tier sampling is part of
# the task definition, not the mutation surface — a candidate must not learn
# to train only on easy tiers. (p_bits_lo, p_bits_hi, operand_bits) per tier.
TIERS = ((1, 3, 32), (4, 8, 48), (9, 16, 64))
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
    """Training labels via exact integer arithmetic — LEGAL at train time."""
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


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# EDIT-REGION-BEGIN  (mutation surface: representation / architecture / recipe)


BASE = 10
D_MODEL = 128
LAYERS = 4
N_HEAD = 4
LR = 5e-4
WD = 0.1
AUX_RAW, AUX_Q = 1.0, 1.0        # loss weight on intermediate segments
TRAIN_STEPS = 3000
BATCH = 256
SEED = 0


class SerialNet(nn.Module):
    def __init__(self):
        super().__init__()
        d = D_MODEL
        self.tok = nn.Embedding(BASE, d)
        self.seg = nn.Embedding(6, d)
        self.sig = nn.Embedding(max(W_RAW, W_IN, W_ANS), d)
        self.pos = nn.Embedding(TOTAL_LEN, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=N_HEAD, dim_feedforward=4 * d, dropout=0.0,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=LAYERS)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, BASE)

    def _emb(self, dig: torch.Tensor, seg_id: int, pos0: int) -> torch.Tensor:
        n, w = dig.shape
        e = self.tok(dig) + self.seg.weight[seg_id] + self.sig.weight[:w].unsqueeze(0)
        pos = torch.arange(pos0, pos0 + w, device=dig.device)
        return e + self.pos(pos).unsqueeze(0)

    def prefix(self, batch: dict) -> torch.Tensor:
        parts, pos0 = [], 0
        for (seg_id, w), key in zip(IN_SEGS, ("x", "y", "p")):
            parts.append(self._emb(batch[key], seg_id, pos0))
            pos0 += w
        return torch.cat(parts, 1)

    def targets(self, batch: dict) -> torch.Tensor:
        return torch.cat([batch[k] for k in ("raw", "q", "ans")], 1)

    def forward_teacher(self, batch: dict) -> torch.Tensor:
        pre, tgt = self.prefix(batch), self.targets(batch)
        parts, pos0, i = [], L_IN, 0
        for seg_id, w in OUT_SEGS:
            parts.append(self._emb(tgt[:, i:i + w], seg_id, pos0))
            pos0 += w
            i += w
        out_e = torch.cat(parts, 1)
        seq = torch.cat([pre, out_e[:, :-1]], 1)
        t = seq.shape[1]
        mask = torch.triu(torch.full((t, t), float("-inf"), device=seq.device), 1)
        h = self.enc(seq, mask=mask, is_causal=True)
        return self.head(self.ln(h[:, L_IN - 1 : L_IN - 1 + tgt.shape[1]]))

    @torch.no_grad()
    def greedy(self, x: torch.Tensor, y: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        seq, pos0 = self.prefix({"x": x, "y": y, "p": p}), L_IN
        outs = []
        for seg_id, w in OUT_SEGS:
            for j in range(w):
                t = seq.shape[1]
                mask = torch.triu(
                    torch.full((t, t), float("-inf"), device=seq.device), 1)
                h = self.enc(seq, mask=mask, is_causal=True)
                nxt = self.head(self.ln(h[:, -1])).argmax(-1)
                outs.append(nxt)
                e = (self.tok(nxt) + self.seg.weight[seg_id]
                     + self.sig.weight[j] + self.pos.weight[pos0]).unsqueeze(1)
                seq = torch.cat([seq, e], 1)
                pos0 += 1
        return torch.stack(outs, 1)          # (N, L_OUT), LSB-first per segment
    
# EDIT-REGION-END


class EvolvedModel(ModularMultiplicationModel):
    def load(self, model_dir: str) -> None:
        self.device = _pick_device()
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
    
    
def train(model_dir: str) -> None:
    torch.manual_seed(SEED)
    rng = random.Random(SEED)
    device = _pick_device()
    net = SerialNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    lw = torch.tensor([AUX_RAW] * W_RAW + [AUX_Q] * W_Q + [1.0] * W_ANS,
                      dtype=torch.float32, device=device)
    warmup = min(300, TRAIN_STEPS)
    net.train()
    for s in range(TRAIN_STEPS):
        for pg in opt.param_groups:
            pg["lr"] = LR * min(1.0, (s + 1) / warmup)
        batch = synthesize(BATCH, rng, device)
        logits = net.forward_teacher(batch)
        tgt = net.targets(batch)
        ce = F.cross_entropy(logits.reshape(-1, BASE), tgt.reshape(-1),
                             reduction="none").view(tgt.shape)
        loss = (ce * lw).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.save(net.state_dict(), Path(model_dir) / "weights.pt")