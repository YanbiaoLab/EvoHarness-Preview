"""Evolution harness v1: search architecture/recipe space on a mini modmul task.

Motivated by the 2026-07-03 double verdict: U2 (parallel heads) and tier-3
direct (single-shot classification) both fail the same way — no serial
computation path from operands through intermediates (raw product, quotient)
to the answer. The literature's working recipes (scratchpad internalization,
Deng et al. 2405.14838) are serial. So the search space here is centered on
ORDERED autoregressive output (raw -> q -> ans) vs the parallel-head control.

Task (fast inner loop, a la AlphaEvolve "search mode"): x*y mod p for all 510
primes p in [256, 4096) — the small end of tier 3. Fitness = exact match on a
FIXED held-out pair set (fresh (x,y) on the same primes: within-prime
generalization is the tier-3 question). One rung-0 evaluation ~= 2-3 min.

Harness mechanics (borrowed from the frontier):
  * archive of every completed evaluation (stepping stones, DGM-style)
  * novelty rejection: identical genomes never re-evaluated (ShinkaEvolve)
  * ASHA rungs: 2000 -> +4000 -> +8000 steps; top third promoted — protects
    late bloomers better than single-shot short runs (grokking caveat)
  * trend bonus in fitness (slope of last evals) — same reason
  * evaluator is FROZEN: genomes touch architecture/recipe only, never eval
  * equal fixed seed per candidate — comparisons are seed-controlled

Usage:
    python -u rns/evolve.py --hours 14            # overnight on the L40S
    python -u rns/evolve.py --smoke               # 2-candidate CPU smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
EVO_DIR = HERE / "evolve"
CKPT_DIR = EVO_DIR / "ckpts"

P_LO, P_HI = 256, 4096          # mini tier-3 prime range
VAL_SIZE = 4096
EVAL_EVERY = 500

SEG_X, SEG_Y, SEG_P, SEG_RAW, SEG_Q, SEG_ANS = 0, 1, 2, 3, 4, 5
ORDER_SEGS = {"rqa": (SEG_RAW, SEG_Q, SEG_ANS), "qa": (SEG_Q, SEG_ANS), "a": (SEG_ANS,)}


def sieve(limit: int) -> list[int]:
    is_p = bytearray([1]) * limit
    is_p[0] = is_p[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if is_p[i]:
            is_p[i * i :: i] = bytearray(len(is_p[i * i :: i]))
    return [i for i in range(2, limit) if is_p[i]]


PRIMES = [p for p in sieve(P_HI) if p >= P_LO]


def widths(base: int, in_bits: int = 12, raw_bits: int = 24) -> dict[int, int]:
    """Digit widths per segment. Defaults = mini task (p < 4096); pass
    in_bits=16 / raw_bits=32 for the full tier-3 range (p < 65536)."""
    w_in = math.ceil(in_bits / math.log2(base))
    w_raw = math.ceil(raw_bits / math.log2(base))
    return {SEG_X: w_in, SEG_Y: w_in, SEG_P: w_in, SEG_RAW: w_raw, SEG_Q: w_in, SEG_ANS: w_in}


def to_digits(v: torch.Tensor, width: int, base: int) -> torch.Tensor:
    """(N,) int64 -> (N, width) LSB-first digits."""
    out = []
    for _ in range(width):
        out.append(v % base)
        v = v // base
    return torch.stack(out, dim=1)


def make_data(
    n: int, gen: torch.Generator, device, primes: list[int] | None = None
) -> dict[str, torch.Tensor]:
    pr = torch.tensor(primes if primes is not None else PRIMES,
                      dtype=torch.long, device=device)
    idx = torch.randint(pr.shape[0], (n,), generator=gen, device=device)
    p = pr[idx]
    x = (torch.rand(n, generator=gen, device=device) * p).long()
    y = (torch.rand(n, generator=gen, device=device) * p).long()
    raw = x * y
    return {"x": x, "y": y, "p": p, "raw": raw, "q": raw // p, "ans": raw % p}


# ---------------------------------------------------------------------------
# Model: one class, two families (ar = causal serial, par = encoder + slots)
# ---------------------------------------------------------------------------

class EvoNet(nn.Module):
    def __init__(self, g: dict):
        super().__init__()
        self.g = g
        base, d = g["base"], g["d"]
        self.W = widths(base, g.get("in_bits", 12), g.get("raw_bits", 24))
        self.out_segs = ORDER_SEGS[g["order"]] if g["arch"] == "ar" else (SEG_RAW, SEG_Q, SEG_ANS)
        self.in_segs = (SEG_X, SEG_Y, SEG_P)
        self.tok_emb = nn.Embedding(base, d)
        self.seg_emb = nn.Embedding(6, d)
        self.sig_emb = nn.Embedding(max(self.W.values()), d)
        total_len = sum(self.W[s] for s in self.in_segs) + sum(self.W[s] for s in self.out_segs)
        self.pos_emb = nn.Embedding(total_len + 1, d)
        self.slot_query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=8, dim_feedforward=4 * d, dropout=0.0,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=g["layers"])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, base)
        self.total_len = total_len

    def _embed_digits(self, digits: torch.Tensor, seg: int, pos0: int) -> torch.Tensor:
        n, w = digits.shape
        e = self.tok_emb(digits)
        e = e + self.seg_emb.weight[seg]
        e = e + self.sig_emb.weight[:w].unsqueeze(0)
        pos = torch.arange(pos0, pos0 + w, device=digits.device)
        return e + self.pos_emb(pos).unsqueeze(0)

    def _out_slot_emb(self, n: int, seg: int, w: int, pos0: int, device) -> torch.Tensor:
        e = self.slot_query.expand(n, w, -1).contiguous()
        e = e + self.seg_emb.weight[seg]
        e = e + self.sig_emb.weight[:w].unsqueeze(0)
        pos = torch.arange(pos0, pos0 + w, device=device)
        return e + self.pos_emb(pos).unsqueeze(0)

    def input_prefix(self, batch: dict) -> torch.Tensor:
        base = self.g["base"]
        parts, pos = [], 0
        for seg, key in ((SEG_X, "x"), (SEG_Y, "y"), (SEG_P, "p")):
            w = self.W[seg]
            parts.append(self._embed_digits(to_digits(batch[key], w, base), seg, pos))
            pos += w
        return torch.cat(parts, dim=1)

    def out_targets(self, batch: dict) -> torch.Tensor:
        base = self.g["base"]
        key = {SEG_RAW: "raw", SEG_Q: "q", SEG_ANS: "ans"}
        return torch.cat(
            [to_digits(batch[key[s]], self.W[s], base) for s in self.out_segs], dim=1
        )

    def forward_teacher(self, batch: dict) -> torch.Tensor:
        """Training forward. Returns logits (N, L_out, base) for output digits."""
        prefix = self.input_prefix(batch)
        n = prefix.shape[0]
        device = prefix.device
        tgt = self.out_targets(batch)

        if self.g["arch"] == "par":
            pos = prefix.shape[1]
            slots = []
            for s in self.out_segs:
                w = self.W[s]
                slots.append(self._out_slot_emb(n, s, w, pos, device))
                pos += w
            h = self.encoder(torch.cat([prefix] + slots, dim=1))
            out_h = h[:, prefix.shape[1]:]
            return self.head(self.ln(out_h))

        # ar: causal LM over [prefix | shifted targets]; logits at position j
        # predict output digit j (last prefix slot predicts the first digit).
        base = self.g["base"]
        pos = prefix.shape[1]
        emb_parts = [prefix]
        for s in self.out_segs:
            w = self.W[s]
            seg_digits = tgt[:, : 0]  # placeholder to keep shapes explicit
            emb_parts.append(None)  # filled below
            pos += w
        # build shifted output embeddings: digits fed in are tgt[:, :-1]
        out_emb = []
        pos = prefix.shape[1]
        flat_i = 0
        for s in self.out_segs:
            w = self.W[s]
            seg_tgt = tgt[:, flat_i : flat_i + w]
            e = self.tok_emb(seg_tgt)
            e = e + self.seg_emb.weight[s]
            e = e + self.sig_emb.weight[:w].unsqueeze(0)
            p_ids = torch.arange(pos, pos + w, device=device)
            out_emb.append(e + self.pos_emb(p_ids).unsqueeze(0))
            pos += w
            flat_i += w
        out_emb = torch.cat(out_emb, dim=1)
        seq = torch.cat([prefix, out_emb[:, :-1]], dim=1)
        t = seq.shape[1]
        mask = torch.triu(torch.full((t, t), float("-inf"), device=device), diagonal=1)
        h = self.encoder(seq, mask=mask, is_causal=True)
        l_out = tgt.shape[1]
        out_h = h[:, prefix.shape[1] - 1 : prefix.shape[1] - 1 + l_out]
        return self.head(self.ln(out_h))

    @torch.no_grad()
    def greedy_decode(self, batch: dict) -> torch.Tensor:
        """Returns predicted output digits (N, L_out)."""
        if self.g["arch"] == "par":
            logits = self.forward_teacher(batch)
            return logits.argmax(dim=-1)
        prefix = self.input_prefix(batch)
        n = prefix.shape[0]
        device = prefix.device
        seq = prefix
        preds = []
        pos = prefix.shape[1]
        for s in self.out_segs:
            w = self.W[s]
            for i in range(w):
                t = seq.shape[1]
                mask = torch.triu(torch.full((t, t), float("-inf"), device=device), diagonal=1)
                h = self.encoder(seq, mask=mask, is_causal=True)
                logits = self.head(self.ln(h[:, -1]))
                nxt = logits.argmax(dim=-1)
                preds.append(nxt)
                e = self.tok_emb(nxt).unsqueeze(1)
                e = e + self.seg_emb.weight[s]
                e = e + self.sig_emb.weight[i]
                e = e + self.pos_emb.weight[pos].unsqueeze(0).unsqueeze(0)
                seq = torch.cat([seq, e], dim=1)
                pos += 1
        return torch.stack(preds, dim=1)


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

def loss_weights(net: EvoNet, g: dict, device) -> torch.Tensor:
    per_seg = {SEG_RAW: g["aux_raw"], SEG_Q: g["aux_q"], SEG_ANS: 1.0}
    ws = []
    for s in net.out_segs:
        ws.extend([per_seg[s]] * net.W[s])
    return torch.tensor(ws, dtype=torch.float32, device=device)


@torch.no_grad()
def val_exact(net: EvoNet, val: dict, device, chunk: int = 2048) -> float:
    net.eval()
    tgt = net.out_targets(val)
    ans_w = net.W[SEG_ANS]
    n = val["x"].shape[0]
    correct = 0
    for i in range(0, n, chunk):
        sub = {k: v[i : i + chunk] for k, v in val.items()}
        pred = net.greedy_decode(sub)
        ok = (pred[:, -ans_w:] == tgt[i : i + chunk][:, -ans_w:]).all(dim=1)
        correct += int(ok.sum().item())
    return correct / n


def run_candidate(g: dict, steps: int, device, val: dict, state=None) -> dict:
    torch.manual_seed(0)
    if state is None:
        net = EvoNet(g).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=g["lr"], weight_decay=g["wd"])
        done = 0
        curve = []
    else:
        net, opt, done, curve = state
    gen = torch.Generator(device=device)
    gen.manual_seed(1234 + done)
    lw = loss_weights(net, g, device)
    warmup = 300

    net.train()
    t0 = time.monotonic()
    for s in range(done, done + steps):
        for pg in opt.param_groups:
            pg["lr"] = g["lr"] * min(1.0, (s + 1) / warmup)
        batch = make_data(g["batch"], gen, device)
        logits = net.forward_teacher(batch)
        tgt = net.out_targets(batch)
        ce = F.cross_entropy(
            logits.reshape(-1, g["base"]), tgt.reshape(-1), reduction="none"
        ).view(tgt.shape)
        loss = (ce * lw).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (s + 1) % EVAL_EVERY == 0:
            e = val_exact(net, val, device)
            curve.append(round(e, 4))
            net.train()
    dt = time.monotonic() - t0
    final = curve[-1] if curve else 0.0
    trend = max(0.0, curve[-1] - curve[-2]) if len(curve) >= 2 else 0.0
    return {
        "state": (net, opt, done + steps, curve),
        "fitness": round(final + 0.3 * trend, 4),
        "exact": final,
        "curve": list(curve),
        "steps": done + steps,
        "params": sum(p.numel() for p in net.parameters()),
        "sec": round(dt, 1),
    }


# ---------------------------------------------------------------------------
# Genome space
# ---------------------------------------------------------------------------

D_CHOICES = (128, 192, 256)
L_CHOICES = (3, 4, 6)
WD_CHOICES = (0.03, 0.1, 0.3)
AUX_CHOICES = (0.0, 0.3, 1.0)


def random_genome(rng: random.Random) -> dict:
    return {
        "arch": rng.choice(("ar", "ar", "par")),  # bias toward the hypothesis
        "order": rng.choice(("rqa", "qa", "a")),
        "base": rng.choice((10, 100)),
        "d": rng.choice(D_CHOICES),
        "layers": rng.choice(L_CHOICES),
        "lr": round(10 ** rng.uniform(-4, -2.9), 6),
        "wd": rng.choice(WD_CHOICES),
        "aux_raw": rng.choice(AUX_CHOICES),
        "aux_q": rng.choice(AUX_CHOICES),
        "batch": 1024,
    }


def designed_seeds() -> list[dict]:
    base = dict(batch=1024)
    return [
        dict(base, arch="ar", order="rqa", base=10, d=256, layers=6, lr=5e-4, wd=0.1, aux_raw=1.0, aux_q=1.0),
        dict(base, arch="ar", order="qa", base=100, d=256, layers=4, lr=5e-4, wd=0.1, aux_raw=0.0, aux_q=1.0),
        dict(base, arch="ar", order="a", base=10, d=256, layers=6, lr=5e-4, wd=0.1, aux_raw=0.0, aux_q=0.0),
        dict(base, arch="par", order="rqa", base=10, d=256, layers=6, lr=5e-4, wd=0.1, aux_raw=0.3, aux_q=0.3),
        dict(base, arch="ar", order="rqa", base=100, d=192, layers=6, lr=3e-4, wd=0.3, aux_raw=1.0, aux_q=1.0),
    ]


def mutate(g: dict, rng: random.Random) -> dict:
    m = dict(g)
    if rng.random() < 0.2:
        m["arch"] = "par" if g["arch"] == "ar" else "ar"
    if rng.random() < 0.4:
        m["order"] = rng.choice(("rqa", "qa", "a"))
    if rng.random() < 0.3:
        m["base"] = 100 if g["base"] == 10 else 10
    if rng.random() < 0.4:
        m["d"] = rng.choice(D_CHOICES)
    if rng.random() < 0.4:
        m["layers"] = rng.choice(L_CHOICES)
    if rng.random() < 0.6:
        m["lr"] = round(min(2e-3, max(5e-5, g["lr"] * math.exp(rng.gauss(0, 0.5)))), 6)
    if rng.random() < 0.3:
        m["wd"] = rng.choice(WD_CHOICES)
    if rng.random() < 0.3:
        m["aux_raw"] = rng.choice(AUX_CHOICES)
    if rng.random() < 0.3:
        m["aux_q"] = rng.choice(AUX_CHOICES)
    return m


def crossover(a: dict, b: dict, rng: random.Random) -> dict:
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in a}


def gid(g: dict) -> str:
    return hashlib.sha1(json.dumps(g, sort_keys=True).encode()).hexdigest()[:10]


