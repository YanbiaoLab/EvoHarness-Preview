"""Architecture: serial autoregressive transformer (raw -> quotient -> answer).

MUTATION SURFACE — architecture and representation.

Derived from the closed-genome winner family (2026-07-03 verdict: a serial
computation path beats parallel heads). The scaffold is the point: the model
emits the raw product, then the quotient, then the answer, so the answer is
conditioned on the intermediates instead of guessed in one parallel shot.

Known ceiling: the digit windows below are fixed decimal widths, so this
family tops out around tier 3. Widening the windows is a legitimate mutation,
but the cost is quadratic in sequence length — the width-generic route is the
limb_horner family.
"""

from __future__ import annotations

import torch
from torch import nn

BASE = 10
W_IN, W_RAW, W_Q, W_ANS = 20, 39, 39, 5
SEG_X, SEG_Y, SEG_P, SEG_RAW, SEG_Q, SEG_ANS = range(6)
IN_SEGS = ((SEG_X, W_IN), (SEG_Y, W_IN), (SEG_P, W_IN))
OUT_SEGS = ((SEG_RAW, W_RAW), (SEG_Q, W_Q), (SEG_ANS, W_ANS))
L_IN = 3 * W_IN
L_OUT = W_RAW + W_Q + W_ANS
TOTAL_LEN = L_IN + L_OUT

D_MODEL = 128
LAYERS = 4
N_HEAD = 4


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
        e = (self.tok(dig) + self.seg.weight[seg_id]
             + self.sig.weight[:w].unsqueeze(0))
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
        mask = torch.triu(
            torch.full((t, t), float("-inf"), device=seq.device), 1)
        h = self.enc(seq, mask=mask, is_causal=True)
        return self.head(self.ln(h[:, L_IN - 1: L_IN - 1 + tgt.shape[1]]))

    @torch.no_grad()
    def greedy(self, x: torch.Tensor, y: torch.Tensor,
               p: torch.Tensor) -> torch.Tensor:
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
        return torch.stack(outs, 1)      # (N, L_OUT), LSB-first per segment
