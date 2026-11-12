from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Limite's local-layer shapes, so the engine's kernels run the drafter's layers.
H, D, NQ, NKV, FFN = 1280, 128, 10, 2, 3328
SCALE = 0.1
EPS = 2.0 ** -23  # Limite's rms_norm epsilon

LAYERS = 2
TAPS = (35, 47)  # the target layers the drafter reads; the last one's output is the target's head input
BLOCK = 8  # the most rows a block has: the bonus token and 7 drafts, the engine's most
WINDOW = 1024  # the context positions a row sees: the engine's local-layer window
TEMP = 0.6  # the temperature the traces are sampled at; drafter.rs scores draft trees at it


def rms(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.shape[-1],), eps=EPS)


def rope_tables(pos: Tensor) -> tuple[Tensor, Tensor]:
    """cos, sin [..., 32] of positions pos [...], as the engine's table: float32(pos) *
    (1/1024)^linspace(0, 1, 32), rounded to bf16."""
    f = (1.0 / 1024) ** torch.linspace(0, 1, 32, dtype=torch.float32, device=pos.device)
    theta = pos.to(torch.float32)[..., None] * f
    return theta.cos().bfloat16().float(), theta.sin().bfloat16().float()


def rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x [..., T, 128]: pairs (2j, 2j + 1), j < 32, become (c a0 + s a1, c a1 - s a0)."""
    xr = x[..., :64].unflatten(-1, (32, 2))
    a0, a1 = xr[..., 0], xr[..., 1]
    out = torch.stack([cos * a0 + sin * a1, cos * a1 - sin * a0], -1).flatten(-2)
    return torch.cat([out, x[..., 64:]], -1)


class Conv2(nn.Module):
    """DFlash-2's two-tap dynamic convolution (Inco AI, Aug 2026): within each block, left
    to right, conv(x)_t = k_t0 * x_t + k_t1 * x_(t-1), each kernel a learned base plus a
    correction from the current hidden state, one per 16 channels. Row 0 has no
    predecessor. Starts as the identity."""

    def __init__(self, group: int = 16) -> None:
        super().__init__()
        self.group = group
        self.base = nn.Parameter(torch.stack([torch.ones(H), torch.zeros(H)]))
        self.corr = nn.Linear(H, 2 * H // group, bias=False)
        nn.init.zeros_(self.corr.weight)

    def forward(self, x: Tensor, rows: int) -> Tensor:
        xs = x.reshape(-1, rows, H)
        prev = F.pad(xs[:, :-1], (0, 0, 1, 0))
        c = self.corr(rms(xs)).unflatten(-1, (2, H // self.group)).repeat_interleave(self.group, -1)
        k = self.base + c
        return (k[..., 0, :] * xs + k[..., 1, :] * prev).reshape(x.shape)


def attend(q: Tensor, kc: Tensor, vc: Tensor, kb: Tensor, vb: Tensor, mask: Tensor, rows: int) -> Tensor:
    """Attention of block rows over the context and over their own block, in one softmax,
    for B spans: q [B, NQ, A * rows, D]; kc, vc [B, NKV, C, D] the context's keys and values;
    kb, vb [B, NKV, A * rows, D] the blocks'; mask [B, A * rows, C] (True = visible). A row
    sees its whole block, so the blocks' part is A small rows x rows products, not one over
    all blocks' keys: linear in the anchors."""
    bs, n, g = q.shape[0], q.shape[2], NQ // NKV
    a = n // rows
    qg = q.view(bs, NKV, g, n, D)  # query head h reads kv head h // g
    sc = torch.einsum("bkgnd,bkcd->bkgnc", qg, kc) * SCALE
    sc = sc.masked_fill(~mask[:, None, None], float("-inf"))
    sb = torch.einsum("bkgard,bkasd->bkgars", qg.view(bs, NKV, g, a, rows, D), kb.view(bs, NKV, a, rows, D))
    p = torch.softmax(torch.cat([sc, (sb * SCALE).reshape(bs, NKV, g, n, rows)], -1), -1)
    c = kc.shape[2]
    out = torch.einsum("bkgnc,bkcd->bkgnd", p[..., :c], vc)
    out = out + torch.einsum("bkgars,bkasd->bkgard", p[..., c:].view(bs, NKV, g, a, rows, rows),
                             vb.view(bs, NKV, a, rows, D)).reshape(bs, NKV, g, n, D)
    return out.reshape(bs, NQ, n, D)


class Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Before attention, after it, before the MLP and after it.
        self.convs = nn.ModuleList(Conv2() for _ in range(4))
        self.wq = nn.Linear(H, NQ * D, bias=False)
        self.wk = nn.Linear(H, NKV * D, bias=False)
        self.wv = nn.Linear(H, NKV * D, bias=False)
        self.wo = nn.Linear(NQ * D, H, bias=False)
        self.wgu = nn.Linear(H, 2 * FFN, bias=False)  # rows interleaved: gate_0, up_0, ...
        self.wd = nn.Linear(FFN, H, bias=False)

    def kv(self, h: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
        """Keys and values [B, 2, T, 128] of normed inputs h [B, T, H]."""
        k = rms(self.wk(h).unflatten(-1, (NKV, D)).transpose(1, 2))
        v = self.wv(h).unflatten(-1, (NKV, D)).transpose(1, 2)
        return rope(k, cos, sin), v

    def forward(self, x: Tensor, ctx: tuple[Tensor, Tensor], cos: Tensor, sin: Tensor, mask: Tensor,
                rows: int) -> Tensor:
        cv = self.convs
        h = rms(cv[0](x, rows))
        q = rope(rms(self.wq(h).unflatten(-1, (NQ, D)).transpose(1, 2)), cos, sin)
        kb, vb = self.kv(h, cos, sin)
        o = self.wo(attend(q, ctx[0], ctx[1], kb, vb, mask, rows).transpose(1, 2).flatten(-2))
        x = x + cv[1](o, rows)
        g, u = self.wgu(rms(cv[2](x, rows))).unflatten(-1, (FFN, 2)).unbind(-1)
        return x + cv[3](self.wd(F.silu(g) * u), rows)


class Drafter(nn.Module):
    def __init__(self, vocab: Tensor) -> None:
        """vocab: the draft vocabulary's token ids, ascending (target.py)."""
        super().__init__()
        self.fc = nn.Linear(len(TAPS) * H, H, bias=False)
        # The fused context of the position before the block, added to every row's input
        # through one projection (from zero), as EAGLE-3 feeds its drafter: the target's
        # state there, without learning to find it among the context keys.
        self.inject = nn.Linear(H, H, bias=False)
        nn.init.zeros_(self.inject.weight)
        self.mask_emb = nn.Parameter(torch.randn(H) * 0.02)
        self.layers = nn.ModuleList(Layer() for _ in range(LAYERS))
        # Raw logits: through Limite's cap, 23 * sigmoid((x + 5) / 7.5), the KL drives them
        # into the sigmoid's flat ends, past 100: nothing left to learn, and float32 ties.
        self.head = nn.Linear(H, len(vocab), bias=False)
        # The final norm's gain, from zero: every row starts uniform, where a gain of 1 would
        # scale a near-zero hidden state up to confident nonsense. export.py folds it into
        # the head's columns.
        self.out_gain = nn.Parameter(torch.zeros(H))
        self.register_buffer("vocab", vocab.clone(), persistent=True)

    @classmethod
    def load(cls, path: Path, dev: torch.device | str = "cpu") -> Drafter:
        """A train.py checkpoint."""
        state = torch.load(path, map_location=dev)["state"]
        model = cls(state["vocab"]).to(dev)
        model.load_state_dict(state)
        return model

    def context(self, feats: Tensor) -> Tensor:
        """feats [B, C, len(TAPS), H] (bf16, normed) -> the fused context [B, C, H], normed."""
        return rms(self.fc(feats.float().flatten(-2)))

    def forward(self, ctx: Tensor, c0: Tensor, bonus: Tensor, off: Tensor, rows: int = BLOCK) -> Tensor:
        """The final hidden states [B, A, rows, H] of A blocks of `rows` rows in each of B
        spans; `head` of them is the logits, and rows 1.. are the ones that draft.

        ctx [B, C, H]: each span's fused context, at positions c0 .. c0 + C - 1 (c0 [B]).
        bonus [B, A, H]: rms_norm(embedding) of each block's bonus token.
        off [B, A]: each bonus token's offset into ctx, at least 1: its position L is
        c0 + off, and its block sees only the context before it."""
        bs, a = off.shape
        dev = ctx.device
        last = ctx[torch.arange(bs, device=dev)[:, None], off - 1]
        x = torch.cat([bonus[:, :, None], self.mask_emb.expand(bs, a, rows - 1, H)], 2)
        x = (x + self.inject(last)[:, :, None]).reshape(bs, a * rows, H)
        pos = ((c0[:, None] + off)[..., None] + torch.arange(rows, device=dev)).reshape(bs, -1)
        cos_b, sin_b = (t[:, None] for t in rope_tables(pos))
        cos_c, sin_c = (t[:, None] for t in rope_tables(c0[:, None] + torch.arange(ctx.shape[1], device=dev)))
        mask = block_mask(off, ctx.shape[1], rows)
        for layer in self.layers:
            x = layer(x, layer.kv(ctx, cos_c, sin_c), cos_b, sin_b, mask, rows)
        return (rms(x) * self.out_gain).view(bs, a, rows, H)


def block_mask(off: Tensor, n_ctx: int, rows: int) -> Tensor:
    """Context visibility [B, A * rows, n_ctx] for blocks at context offsets off [B, A]:
    row i of a block sees context offsets [off + i - WINDOW, off), as the engine's
    local-layer window (keys from position - WINDOW). Each row also sees its whole block
    (attend)."""
    dev = off.device
    o = off.repeat_interleave(rows, dim=-1)[..., None]
    row_i = torch.arange(off.shape[-1] * rows, device=dev)[:, None] % rows
    q = torch.arange(n_ctx, device=dev)[None]
    return (q < o) & (q >= o + row_i - WINDOW)
