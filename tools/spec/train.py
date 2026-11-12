from __future__ import annotations

import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from itertools import accumulate
from pathlib import Path

import torch
from safetensors import safe_open

from drafter import BLOCK, TAPS, TEMP, WINDOW, Drafter, H, rms
from traces import SPEC

ANCHORS, SPANS, SPAN = 1024, 4, 512  # a step: anchors, the traces they come from, positions a trace's come from
BLOCK_ROWS = (2, 4, 6, 8, 8)  # block sizes a step draws from
GAMMA = 7.0  # row i's weight in the loss: exp(-(i - 1) / GAMMA)
LR, WARMUP, DECAY = 4e-4, 200, 0.2  # peak, warmup steps, the share of the run decaying to zero
ANCHORS_PER_POSITION = 4  # the run's length, unless --steps; past this the drafter memorises the traces
READERS = 4  # threads packing steps ahead, and steps ahead
BUCKETS = (8192, 32768, 65536)  # the evaluation's position buckets end here


def device() -> torch.device:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def autocast(dev: torch.device):
    """bf16 matmuls on CUDA, over fp32 weights."""
    return torch.autocast("cuda", dtype=torch.bfloat16) if dev.type == "cuda" else nullcontext()


class Features:
    """Feature dumps: per trace its tokens, its prompt's length and its features at every
    position, read from disk a slice at a time."""

    def __init__(self, dirs: list[Path]):
        self.files = sorted(f for d in dirs for f in Path(d).glob("*.safetensors"))
        assert self.files, f"no feature dumps in {dirs}"
        self.tokens, self.prompt_len = [], []
        for f in self.files:
            with safe_open(f, "pt") as st:
                assert st.get_tensor("taps").tolist() == list(TAPS), f"{f}: not the drafter's taps"
                self.tokens.append(st.get_tensor("tokens").to(torch.int64))
                self.prompt_len.append(int(st.get_tensor("prompt_len")[0]))
        self.positions = sum(len(t) for t in self.tokens)

    def anchors(self, i: int) -> range:
        """Trace i's anchors: past the prompt, with a whole block's tokens after them."""
        return range(max(1, self.prompt_len[i]), len(self.tokens[i]) - BLOCK + 1)

    def feat(self, i: int, a: int, b: int) -> torch.Tensor:
        """Trace i's features at positions a .. b - 1: bf16 [b - a, taps, H]."""
        with safe_open(self.files[i], "pt") as st:
            return st.get_slice("feat")[a:b]


class Target:
    """target.py's file: the target's embedding, and its head over the draft vocabulary."""

    def __init__(self, path: Path, dev: torch.device):
        t = torch.load(path)
        self.vocab = t["vocab"]
        self.embed = t["embed"].to(dev)  # bf16 [ids, H]
        self.head = t["head"].to(dev)  # fp32 [V, H]
        self.index = torch.full((len(self.embed),), -1, dtype=torch.long)  # id -> its row of the vocabulary
        self.index[self.vocab] = torch.arange(len(self.vocab))
        self.index = self.index.to(dev)

    def bonus(self, ids: torch.Tensor) -> torch.Tensor:
        return rms(self.embed[ids].float())

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        """Raw logits over the draft vocabulary, from the last layer's features h [..., H]."""
        return h.float() @ self.head.T


def pack(data: Features, spans: list[tuple[int, list[int]]], rows: int, per: int, ctx_len: int, pin: bool = False):
    """Spans (trace, ascending anchors, at most `per`, within ctx_len - WINDOW positions) as
    fixed-shape CPU tensors, for blocks of `rows` rows: one pass for the whole step, and
    one compiled graph per block size.

    feat [B, ctx_len + rows, taps, H]: each span's features from its context's start, c0 [B],
    zeros past its end; off [B, per]: each anchor's offset into them; bonus [B, per]: the
    token at each anchor; nxt [B, per, rows - 1]: the tokens after it; live [B, per]: 1, or 0
    where a short span repeats its last anchor."""
    bs = len(spans)
    feat = torch.zeros(bs, ctx_len + rows, len(TAPS), H, dtype=torch.bfloat16, pin_memory=pin)
    off = torch.zeros(bs, per, dtype=torch.long)
    c0 = torch.zeros(bs, dtype=torch.long)
    bonus = torch.zeros(bs, per, dtype=torch.long)
    nxt = torch.zeros(bs, per, rows - 1, dtype=torch.long)
    live = torch.zeros(bs, per)
    for b, (i, js) in enumerate(spans):
        c0[b] = max(0, js[0] - WINDOW)
        f = data.feat(i, int(c0[b]), js[-1] + rows - 1)  # the context, then the drafted positions
        feat[b, :len(f)] = f
        jt = torch.tensor(js + [js[-1]] * (per - len(js)))
        tokens = data.tokens[i]
        off[b], bonus[b] = jt - c0[b], tokens[jt]
        nxt[b] = tokens[jt[:, None] + 1 + torch.arange(rows - 1)]
        live[b, :len(js)] = 1
    return feat, off, c0, bonus, nxt, live


def draft(model: Drafter, target: Target, feat, off, c0, bonus, rows: int) -> torch.Tensor:
    """The draft logits [B, per, rows - 1, V] of a packed step: raw, as the engine uses them."""
    with autocast(feat.device):
        ctx = model.context(feat[:, :feat.shape[1] - rows])
        return model.head(model(ctx, c0, target.bonus(bonus), off, rows)[:, :, 1:])


def accepted(drafts: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Leading matches: [..., K] -> [...]."""
    return torch.cumprod((drafts == truth).long(), -1).sum(-1)


def loss_and_tau(model: Drafter, target: Target, packed, rows: int, weights: torch.Tensor):
    feat, off, c0, bonus, nxt, live = packed
    logits = draft(model, target, feat, off, c0, bonus, rows).float()
    with torch.no_grad():
        # The target's distribution for the token at L + i is its head at L + i - 1.
        spans = torch.arange(len(off), device=off.device)[:, None, None]
        at = off[..., None] + torch.arange(rows - 1, device=off.device)
        p = torch.softmax(target.logits(feat[spans, at, -1]) / TEMP, -1)
    kl = (p * (torch.log(p.clamp_min(1e-12)) - torch.log_softmax(logits / TEMP, -1))).sum(-1)  # [B, per, rows - 1]
    w = weights[:rows - 1]
    loss = (((kl * w).sum(-1) / w.sum()) * live).sum() / live.sum()
    with torch.no_grad():
        tau = ((1 + accepted(logits.argmax(-1), target.index[nxt])) * live).sum() / live.sum()
    return loss, tau


# ------------------------------------------------------------------ training

def train(args) -> None:
    dev = device()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    data = Features(args.features)
    held = Features(args.eval) if args.eval else None
    target = Target(args.target, dev)
    if args.init:
        model = Drafter.load(args.init, dev)
        assert torch.equal(model.vocab.cpu(), target.vocab), "--init has another draft vocabulary"
    else:
        model = Drafter(target.vocab).to(dev)
        with torch.no_grad():
            # The head starts as the target's, and every layer as the identity.
            model.head.weight.copy_(target.head)
            for layer in model.layers:
                layer.wo.weight.zero_()
                layer.wd.weight.zero_()
    steps = args.steps or round(ANCHORS_PER_POSITION * data.positions / ANCHORS)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters; {len(data.files)} traces, "
          f"{data.positions} positions; {steps} steps of {ANCHORS} anchors; {dev}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
    weights = torch.exp(-torch.arange(BLOCK - 1, dtype=torch.float32, device=dev) / GAMMA)
    step_fn = torch.compile(loss_and_tau, dynamic=False) if args.compile else loss_and_tau
    per = ANCHORS // SPANS
    by_length = list(accumulate(len(t) for t in data.tokens))

    def plan():
        """The run's steps: the block size, and each span's (trace, anchors). A trace's
        anchors come from one span of SPAN positions: their contexts overlap, where anchors
        spread over the trace would each need a window of features of their own."""
        for _ in range(steps):
            rows = rng.choice(BLOCK_ROWS)
            spans = []
            while len(spans) < SPANS:
                i = rng.choices(range(len(data.files)), cum_weights=by_length)[0]
                anchors = data.anchors(i)
                if len(anchors) < 1:
                    continue
                start = rng.randrange(max(1, len(anchors) - SPAN + 1))
                span = anchors[start:start + SPAN]
                spans.append((i, sorted(rng.sample(span, min(per, len(span))))))
            yield rows, spans

    # Steps are read and packed ahead of the one that needs them.
    pool, queue, todo = ThreadPoolExecutor(READERS), [], plan()

    cuda = dev.type == "cuda"  # pinned memory, and copies to the GPU that don't wait

    def fill():
        while len(queue) < READERS and (nxt := next(todo, None)):
            queue.append((nxt[0], pool.submit(pack, data, nxt[1], nxt[0], per, WINDOW + SPAN, cuda)))

    t0, log = time.time(), []
    for step in range(steps):
        lr = LR * min(1.0, (step + 1) / WARMUP) * min(1.0, (steps - step) / (DECAY * steps))
        for group in opt.param_groups:
            group["lr"] = lr
        fill()
        rows, packed = queue.pop(0)
        packed = [x.to(dev, non_blocking=cuda) for x in packed.result()]
        opt.zero_grad(set_to_none=True)
        loss, tau = step_fn(model, target, packed, rows, weights)
        loss.backward()
        log.append((loss.detach(), tau))  # read at the next report: no sync here
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % args.log == 0:
            loss, tau = (torch.stack(x).mean().item() for x in zip(*log))
            print(f"step {step + 1}: loss {loss:.4f}  tau {tau:.3f}  lr {lr:.2e}  "
                  f"{(time.time() - t0) / (step + 1):.3f} s/step", flush=True)
            log = []
        if (step + 1) % args.save == 0 or step + 1 == steps:
            torch.save({"state": model.state_dict()}, args.out)
            if held:
                evaluate(model, target, held, args.every, args.ks)


# ------------------------------------------------------------------ evaluation

@torch.no_grad()
def evaluate(model: Drafter, target: Target, data: Features, every: int, ks: list[int], per: int = 8, spans: int = 8):
    """Prints tau for K drafts a step (a block of K + 1 rows, as the engine runs it), by the
    bonus token's position, over one anchor per `every` positions of each trace. Replaying overstates the engine's
    tokens per step by 6-12%: the engine's steps skip the positions inside easy runs, and
    land on hard ones more often."""
    model.eval()
    dev = target.head.device
    edges = torch.tensor(BUCKETS, device=dev)
    total = torch.zeros(len(BUCKETS) + 1, len(ks), device=dev)
    count = torch.zeros(len(BUCKETS) + 1, device=dev)
    ctx_len = WINDOW + (per - 1) * every + 1
    for i in range(len(data.files)):
        anchors = list(data.anchors(i)[::every])
        chunks = [(i, anchors[a:a + per]) for a in range(0, len(anchors), per)]
        for a in range(0, len(chunks), spans):
            feat, off, c0, bonus, nxt, live = (x.to(dev) for x in pack(data, chunks[a:a + spans], max(ks) + 1, per, ctx_len))
            bucket = torch.bucketize(c0[:, None] + off, edges, right=True)
            count.index_put_((bucket,), live, accumulate=True)
            for m, k in enumerate(ks):
                logits = draft(model, target, feat[:, :ctx_len + k + 1], off, c0, bonus, k + 1)
                tau = 1 + accepted(logits.argmax(-1), target.index[nxt[..., :k]])
                total[:, m].index_put_((bucket,), tau * live, accumulate=True)
    model.train()
    names = [f"{a // 1024}k-{b // 1024}k" for a, b in zip((0, *BUCKETS), BUCKETS)] + [f"{BUCKETS[-1] // 1024}k+"]
    print(f"  {'position':9s}  anchors  " + "  ".join(f"tau@{k:<2d}" for k in ks))
    for name, t, n in zip([*names, "all"], [*total, total.sum(0)], [*count, count.sum()]):
        if n:
            print(f"  {name:9s}  {int(n):7d}  " + "  ".join(f"{x:6.3f}" for x in (t / n).tolist()), flush=True)


def run_eval(args) -> None:
    dev = device()
    model, target = Drafter.load(args.model, dev), Target(args.target, dev)
    assert torch.equal(model.vocab.cpu(), target.vocab), "--target has another draft vocabulary"
    evaluate(model, target, Features(args.features), args.every, args.ks)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--features", type=Path, nargs="+", required=True, help="directories of feature dumps")
        p.add_argument("--target", type=Path, default=SPEC / "target.pt", help="target.py's file")
        p.add_argument("--every", type=int, default=128, help="evaluation: one anchor per this many positions")
        p.add_argument("--ks", type=lambda s: [int(k) for k in s.split(",")], default=[1, 3, 4, 7],
                       help="evaluation: drafts a step")
    t = sub.choices["train"]
    t.add_argument("--out", type=Path, required=True, help="the checkpoint, written at every --save")
    t.add_argument("--eval", type=Path, nargs="+", help="held-out feature dumps, evaluated at every --save")
    t.add_argument("--init", type=Path, help="a checkpoint to start from")
    t.add_argument("--steps", type=int, help=f"default: {ANCHORS_PER_POSITION} anchors per position of the features")
    t.add_argument("--compile", action="store_true", help="torch.compile the step (CUDA)")
    t.add_argument("--log", type=int, default=500)
    t.add_argument("--save", type=int, default=20000)
    t.add_argument("--seed", type=int, default=0)
    sub.choices["eval"].add_argument("--model", type=Path, required=True, help="a checkpoint")
    args = ap.parse_args()
    {"train": train, "eval": run_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
