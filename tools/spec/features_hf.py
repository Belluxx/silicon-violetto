from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from drafter import EPS, TAPS, H
from traces import MODEL, read_requests, read_traces


@torch.inference_mode()
def features(model, ids: list[int], chunk: int, device: str) -> torch.Tensor:
    """bf16 [len(ids), len(TAPS), H], on the CPU."""
    from transformers import DynamicCache

    caught: dict[int, torch.Tensor] = {}

    def hook(layer: int):
        def f(_module, _inputs, output):
            caught[layer] = output[0] if isinstance(output, tuple) else output
        return f

    handles = [model.model.layers[t].register_forward_hook(hook(t)) for t in TAPS]
    t_ids = torch.tensor(ids, device=device)
    out = torch.empty(len(ids), len(TAPS), H, dtype=torch.bfloat16)
    cache = DynamicCache(config=model.config)
    try:
        a = 0
        while a < len(ids):
            # Shorter chunks deep in a trace: a chunk's attention masks and scores grow
            # with the keys behind it.
            b = min(a + (chunk if a < 16384 else max(512, chunk // 4) if a < 49152 else 512), len(ids))
            model.model(input_ids=t_ids[None, a:b], past_key_values=cache, use_cache=True)
            for k, t in enumerate(TAPS):
                out[a:b, k] = F.rms_norm(caught[t][0].float(), (H,), eps=EPS).to(torch.bfloat16).cpu()
            a = b
    finally:
        for h in handles:
            h.remove()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=Path, nargs="+", required=True)
    ap.add_argument("--traces", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", default="0/1", help="i/n: every n-th trace, from the i-th")
    ap.add_argument("--mem-fraction", type=float, default=0.0, help="this process's share of the GPU's memory")
    ap.add_argument("--chunk", type=int, default=4096, help="positions a forward pass, early in a trace")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "mps")
    ap.add_argument("--model", type=Path, default=MODEL, help="the HF checkpoint's directory")
    args = ap.parse_args()

    requests, traces = read_requests(args.requests), read_traces(args.traces)
    ids = {name: requests[name] + traces[name] for name in traces if name in requests}
    i, n = (int(x) for x in args.shard.split("/"))
    names = sorted(ids, key=lambda name: (-len(ids[name]), name))[i::n]
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mem_fraction:
        torch.cuda.set_per_process_memory_fraction(args.mem_fraction)

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa")
    model = model.to(args.device).eval()  # eval() after the move: it derives the folded inference constants
    t0, done = time.time(), 0
    for c, name in enumerate(names):
        path = args.out / f"{name}.safetensors"
        if path.exists():
            continue
        feat = features(model, ids[name], args.chunk, args.device)
        tmp = path.with_suffix(".tmp")
        save_file({"tokens": torch.tensor(ids[name], dtype=torch.uint32),
                   "prompt_len": torch.tensor([len(requests[name])], dtype=torch.uint32),
                   "taps": torch.tensor(TAPS, dtype=torch.uint32),
                   "feat": feat}, tmp)
        tmp.rename(path)
        done += len(ids[name])
        print(f"{c + 1}/{len(names)} {name}: {len(ids[name])} tokens ({done / (time.time() - t0):.0f} tok/s)", flush=True)
    print(f"done: {done} tokens in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
