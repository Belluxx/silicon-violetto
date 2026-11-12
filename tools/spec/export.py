from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from convert import BF16, quantize, write  # noqa: E402
from drafter import BLOCK, LAYERS, TAPS, WINDOW, H  # noqa: E402

CONV_ROWS = 192  # a correction's 160 rows, padded to the GEMM's 64-row tiles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bf16", action="store_true", help="keep the matrices in bf16")
    args = ap.parse_args()
    sd = torch.load(args.model, map_location="cpu")["state"]
    out: dict[str, torch.Tensor] = {}

    def matrix(name: str, w: torch.Tensor, group: int = 64) -> None:
        w = w.float().contiguous()
        if args.bf16:
            out[name] = w.to(BF16)
        else:
            q, scale = quantize(w, group)
            out[name + ".scale"], out[name] = scale, q

    for k in range(len(TAPS)):
        matrix(f"fc.{k}", sd["fc.weight"][:, k * H:(k + 1) * H])
    out["mask_emb"] = sd["mask_emb"].float().contiguous()
    matrix("inject", sd["inject.weight"])
    for i in range(LAYERS):
        p = f"layers.{i}."
        matrix(f"D{i:02d}.qkv", torch.cat([sd[p + "wq.weight"], sd[p + "wk.weight"], sd[p + "wv.weight"]]))
        matrix(f"D{i:02d}.o", sd[p + "wo.weight"])
        matrix(f"D{i:02d}.gate_up", sd[p + "wgu.weight"])
        matrix(f"D{i:02d}.down", sd[p + "wd.weight"])
        for c in range(4):
            out[f"D{i:02d}.conv{c}.base"] = sd[f"{p}convs.{c}.base"].float().contiguous()
            corr = sd[f"{p}convs.{c}.corr.weight"].float()
            matrix(f"D{i:02d}.conv{c}.corr", torch.cat([corr, corr.new_zeros(CONV_ROWS - len(corr), H)]))
    # Every layer's k | v rows: the context goes through all the layers' at once.
    matrix("kv_all", torch.cat([sd[f"layers.{i}.{w}.weight"] for i in range(LAYERS) for w in ("wk", "wv")]))
    matrix("head", sd["head.weight"] * sd["out_gain"][None], 8)
    out["vocab"] = sd["vocab"].float()
    assert torch.equal(out["vocab"].long(), sd["vocab"])
    out["meta"] = torch.tensor([LAYERS, len(TAPS), BLOCK, WINDOW, len(sd["vocab"]), *TAPS], dtype=torch.float32)
    write(args.out, out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB, {'bf16' if args.bf16 else 'int8'})")


if __name__ == "__main__":
    main()
