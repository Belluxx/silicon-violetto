from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from convert import head_weight, read  # noqa: E402
from traces import SPEC, read_requests, read_traces  # noqa: E402

VOCAB = 151667  # the tokenizer's ids
STOPS = (151643, 151645)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=Path, nargs="+", default=[SPEC / "train_requests.txt", SPEC / "train_long_requests.txt"])
    ap.add_argument("--traces", type=Path, nargs="+", default=[SPEC / "train_traces.jsonl", SPEC / "train_long_traces.jsonl"])
    ap.add_argument("--weights", type=Path, default=ROOT / "models" / "limite-int8.vlt")
    ap.add_argument("--size", type=int, default=16384, help="the draft vocabulary's ids: the drafter drafts no others")
    ap.add_argument("--out", type=Path, default=SPEC / "target.pt")
    args = ap.parse_args()

    traces = read_traces(args.traces)
    counts = Counter()
    for tokens in traces.values():
        counts.update(tokens)
    prompts = Counter(t for name, ids in read_requests(args.requests).items() if name in traces for t in ids)
    ids = dict.fromkeys([*STOPS, *(t for t, _ in counts.most_common()), *(t for t, _ in prompts.most_common()), *range(VOCAB)])
    vocab = torch.tensor(sorted(list(ids)[:args.size]), dtype=torch.long)
    covered = sum(counts[t] for t in vocab.tolist()) / counts.total()
    print(f"{len(traces)} traces, {counts.total()} generated tokens, {len(counts)} distinct ids; "
          f"the {len(vocab)} kept cover {100 * covered:.2f}%")

    weights = read(args.weights)
    torch.save({"vocab": vocab, "embed": weights["embed"].clone(), "head": head_weight(weights)[vocab].clone()}, args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
