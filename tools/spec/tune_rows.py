from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from benchmark import ENGINE, WEIGHTS, write_prompts
from traces import SPEC


def measure(drafter: Path, prompts: Path, context: int, batches: str, rows: str) -> dict[tuple[int, int], tuple[float, float]]:
    """(sequences, rows) -> (tokens per step, ms per step) at a context, from `violetto bench-spec`."""
    run = subprocess.run([ENGINE, "bench-spec", "--weights", WEIGHTS, "--drafter", drafter, "--rows", rows, "--batch", batches,
                          "--ctx", str(context), "--steps", "256", "--temperature", "0.6", "--prompts", prompts],
                         capture_output=True, text=True, check=True)
    cells = {}
    for line in run.stdout.splitlines():
        f = [x.strip() for x in line.strip("| ").split("|")]
        if line.startswith("| ") and f[0].isdigit():
            cells[int(f[0]), int(f[1])] = float(f[5]), float(f[6])
    return cells


def main() -> None:
    ap = argparse.ArgumentParser(description="The best speculative rows by batch and context, for --spec-tokens-short, --spec-kv, --spec-rows-long and --spec-tokens.")
    ap.add_argument("--drafter", type=Path, default=SPEC / "drafter.vlt")
    ap.add_argument("--requests", type=Path, default=SPEC / "eval_requests.txt", help="the held-out requests")
    ap.add_argument("--traces", type=Path, default=SPEC / "eval_traces.jsonl", help="their traces")
    ap.add_argument("--contexts", default="1024,8192,32768")
    ap.add_argument("--batches", default="1,4,8")
    ap.add_argument("--rows", default="2,4,6,8")
    args = ap.parse_args()
    rows = [int(r) for r in args.rows.split(",")]
    batches = [int(n) for n in args.batches.split(",")]
    contexts = [int(c) for c in args.contexts.split(",")]
    with tempfile.TemporaryDirectory() as tmp:
        prompts = Path(tmp) / "prompts.txt"  # cut at the longest context; `bench-spec` cuts them shorter
        assert write_prompts(prompts, args.requests, args.traces, max(contexts), 16), "no held-out trace is long enough"
        grid = {c: measure(args.drafter, prompts, c, args.batches, args.rows) for c in contexts}
    pooled = {}
    for r in rows:
        samples = [(cell[n, r][0], n) for cell in grid.values() for n in batches if n > 1]
        pooled[r] = sum(t * n for t, n in samples) / sum(n for _, n in samples) if samples else 1.0
    print("tokens per step:", {r: round(t, 2) for r, t in pooled.items()})
    print("| sequences x context | " + " | ".join(f"{r} rows" for r in rows) + " |")
    print("|---|" + "---:|" * len(rows))
    for context, cell in grid.items():
        for n in batches:
            speed = {r: pooled[r] * cell[n, 1][1] / cell[n, r][1] for r in rows}
            best = max(speed, key=speed.get)
            print(f"| {n} x {context} | " + " | ".join(f"**{s:.2f}x**" if r == best else f"{s:.2f}x" for r, s in speed.items()) + " |")


if __name__ == "__main__":
    main()
