from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from pathlib import Path

from traces import ROOT, SPEC, read_requests, read_traces

ENGINE = ROOT / "target" / "release" / "violetto"
WEIGHTS = ROOT / "models" / "limite-int8.vlt"
NEW_TOKENS = 1024
LONG = 32768  # shorter contexts cut the traces that reach this one
# Name: sequences at a time, context, prompts.
WORKLOADS = {
    "1x1024": (1, 1024, 16),
    "1x4096": (1, 4096, 16),
    "1x16384": (1, 16384, 8),
    "1x32768": (1, 32768, 8),
    "1x65536": (1, 65536, 4),
    "4x4096": (4, 4096, 16),
    "8x1024": (8, 1024, 16),
}


def ends_in_a_loop(tokens: list[int], tail: int = 1024, n: int = 8) -> bool:
    """Whether nearly all of the last `tail` tokens' n-grams already appear before them. Such
    prompts are left out: every draft matches a loop, which says nothing about the drafter."""
    seen = {tuple(tokens[i:i + n]) for i in range(len(tokens) - tail - n)}
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - tail, len(tokens) - n)]
    return sum(g in seen for g in grams) > 0.9 * len(grams)


def write_prompts(path: Path, requests: Path, traces: Path, context: int, count: int) -> int:
    """A request file of up to `count` prompts of `context` tokens: held-out traces (prompt,
    then output) of at least max(context, LONG) + NEW_TOKENS tokens, one per problem, cut at
    the context. Returns how many."""
    prompts, outputs = read_requests([requests]), read_traces([traces])
    lines, problems = [], set()
    for name, output in outputs.items():
        ids = prompts[name] + output
        problem = name.rsplit("_s", 1)[0]  # its other samples are left out
        if len(ids) < max(context, LONG) + NEW_TOKENS or problem in problems or ends_in_a_loop(ids[:context]):
            continue
        problems.add(problem)
        lines.append(f"{name} seed={len(lines)} {' '.join(map(str, ids[:context]))}\n")
        if len(lines) == count:
            break
    path.write_text("".join(lines))
    return len(lines)


def generate(requests: Path, sequences: int, tokens: int, flags: list[str]) -> int:
    """tok/s of one `violetto generate` run."""
    run = subprocess.run([ENGINE, "generate", "--weights", WEIGHTS, "--requests", requests, "--max-seqs", str(sequences),
                          "--max-tokens", str(tokens), "--stop", "", "--out", requests.with_suffix(".jsonl"), *flags],
                         capture_output=True, text=True, check=True)
    return int(re.search(r"; (\d+) tok/s", run.stdout + run.stderr)[1])


def batched(drafter: Path, prompts: Path) -> tuple[int, int]:
    """Plain and drafted tok/s of 16 sequences at 32k, from `violetto bench-spec`'s table."""
    run = subprocess.run([ENGINE, "bench-spec", "--weights", WEIGHTS, "--drafter", drafter, "--rows", "2", "--batch", "16",
                          "--ctx", "32768", "--steps", "256", "--temperature", "0.6", "--prompts", prompts],
                         capture_output=True, text=True, check=True)
    rows = [line.split("|") for line in run.stdout.splitlines() if line.startswith("| 16 ")]
    return int(rows[0][4]), int(rows[1][4])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", type=Path, default=SPEC / "drafter.vlt")
    ap.add_argument("--requests", type=Path, default=SPEC / "eval_requests.txt", help="the held-out requests")
    ap.add_argument("--traces", type=Path, default=SPEC / "eval_traces.jsonl", help="their traces")
    ap.add_argument("--only", nargs="+", help="workloads to run (default: all)")
    args = ap.parse_args()
    print("| workload | prompts | plain tok/s | drafter tok/s | speedup |\n|---|---:|---:|---:|---:|", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        prompts = Path(tmp) / "prompts.txt"
        for name in args.only or [*WORKLOADS, "aime", "16x32768"]:
            if name == "aime":
                sequences, tokens, count = 16, 4096, 32
                first = args.requests.read_text().split("\n", 1)[0].split()[2:]
                prompts.write_text("".join(f"a{seed} seed={seed} {' '.join(first)}\n" for seed in range(count)))
            else:
                sequences, context, most = WORKLOADS.get(name, (16, 32768, 16))
                tokens, count = NEW_TOKENS, write_prompts(prompts, args.requests, args.traces, context, most)
            if count == 0:
                print(f"| {name} | 0 | | | no held-out trace is long enough |", flush=True)
                continue
            if name == "16x32768":
                plain, drafted = batched(args.drafter, prompts)
            else:
                plain = generate(prompts, sequences, tokens, [])
                drafted = generate(prompts, sequences, tokens, ["--drafter", str(args.drafter)])
            print(f"| {name} | {count} | {plain} | {drafted} | {drafted / plain:.2f}x |", flush=True)

if __name__ == "__main__":
    main()
