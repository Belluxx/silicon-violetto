from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from convert import read  # noqa: E402
from drafter import TAPS, TEMP, Drafter, rms  # noqa: E402

ENGINE = ROOT / "target" / "release" / "violetto"
SPEC = ROOT / "data" / "spec"
CANDS = 4  # drafter.rs: candidates per drafted row


def draft_tree(logits: torch.Tensor, vocab: torch.Tensor) -> list[tuple[int, int, int, int]]:
    """kernels.metal's draft_top and draft_tree on a block's logits [rows, V]: rows 1.. as
    (token, parent, depth, the rows it sees), best-first by the product of the drafter's
    probabilities along a path."""
    rows = logits.shape[0]
    top = torch.softmax(logits.double() / TEMP, -1).topk(CANDS, -1)
    ids = vocab[top.indices]
    score, depth, taken, sees, out = [1.0], [0], [0], [1], []
    while len(score) < rows:
        best, parent = -1.0, -1
        for i in range(len(score)):
            if depth[i] + 1 < rows and taken[i] < CANDS:
                s = score[i] * top.values[depth[i] + 1, taken[i]].item()
                if s > best:
                    best, parent = s, i
        if parent < 0:
            break
        d, n = depth[parent] + 1, len(score)
        out.append((int(ids[d, taken[parent]]), parent, d, sees[parent] | 1 << n))
        taken[parent] += 1
        score.append(best), depth.append(d), taken.append(0), sees.append(sees[parent] | 1 << n)
    return out


def random_drafter() -> Drafter:
    """A drafter with nothing at its initial zero or identity, in weights bf16 keeps exactly.
    Its logits are small: what only a trained drafter's logits of 100 and more would break
    doesn't show here."""
    torch.manual_seed(0)
    model = Drafter(torch.randperm(151667)[:16384].sort().values)
    with torch.no_grad():
        model.out_gain.uniform_(0.5, 1.5)
        model.inject.weight.normal_(0.0, 0.05)
        for layer in model.layers:
            for conv in layer.convs:
                conv.base.add_(torch.randn_like(conv.base) * 0.3)
                conv.corr.weight.normal_(0.0, 0.05)
        for p in model.parameters():
            p.mul_(0.5).copy_(p.bfloat16().float())
    return model.eval()


def run(*cmd) -> None:
    subprocess.run([str(c) for c in cmd], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8, help="the block's rows")
    ap.add_argument("--n", type=int, default=4, help="prompts to check")
    ap.add_argument("--weights", type=Path, default=ROOT / "models" / "limite-int8.vlt")
    ap.add_argument("--requests", type=Path, default=SPEC / "eval_requests.txt", help="traces.py's held-out set")
    args = ap.parse_args()
    work = Path(tempfile.mkdtemp(prefix="drafter-check-"))

    # The first prompts that differ: the held-out set has two samples of some problems.
    lines = list({line.split(None, 2)[2]: line for line in args.requests.read_text().splitlines() if line.strip()}.values())[:args.n]
    (work / "requests.txt").write_text("\n".join(lines) + "\n")
    model = random_drafter()
    torch.save({"state": model.state_dict()}, work / "drafter.pt")
    run(sys.executable, ROOT / "tools" / "spec" / "export.py", "--model", work / "drafter.pt", "--out", work / "drafter.vlt", "--bf16")
    run(ENGINE, "drafter-check", "--weights", args.weights, "--drafter", work / "drafter.vlt",
        "--requests", work / "requests.txt", "--rows", args.rows, "--out", work / "engine.json")
    engine = json.loads((work / "engine.json").read_text())
    # The features of each prompt and its bonus token.
    (work / "traces.jsonl").write_text("".join(
        json.dumps({"name": e["name"], "finish": "length", "prompt_tokens": 0, "tokens": [e["bonus"]]}) + "\n" for e in engine))
    run(ENGINE, "features", "--weights", args.weights, "--requests", work / "requests.txt", "--traces", work / "traces.jsonl",
        "--out", work / "feat", "--taps", ",".join(map(str, TAPS)))

    embed = read(args.weights)["embed"]
    worst, differ = 0.0, 0
    for e in engine:
        f = load_file(work / "feat" / f"{e['name']}.safetensors")
        p = int(f["prompt_len"][0])  # the bonus token's position: its block sees the prompt
        with torch.no_grad():
            ctx = model.context(f["feat"][None, :p])
            bonus = rms(embed[torch.tensor([[e["bonus"]]])].float())
            ref = model.head(model(ctx, torch.tensor([0]), bonus, torch.tensor([[p]]), args.rows))[0, 0]
        got = torch.tensor(e["logits"]).view(args.rows, -1)
        err = ((got - ref).abs().max() / ref.abs().max()).item()
        tree_ref = draft_tree(ref, model.vocab)
        tree_eng = [(e["feed"][i], (n >> 8) & 0xFF, n & 0xFF, n >> 16) for i, n in enumerate(e["tree"][1:args.rows], 1)]
        same = sum(a == b for a, b in zip(tree_ref, tree_eng))
        worst, differ = max(worst, err), differ + len(tree_ref) - same
        print(f"  {e['name']:8s} prompt {p:3d} bonus {e['bonus']:6d}: logits rel err {err:.2e}; tree rows "
              f"{same}/{len(tree_ref)} equal (token, parent, depth: engine {[r[:3] for r in tree_eng]}, "
              f"torch {[r[:3] for r in tree_ref]})")
    print(f"worst logits error {worst:.2e}; {differ} tree rows differ  ({work})")


if __name__ == "__main__":
    main()
