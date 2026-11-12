from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from collections import Counter
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "limite-1b-violetto"
SPEC = ROOT / "data" / "spec"
PROBLEMS = ROOT / "data" / "problems"
# limite-evals' AIME instruction, prepended to the problem.
INSTRUCTION = "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}.\n\n"
# NuminaMath-1.5's sources, by their share of the problems: competition problems of every
# kind (proofs and multiple choice too) and some easier ones. Its synthetic problems are
# left out.
NUMINA_MIX = {"olympiads": 0.34, "aops_forum": 0.24, "cn_contest": 0.10, "amc_aime": 0.08,
              "olympiads_ref": 0.05, "inequalities": 0.04, "number_theory": 0.04, "cn_k12": 0.11}
# The eval sets, at pinned revisions: (dataset, revision, cache file, samples in the held-out
# set). No problem of theirs is trained on.
EVAL_SETS = {
    "aime": ("MathArena/aime_2026", "10b4e45b7a503075d4da8a0d57916a4f06ce6bd2", "aime/aime_2026.json", 2),
    "apex": ("MathArena/apex-shortlist", "f3efdf224ef665f129ddaae37699f6098c65781b", "apex/apex_shortlist.json", 1),
    "hmmt": ("MathArena/hmmt_feb_2026", "02fba4f74d8e68e73e66a02d540fd979c05c274c", "hmmt26/hmmt_feb_2026.json", 0),
}
EVAL_TOKENS = 131072 - 4096  # the eval protocol's budget
NUMINA_SHARE = 11000 / 12250  # of the training problems; the others are AIME 1983-2024's and Omni-MATH's
OMNI_DIFFICULTY = 4.0  # Omni-MATH's problems from this difficulty up


def numina_tokens(j: int) -> int:
    """The budget of the j-th NuminaMath problem: 3 of 11 run to 32768 tokens, where 68% of
    traces finish."""
    return 32768 if 3 * j % 11 < 3 else 4096


def hard_tokens(j: int) -> int:
    """The budget of the j-th AIME or Omni-MATH problem: 1 of 5 runs to the eval's."""
    return EVAL_TOKENS if j % 5 == 0 else 8192


# ------------------------------------------------------------------ files

def read_requests(paths: list[Path]) -> dict[str, list[int]]:
    """Request files (`name options id id ...` per line): name -> prompt ids."""
    out = {}
    for path in paths:
        for line in open(path):
            f = line.split()
            if f:
                out[f[0]] = [int(x) for x in f[2:]]
    return out


def read_traces(paths: list[Path]) -> dict[str, list[int]]:
    """`violetto generate` outputs (one JSON object per line): name -> generated ids."""
    out = {}
    for path in paths:
        for line in open(path):
            if line.strip():
                d = json.loads(line)
                out[d["name"]] = d["tokens"]
    return out


# ------------------------------------------------------------------ prompts

@cache
def template():
    from jinja2 import Environment

    env = Environment()
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
    return env.from_string((MODEL / "chat_template.jinja").read_text())


def chat_ids(tok, problem: str) -> list[int]:
    """The eval protocol's prompt for a problem, through the model's chat template."""
    text = template().render(messages=[{"role": "user", "content": INSTRUCTION + problem}], add_generation_prompt=True)
    return tok.encode(text, add_special_tokens=False).ids


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


# ------------------------------------------------------------------ problems

def eval_problems(name: str) -> list[dict]:
    """An eval set's problems ({"idx", "problem", "gold"}), from data/problems or, the first
    time, from Hugging Face."""
    dataset, revision, cache_file, _ = EVAL_SETS[name]
    path = PROBLEMS / cache_file
    if not path.exists():
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        parquet = hf_hub_download(dataset, "data/train-00000-of-00001.parquet", revision=revision, repo_type="dataset")
        rows = sorted(pq.read_table(parquet).to_pylist(), key=lambda r: int(r["problem_idx"]))
        problems = [{"idx": int(r["problem_idx"]), "problem": r["problem"], "gold": str(r["answer"]).strip()} for r in rows]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"dataset": dataset, "revision": revision, "problems": problems}, indent=1))
    return json.loads(path.read_text())["problems"]


def in_eval_sets(key: str) -> bool:
    """Whether a normalized problem is an eval problem, or the start of one: the same
    problem with another ending."""
    return any(key[:200] in e for e in eval_keys())


@cache
def eval_keys() -> list[str]:
    return [norm(p["problem"]) for name in EVAL_SETS for p in eval_problems(name)]


def fetch(path: Path, repo: str, file: str) -> Path:
    """A problem file, from the Hugging Face dataset the first time."""
    if not path.exists():
        from huggingface_hub import hf_hub_download

        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(hf_hub_download(repo, file, repo_type="dataset"), path)
    return path


def hard_problems(count: int) -> list[tuple[str, int, str]]:
    """The first `count` problems of AIME 1983-2024 and Omni-MATH, in a fixed shuffle of
    both: (id, seed, text). Once each, and none from an eval set."""
    with open(fetch(PROBLEMS / "aime_1983_2024.csv", "gneubig/aime-1983-2024", "AIME_Dataset_1983_2024.csv")) as f:
        pool = [(f"aime{r['ID']}", r["Question"]) for r in csv.DictReader(f)]
    with open(fetch(PROBLEMS / "omni_math.jsonl", "KbsdJames/Omni-MATH", "test.jsonl")) as f:
        omni = [json.loads(line) for line in f]
    pool += [(f"omni{i}", d["problem"]) for i, d in enumerate(omni) if d["difficulty"] >= OMNI_DIFFICULTY]
    seen, out = set(), []
    for seed, (name, text) in enumerate(pool):
        key = norm(text)
        if key not in seen and not in_eval_sets(key):
            seen.add(key)
            out.append((name, seed, text))
    random.Random("hard").shuffle(out)
    return out[:count]


def numina_problems(count: int) -> list[tuple[str, int, str]]:
    """The first `count` problems of NuminaMath-1.5, in a fixed order: (id, seed, text). Its
    real, valid problems of NUMINA_MIX's sources, once each, none from an eval set or among
    hard_problems (Omni-MATH's are olympiad problems too); each source shuffled once, then
    all interleaved by their shares."""
    import pyarrow.parquet as pq

    if not list((PROBLEMS / "numina").rglob("*.parquet")):
        from huggingface_hub import snapshot_download

        snapshot_download("AI-MO/NuminaMath-1.5", repo_type="dataset", allow_patterns="data/*.parquet",
                          local_dir=PROBLEMS / "numina")
    hard = {norm(text)[:200] for _, _, text in hard_problems(1 << 30)}
    by_source: dict[str, list[tuple[int, str]]] = {src: [] for src in NUMINA_MIX}
    seen, row = set(), -1
    for path in sorted((PROBLEMS / "numina").rglob("*.parquet"), key=lambda p: p.name):
        for r in pq.read_table(path, columns=["problem", "source", "synthetic", "problem_is_valid"]).to_pylist():
            row += 1
            key = norm(r["problem"])
            if r["synthetic"] or r["problem_is_valid"] != "Yes" or r["source"] not in NUMINA_MIX or key in seen:
                continue
            seen.add(key)
            if key[:200] not in hard and not in_eval_sets(key):
                by_source[r["source"]].append((row, r["problem"]))
    ranked = []
    for src, problems in by_source.items():
        random.Random(src).shuffle(problems)
        ranked += [((j + 0.5) / NUMINA_MIX[src], row, text) for j, (row, text) in enumerate(problems)]
    ranked.sort()
    return [(f"numina{row}", row, text) for _, row, text in ranked[:count]]


def main() -> None:
    from tokenizers import Tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", type=int, default=12250, help="training problems, one trace each (a larger number only adds requests)")
    ap.add_argument("--out", type=Path, default=SPEC, help="the directory of the request files")
    args = ap.parse_args()
    tok = Tokenizer.from_file(str(MODEL / "tokenizer.json"))
    args.out.mkdir(parents=True, exist_ok=True)

    def write(file: str, requests: list[tuple[str, int, int, str]]) -> None:
        """(name, seed, most new tokens, problem) per request."""
        lines = [f"{name} seed={seed},max_tokens={tokens} {' '.join(map(str, chat_ids(tok, text)))}\n"
                 for name, seed, tokens, text in requests]
        (args.out / file).write_text("".join(lines))
        budgets = sorted(Counter(r[2] for r in requests).items())
        print(f"{args.out / file}: " + ", ".join(f"{n} requests of up to {tokens} tokens" for tokens, n in budgets))

    numina = round(NUMINA_SHARE * args.problems)
    train = [(f"{name}_s0", seed, numina_tokens(j), text) for j, (name, seed, text) in enumerate(numina_problems(numina))]
    train += [(f"{name}_s0", seed, hard_tokens(j), text)
              for j, (name, seed, text) in enumerate(hard_problems(args.problems - numina))]
    write("train_requests.txt", [r for r in train if r[2] < EVAL_TOKENS])
    write("train_long_requests.txt", [r for r in train if r[2] == EVAL_TOKENS])
    write("eval_requests.txt", [(f"{name}{p['idx']}_s{k}", k, EVAL_TOKENS, p["problem"])
                                for name, (*_, samples) in EVAL_SETS.items() for p in eval_problems(name) for k in range(samples)])


if __name__ == "__main__":
    main()
