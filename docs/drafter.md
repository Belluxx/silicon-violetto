# The drafter

A small model that drafts the next tokens of [Limite 1B Violetto](https://huggingface.co/paradigma-inc/limite-1b-violetto)
so that the engine can verify several of them in one step.

The trained weights are on Hugging Face as [Belluxx/limite-1b-violetto-drafter](https://huggingface.co/Belluxx/limite-1b-violetto-drafter):

```bash
uv run hf download Belluxx/limite-1b-violetto-drafter drafter.vlt --local-dir data/spec
./target/release/violetto serve --weights models/limite-int8.vlt --drafter data/spec/drafter.vlt
```

**Important**: remember to raise `--max-seqs` (which is the defaulted to `1`) depending on your workload.

## Architecture

In large part taken from the [DFlash](https://arxiv.org/abs/2602.06036) family.
It reads from layers 35 and 47 (fused by one projection) of Limite Violetto.

Details:
- Reading also layer 35 other than the last layer was the biggest improvement for first drafted token.
- Training anchors come from anywhere in traces of up to 126k tokens and acceptance doesn't seem to fall at long contexts.
- GPU draft trees (from [DDTree](https://arxiv.org/abs/2604.12989)): show 5% to 15% speed improvement compared to a single chain of drafts (but only for single stream decode).
- Uses DFlash-2's two-tap convolutions

### Acceptance

Tokens per step on held-out traces, for K drafts a step, by position in the trace:

| K | 0-8k | 8-32k | 32-64k | 64k+ | All |
|---|---:|---:|---:|---:|---:|
| 3 | 2.71 | 2.65 | 2.65 | 2.90 | 2.70 |
| 4 | 2.97 | 2.90 | 2.88 | 3.23 | 2.96 |
| 7 | 3.33 | 3.20 | 3.17 | 3.75 | 3.30 |

## Training data

The drafter learns from Limite's own writing: 12k reasoning traces, 103M positions, generated
by this engine at T = 0.6 and top-p 0.95.

| Problems | Traces | Up to |
|---|---:|---:|
| [NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5), competition problems | 8,000 | 4,096 tokens |
| NuminaMath-1.5 | 3,000 | 32,768 tokens |
| [AIME 1983-2024](https://huggingface.co/datasets/gneubig/aime-1983-2024) and [Omni-MATH](https://huggingface.co/datasets/KbsdJames/Omni-MATH) | 1,000 | 8,192 tokens |
| AIME 1983-2024 and Omni-MATH | 216 | 126,976 tokens |

Two things helped a lot: many different problems, and traces near max length.
More training on the same traces didn't help nearly as much.

The loss is the KL from Limite's distribution at T = 0.6.
Blocks of 2 to 8 rows, 400M anchors, learning rate 4e-4: 3.7 hours on one A100.

## Train your own

`tools/spec` has the whole pipeline, `traces.py` builds the traces dataset mix above.
Generating the traces takes about a day on an M4 Max.

```bash
# Locally on Mac: requests, traces (resumable)
uv run tools/spec/traces.py
./target/release/violetto generate --weights models/limite-int8.vlt --max-seqs 32 --resume \
    --requests data/spec/train_requests.txt --out data/spec/train_traces.jsonl
./target/release/violetto generate --weights models/limite-int8.vlt --max-seqs 16 --resume \
    --requests data/spec/train_long_requests.txt --out data/spec/train_long_traces.jsonl
./target/release/violetto generate --weights models/limite-int8.vlt --max-seqs 16 --resume \
    --requests data/spec/eval_requests.txt --out data/spec/eval_traces.jsonl
uv run tools/spec/target.py       # data/spec/target.pt

# On CUDA, with tools/, data/spec and the HF checkpoint in models/limite-1b-violetto: the target's features of every trace, then training
python tools/spec/features_hf.py --out feat/train \
    --requests data/spec/train_requests.txt data/spec/train_long_requests.txt \
    --traces data/spec/train_traces.jsonl data/spec/train_long_traces.jsonl
python tools/spec/features_hf.py --out feat/eval \
    --requests data/spec/eval_requests.txt --traces data/spec/eval_traces.jsonl
python tools/spec/train.py train --features feat/train --eval feat/eval --out drafter.pt --compile
python tools/spec/train.py eval --model drafter.pt --features feat/eval

# Locally on Mac: export and test
uv run tools/spec/export.py --model drafter.pt --out data/spec/mine.vlt
uv run tools/spec/check_engine.py
uv run tools/spec/benchmark.py --drafter data/spec/mine.vlt
```

- For several GPUs `features_hf.py --shard i/n` splits the traces over processes, and `--mem-fraction` caps each one's memory.
- On a host with many cores set `OMP_NUM_THREADS` to 2-4.
- `--compile` needs gcc, g++ and python3-dev.
- For another M chip, `tools/spec/tune_rows.py` measures the best rows per step for it (took about 12 minutes on an M4 Max)

## References

- DFlash, [arXiv 2602.06036](https://arxiv.org/abs/2602.06036)
- DFlash 2, [Inco AI blog](https://inco.ai/blog/dflash2)
- DDTree, [arXiv 2604.12989](https://arxiv.org/abs/2604.12989)
