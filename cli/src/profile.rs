use violetto::{skip, Engine, EngineConfig, Matmul, Run, SeqId, FFN, HIDDEN, N_LAYERS, QKV, VOCAB};

use crate::bench::{decode, ids, median, medians, prefill, PREFILL};
use crate::{EngineArgs, List, Res, SamplingArgs};

#[derive(clap::Args)]
pub struct Args {
    /// Context of each sequence
    #[arg(long, default_value_t = 1024)]
    ctx: usize,
    /// Sequences
    #[arg(long, default_value_t = 1)]
    n: usize,
    /// Decode steps timed
    #[arg(long, default_value_t = 64)]
    steps: usize,
    /// Time one prefill step of this many tokens per sequence, instead of decode steps
    #[arg(long, default_value_t = 0)]
    chunk: usize,
    /// Compute the context, instead of filling it: timing doesn't depend on the KV values
    #[arg(long)]
    prefill: bool,
    /// Only the groups whose names start with one of these, e.g. attention,head
    #[arg(long)]
    groups: Option<List<String>>,
    #[command(flatten)]
    sampling: SamplingArgs,
    #[command(flatten)]
    engine: EngineArgs,
}

const GROUPS: [(&str, u32); 9] = [
    ("qkv matmul", skip::QKV),
    ("qkv_post", skip::POST),
    ("attention", skip::ATTN),
    ("attn reduce", skip::REDUCE),
    ("o matmul", skip::O),
    ("gate_up matmul", skip::GATE_UP),
    ("down matmul", skip::DOWN),
    ("mudd", skip::MUDD),
    ("head + sampling", skip::HEAD),
];

pub fn profile(a: &Args) -> Res {
    let (ctx, n, steps, chunk) = (a.ctx, a.n, a.steps, a.chunk);
    let cfg = EngineConfig {
        max_seqs: n.max(2),
        max_step_tokens: 512.max(n * chunk),
        max_chunk: 512.max(chunk),
        global_kv_tokens: n * (ctx + steps.max(chunk) + 256) + 1024,
        max_seq_len: ctx + steps.max(chunk) + 1,
        max_head_rows: n.max(8),
    };
    let mut eng = a.engine.load(cfg)?;
    // GPU ms of a step with the kernel groups of `mask` left out.
    let once = |eng: &mut Engine, mask: u32| -> Res<f64> {
        let seqs: Vec<SeqId> = (0..n).map(|_| eng.new_seq()).collect::<Result<_, _>>()?;
        for (j, &s) in seqs.iter().enumerate() {
            match a.prefill {
                true => drop(prefill(eng, s, &ids(ctx, 5 + j as u64), 512)?),
                false => eng.bench_fill(s, ctx)?,
            }
        }
        eng.set_profile_skip(mask);
        let ms = if chunk > 0 {
            let toks = ids(chunk, 9);
            let runs: Vec<Run> = seqs.iter().map(|&s| Run { seq: s, tokens: &toks }).collect();
            let ms = eng.step(&runs, &PREFILL)?.gpu_seconds * 1e3;
            seqs.into_iter().for_each(|s| eng.free_seq(s));
            ms
        } else {
            medians(&decode(eng, &a.sampling, &seqs, &vec![100; n], steps)?.1).0
        };
        eng.set_profile_skip(0);
        Ok(ms)
    };
    // A prefill step is a single sample: the median of 5.
    let time = |eng: &mut Engine, mask: u32| -> Res<f64> {
        let v = (0..if chunk > 0 { 5 } else { 1 }).map(|_| once(eng, mask)).collect::<Res<Vec<f64>>>()?;
        Ok(median(&v))
    };
    time(&mut eng, 0)?;
    let full = time(&mut eng, 0)?;
    match chunk {
        0 => println!("decode step, {n} sequence(s) at context {ctx}: {full:.3} ms GPU"),
        _ => println!("prefill step of {chunk} tokens x {n} sequence(s) at context {ctx}: {full:.3} ms GPU"),
    }
    let all = GROUPS.iter().fold(0, |m, g| m | g.1);
    let none = time(&mut eng, all)?;
    println!("| group | full minus group, ms | group alone, ms | share (alone) |\n|---|---:|---:|---:|");
    let (mut sum_minus, mut sum_alone) = (0.0, 0.0);
    for (name, mask) in GROUPS {
        if a.groups.as_ref().is_some_and(|only| !only.0.iter().any(|g| name.starts_with(g.as_str()))) {
            continue;
        }
        let minus = full - time(&mut eng, mask)?;
        let alone = time(&mut eng, all & !mask)? - none;
        (sum_minus, sum_alone) = (sum_minus + minus, sum_alone + alone);
        println!("| {name} | {minus:.3} | {alone:.3} | {:.1}% |", 100.0 * alone / full);
    }
    println!("| (embed, final norm; everything skipped) | | {none:.3} | |");
    println!("| sum of groups | {sum_minus:.3} | {sum_alone:.3} | |");
    Ok(())
}

#[derive(clap::Args)]
pub struct MatmulArgs {
    /// Token counts
    #[arg(long, default_value = "1,2,4,8,16,32,64,128,512")]
    n: List<usize>,
    /// Reuse layer 0's matrix, so that the weights come from the cache
    #[arg(long)]
    same: bool,
    #[command(flatten)]
    engine: EngineArgs,
}

pub fn matmul(a: &MatmulArgs) -> Res {
    let cfg = EngineConfig { max_seqs: 1, max_step_tokens: 512, max_chunk: 512, global_kv_tokens: 1024, max_seq_len: 1024, max_head_rows: 64 };
    let mut eng = a.engine.load(cfg)?;
    let kinds = [
        ("qkv", QKV * HIDDEN, N_LAYERS),
        ("o", HIDDEN * HIDDEN, N_LAYERS),
        ("gate_up", 2 * FFN * HIDDEN, N_LAYERS),
        ("down", HIDDEN * FFN, N_LAYERS),
        ("head", VOCAB * HIDDEN, 1),
    ];
    // Warm-up: the first kind measured would otherwise pay for the GPU clocking up.
    for _ in 0..4 {
        eng.bench_matmul("gate_up", 1, Matmul::Gemv { rows: 2, splits: 1 }, 8, a.same)?;
    }
    println!("| matmul | n | variant | ms | GB/s (weights) | TFLOP/s |\n|---|---:|---|---:|---:|---:|");
    for (kind, params, layers) in kinds {
        for &n in a.n.0.iter().filter(|&&n| kind != "head" || n <= 64) {
            let mut variants = Vec::new();
            if n <= 8 {
                // The SwiGLU epilogue needs a (gate, up) row pair per simdgroup.
                let rows = if kind == "gate_up" { &[2, 4][..] } else { &[1, 2, 4][..] };
                let splits = if n == 1 { &[1, 2, 4][..] } else { &[1][..] };
                variants.extend(rows.iter().flat_map(|&rows| splits.iter().map(move |&splits| Matmul::Gemv { rows, splits })));
            }
            for tiles in [1, 2].into_iter().filter(|&t| 8 * t >= n) {
                for rows in [1, 2] {
                    variants.extend([4, 8].map(|splits| Matmul::Stream { rows, tiles, splits }));
                }
            }
            for tile in [16, 32, 64].into_iter().filter(|&t| t >= n || t == 64) {
                variants.extend([1, 2, 4].map(|k_tiles| Matmul::Gemm { tile, k_tiles }));
            }
            for v in variants {
                let (t, bytes) = eng.bench_matmul(kind, n, v, 8, a.same)?;
                let label = match v {
                    Matmul::Gemv { rows, splits: 1 } => format!("gemv r{rows}"),
                    Matmul::Gemv { rows, splits } => format!("gemv r{rows} s{splits}"),
                    Matmul::Gemm { tile, k_tiles } => format!("gemm m{tile} p{k_tiles}"),
                    Matmul::Stream { rows, tiles, splits } => format!("stream r{rows} t{tiles} s{splits}"),
                };
                let flops = (2 * params * layers * n) as f64;
                println!("| {kind} | {n} | {label} | {:.3} | {:.0} | {:.2} |", t * 1e3, bytes as f64 / t / 1e9, flops / t / 1e12);
            }
        }
    }
    Ok(())
}
