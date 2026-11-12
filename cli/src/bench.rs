use std::path::PathBuf;
use std::time::Instant;

use serde_json::json;
use violetto::{Draft, Engine, EngineConfig, Event, Prompt, Request, Rows, Run, SeqId, StepOptions, StepStats};

use crate::{files, EngineArgs, List, Res, SamplingArgs};

#[derive(clap::Args)]
pub struct Args {
    /// Contexts of the single-stream decode table
    #[arg(long, default_value = "128,1024,4096,16384,32768")]
    decode_ctx: List<usize>,
    /// Prompt lengths of the prefill table
    #[arg(long, default_value = "128,512,2048,8192")]
    prefill: List<usize>,
    /// Sequence counts of the batched decode table
    #[arg(long, default_value = "1,2,4,8,16,32")]
    batch: List<usize>,
    /// Context of the batched decode table
    #[arg(long, default_value_t = 1024)]
    batch_ctx: usize,
    /// Fill the batched prompts without computing them: decode speed doesn't depend on the KV values
    #[arg(long)]
    batch_fill: bool,
    /// Tokens decoded per measurement
    #[arg(long, default_value_t = 128)]
    steps: usize,
    /// Prompt tokens a sequence adds per step
    #[arg(long, default_value_t = 512)]
    chunk: usize,
    /// Also write the tables as JSON
    #[arg(long)]
    json: Option<PathBuf>,
    #[command(flatten)]
    sampling: SamplingArgs,
    #[command(flatten)]
    engine: EngineArgs,
}

pub fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

/// Deterministic pseudo-random ids in the real vocabulary; speed doesn't depend on them.
pub fn ids(n: usize, seed: u64) -> Vec<u32> {
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((x >> 33) % 151000) as u32
        })
        .collect()
}

pub const PREFILL: StepOptions = StepOptions { head: Rows::Last, hidden: Rows::None, read_logits: false, read_taps: false };

/// Prefills `toks` in synchronous steps of `chunk`; returns the last token drawn and the seconds.
pub fn prefill(eng: &mut Engine, id: SeqId, toks: &[u32], chunk: usize) -> Res<(u32, f64)> {
    let t0 = Instant::now();
    let mut last = 0;
    for c in toks.chunks(chunk) {
        last = eng.step(&[Run { seq: id, tokens: c }], &PREFILL)?.ids[0];
    }
    Ok((last, t0.elapsed().as_secs_f64()))
}

/// Decodes `steps` tokens for each of `seqs` (prefilled; `first[i]` is fed to `seqs[i]` first) in
/// one generation, which frees them, each with its drafter (replay mistakes at `mistakes` of
/// the positions). Returns the wall seconds, the steps and each sequence's tokens.
#[allow(clippy::type_complexity)]
pub fn decode_drafted(eng: &mut Engine, sampling: &SamplingArgs, seqs: &[SeqId], first: &[u32], steps: usize, drafts: Vec<Draft>, mistakes: f32) -> Res<(f64, Vec<StepStats>, Vec<Vec<u32>>)> {
    let mut g = eng.generator()?;
    g.set_replay_mistakes(mistakes, 1);
    for (i, ((&seq, &t), draft)) in seqs.iter().zip(first).zip(drafts).enumerate() {
        g.submit(Request { prompt: Prompt::Resume(seq, t), max_tokens: steps, stop: Vec::new(), sampling: sampling.with(0.0, i as u64), draft })?;
    }
    let mut tokens = vec![Vec::new(); seqs.len()];
    let t0 = Instant::now();
    while !g.is_idle() {
        for ev in g.step()? {
            if let Event::Token(id, t) = ev {
                tokens[id.0].push(t);
            }
        }
    }
    Ok((t0.elapsed().as_secs_f64(), std::mem::take(&mut g.stats), tokens))
}

/// `decode_drafted` without drafters.
pub fn decode(eng: &mut Engine, sampling: &SamplingArgs, seqs: &[SeqId], first: &[u32], steps: usize) -> Res<(f64, Vec<StepStats>)> {
    let (wall, stats, _) = decode_drafted(eng, sampling, seqs, first, steps, vec![Draft::None; seqs.len()], 0.0)?;
    Ok((wall, stats))
}

/// Median GPU ms and encode us of the steps.
pub fn medians(stats: &[StepStats]) -> (f64, f64) {
    let gpu: Vec<f64> = stats.iter().map(|s| s.gpu_seconds).collect();
    let enc: Vec<f64> = stats.iter().map(|s| s.encode_seconds).collect();
    (median(&gpu) * 1e3, median(&enc) * 1e6)
}

/// Pipelines compiled, and the weights' pages faulted in.
pub fn warm_up(eng: &mut Engine, sampling: &SamplingArgs, chunk: usize) -> Res {
    let id = eng.new_seq()?;
    let (t, _) = prefill(eng, id, &ids(256, 1), chunk)?;
    decode(eng, sampling, &[id], &[t], 16).map(|_| ())
}

pub fn bench(a: &Args) -> Res {
    let (steps, chunk, batch_ctx) = (a.steps, a.chunk, a.batch_ctx);
    let max_batch = a.batch.max().unwrap_or(1);
    let longest = a.decode_ctx.max().max(a.prefill.max()).unwrap_or(0) + steps + 1;
    let cfg = EngineConfig {
        max_seqs: max_batch.max(2),
        max_step_tokens: chunk.max(max_batch),
        max_chunk: chunk,
        global_kv_tokens: (longest + 256).max(max_batch * (batch_ctx + steps + 256)),
        max_seq_len: longest.max(batch_ctx + steps + 1),
        max_head_rows: max_batch.max(8),
    };
    let mut eng = a.engine.load(cfg)?;
    warm_up(&mut eng, &a.sampling, chunk)?;

    let how = match a.sampling.with(0.0, 0) {
        s if s.temperature == 0.0 => "greedy".to_string(),
        s => format!("T={} top_p={}", s.temperature, s.top_p),
    };
    println!("## decode, single stream ({how}, tokens fed back on the GPU, pipelined)");
    println!("| context | tok/s (wall) | GPU ms/step (median) | encode us/step (median) |\n|---:|---:|---:|---:|");
    let mut decodes = Vec::new();
    for &ctx in &a.decode_ctx.0 {
        let id = eng.new_seq()?;
        let (t, _) = prefill(&mut eng, id, &ids(ctx, 2 + ctx as u64), chunk)?;
        let (wall, stats) = decode(&mut eng, &a.sampling, &[id], &[t], steps)?;
        let (tps, (gpu_ms, enc_us)) = (steps as f64 / wall, medians(&stats));
        println!("| {ctx} | {tps:.1} | {gpu_ms:.3} | {enc_us:.0} |");
        decodes.push(json!({"context": ctx, "tok_s": tps, "gpu_ms": gpu_ms, "encode_us": enc_us}));
    }

    println!("\n## prefill (chunks of {chunk}, synchronous steps)");
    println!("| tokens | tok/s | seconds |\n|---:|---:|---:|");
    let mut prefills = Vec::new();
    for &len in &a.prefill.0 {
        let id = eng.new_seq()?;
        let (_, secs) = prefill(&mut eng, id, &ids(len, 99 + len as u64), chunk)?;
        eng.free_seq(id);
        let tps = len as f64 / secs;
        println!("| {len} | {tps:.0} | {secs:.3} |");
        prefills.push(json!({"tokens": len, "tok_s": tps, "seconds": secs}));
    }

    let filled = if a.batch_fill { ", filled without computing them" } else { "" };
    println!("\n## batched decode ({batch_ctx}-token prompts{filled}, N sequences, one token each per step)");
    println!("| N | aggregate tok/s | GPU ms/step | encode us/step |\n|---:|---:|---:|---:|");
    let mut batched = Vec::new();
    for &n in &a.batch.0 {
        let seqs: Vec<SeqId> = (0..n).map(|_| eng.new_seq()).collect::<Result<_, _>>()?;
        let mut first = Vec::new();
        for (j, &s) in seqs.iter().enumerate() {
            first.push(match a.batch_fill {
                true => eng.bench_fill(s, batch_ctx).map(|_| 100)?,
                false => prefill(&mut eng, s, &ids(batch_ctx, 7 + j as u64), chunk)?.0,
            });
        }
        let bsteps = steps.min(64);
        let (wall, stats) = decode(&mut eng, &a.sampling, &seqs, &first, bsteps)?;
        let (tps, (gpu_ms, enc_us)) = ((n * bsteps) as f64 / wall, medians(&stats));
        println!("| {n} | {tps:.0} | {gpu_ms:.3} | {enc_us:.0} |");
        batched.push(json!({"n": n, "context": batch_ctx, "tok_s": tps, "gpu_ms": gpu_ms, "encode_us": enc_us}));
    }

    if let Some(p) = &a.json {
        files::write(p, format!("{:#}\n", json!({"decode": decodes, "prefill": prefills, "batch": batched})))?;
    }
    Ok(())
}
