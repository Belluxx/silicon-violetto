use std::path::PathBuf;

use serde_json::json;
use violetto::{Draft, Engine, EngineConfig, Prompt, Request, SeqId, StepStats};

use crate::bench::{decode_drafted, ids, prefill, warm_up};
use crate::{files, load_drafter, EngineArgs, List, Res, SamplingArgs};

#[derive(clap::Args)]
pub struct Args {
    /// Rows per sequence and step to measure
    #[arg(long, default_value = "2,4,8")]
    rows: List<usize>,
    /// Sequence counts to measure
    #[arg(long, default_value = "1,16")]
    batch: List<usize>,
    /// Context the prompts are cut at
    #[arg(long, default_value_t = 1024)]
    ctx: usize,
    /// Request file of real prompts, taken in turn [default: random ids, which draft poorly]
    #[arg(long)]
    prompts: Option<PathBuf>,
    /// Engine's drafter, from tools/spec/export.py [default: the replay drafter]
    #[arg(long)]
    drafter: Option<PathBuf>,
    /// Shares of the replay drafter's positions drafted wrong
    #[arg(long, default_value = "0.12,0.3")]
    mistakes: List<f32>,
    /// Tokens decoded per sequence
    #[arg(long, default_value_t = 128)]
    steps: usize,
    /// Prompt tokens a sequence adds per step
    #[arg(long, default_value_t = 512)]
    chunk: usize,
    /// Also write the table as JSON
    #[arg(long)]
    json: Option<PathBuf>,
    #[command(flatten)]
    sampling: SamplingArgs,
    #[command(flatten)]
    engine: EngineArgs,
}

/// The most tokens request i of `lens` (prompt lengths, all submitted at once) can emit while
/// the generator prefills the ones after it: decoding runs take their rows of each step's
/// `step_tokens` first, prompt chunks the rest.
fn prefill_lag(lens: &[usize], i: usize, rows: usize, step_tokens: usize) -> usize {
    let per_step = step_tokens.saturating_sub(lens.len() * rows).max(1);
    rows * (lens[i + 1..].iter().sum::<usize>().div_ceil(per_step) + lens.len() - i)
}

/// The prompts through one generation with the engine's drafter, for `steps` tokens once every
/// prompt is in: the earlier ones get the tokens they emit while the later ones prefill on
/// top (prefill_lag).
fn decode_model(eng: &mut Engine, sampling: &SamplingArgs, prompts: &[Vec<u32>], steps: usize, rows: usize, step_tokens: usize) -> Res<Vec<StepStats>> {
    let mut g = eng.generator()?;
    let lens: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
    for (i, p) in prompts.iter().enumerate() {
        let max_tokens = steps + prefill_lag(&lens, i, rows, step_tokens);
        g.submit(Request { prompt: Prompt::Tokens(p.clone()), max_tokens, stop: Vec::new(), sampling: sampling.with(0.0, i as u64), draft: Draft::Model { rows } })?;
    }
    while !g.is_idle() {
        g.step()?;
    }
    Ok(std::mem::take(&mut g.stats))
}

/// (tok/s over their GPU time, tokens per sequence and step, GPU ms per step) of the steps.
fn rates(steps: &[&StepStats], n: usize) -> (f64, f64, f64) {
    let secs: f64 = steps.iter().map(|s| s.gpu_seconds).sum();
    let emitted = steps.iter().map(|s| s.emitted).sum::<usize>() as f64;
    (emitted / secs, emitted / (steps.len() * n) as f64, 1e3 * secs / steps.len() as f64)
}

pub fn bench_spec(a: &Args) -> Res {
    let (steps, chunk, ctx) = (a.steps, a.chunk, a.ctx);
    let (max_n, max_rows) = (a.batch.max().unwrap_or(1), a.rows.max().unwrap_or(1));
    let step_tokens = chunk.max(max_n * max_rows);
    let model = a.drafter.is_some();
    // With the drafter, the generator runs the prompts, and the first ones decode while the
    // rest prefill.
    let lags: Vec<usize> = match model {
        true => (0..max_n).map(|i| prefill_lag(&vec![ctx; max_n], i, max_rows, step_tokens)).collect(),
        false => vec![0],
    };
    let longest = ctx + steps + 1 + lags[0];
    let cfg = EngineConfig {
        max_seqs: max_n.max(2),
        max_step_tokens: step_tokens,
        max_chunk: chunk,
        global_kv_tokens: (longest + 256).max(max_n * (ctx + steps + 256) + lags.iter().sum::<usize>()),
        max_seq_len: longest,
        max_head_rows: (max_n * max_rows).max(8),
    };
    // Rows as given, not by the rows policy, unless --spec-tokens.
    let mut tuning = a.engine.tuning()?;
    tuning.spec_tokens = a.engine.spec_tokens.unwrap_or(0);
    let mut eng = a.engine.load_with(cfg, tuning)?;
    if let Some(d) = &a.drafter {
        load_drafter(&mut eng, d)?;
    }
    warm_up(&mut eng, &a.sampling, chunk)?;

    let real = a.prompts.as_deref().map(|p| files::prompts(p, ctx)).transpose()?;
    let which = if model { "the engine's drafter" } else { "the replay drafter" };
    println!("## speculative decoding ({ctx}-token prompts, {which}; steps that ran every sequence: tok/s over their GPU time)");
    println!("| N | rows | mistakes | tok/s | vs 1 row | tokens/step | GPU ms/step |\n|---:|---:|---:|---:|---:|---:|---:|");
    let mut table = Vec::new();
    for &n in &a.batch.0 {
        let prompts: Vec<Vec<u32>> = match &real {
            Some(all) => (0..n).map(|j| all[j % all.len()].clone()).collect(),
            None => (0..n).map(|j| ids(ctx, 31 + j as u64)).collect(),
        };
        let fill = |eng: &mut Engine| -> Res<(Vec<SeqId>, Vec<u32>)> {
            let seqs: Vec<SeqId> = (0..n).map(|_| eng.new_seq()).collect::<Result<_, _>>()?;
            let first = seqs.iter().zip(&prompts).map(|(&s, p)| Ok(prefill(eng, s, p, chunk)?.0)).collect::<Res<_>>()?;
            Ok((seqs, first))
        };
        // Plain decoding: the baseline, and the tokens the replay drafter drafts.
        let (seqs, first) = fill(&mut eng)?;
        let (_, stats, recorded) = decode_drafted(&mut eng, &a.sampling, &seqs, &first, steps, vec![Draft::None; n], 0.0)?;
        let (base, _, gpu) = rates(&stats.iter().filter(|s| s.tokens == n).collect::<Vec<_>>(), n);
        println!("| {n} | 1 | - | {base:.0} | 1.00 | 1.00 | {gpu:.3} |");
        for &rows in &a.rows.0 {
            for &m in if model { &[0.0][..] } else { &a.mistakes.0[..] } {
                let stats = if model {
                    decode_model(&mut eng, &a.sampling, &prompts, steps, rows, step_tokens)?
                } else {
                    let (seqs, first) = fill(&mut eng)?;
                    let trace = |j: usize| prompts[j].iter().chain([&first[j]]).chain(&recorded[j]).copied().collect();
                    let drafts = (0..n).map(|j| Draft::Replay { rows, trace: trace(j) }).collect();
                    let (_, stats, tokens) = decode_drafted(&mut eng, &a.sampling, &seqs, &first, steps, drafts, m)?;
                    // Where a sequence leaves the recorded tokens (a draw near a CDF boundary that
                    // the verify layout's rounding flipped), its drafts are all wrong from there.
                    let mut left: Vec<usize> = tokens.iter().zip(&recorded).map(|(a, b)| a.iter().zip(b).position(|(x, y)| x != y).unwrap_or(steps)).collect();
                    left.sort();
                    let kept = left.iter().filter(|&&k| k == steps).count();
                    eprintln!("  rows {rows} mistakes {m}: {kept}/{n} sequences kept the recorded tokens; the others left at {:?}", &left[..n - kept]);
                    stats
                };
                // The steps that ran every sequence: the tail, with the sequences left alone
                // after the others finished, measures nothing a busy server sees.
                let full: Vec<&StepStats> = stats.iter().filter(|s| s.tokens == n * rows && (s.decode || !model)).collect();
                let (tps, per_step, gpu) = rates(&full, n);
                let mistakes = if model { "drafter".to_string() } else { m.to_string() };
                println!("| {n} | {rows} | {mistakes} | {tps:.0} | {:.2} | {per_step:.2} | {gpu:.3} |", tps / base);
                table.push(json!({"n": n, "context": ctx, "rows": rows, "mistakes": mistakes, "tok_s": tps, "speedup": tps / base, "tokens_per_step": per_step, "gpu_ms": gpu}));
            }
        }
    }
    if let Some(p) = &a.json {
        files::write(p, format!("{:#}\n", json!({"speculative": table})))?;
    }
    Ok(())
}
