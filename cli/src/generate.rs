use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use serde_json::json;
use violetto::{Draft, EngineConfig, Event, Finish, Prompt, Request};

use crate::files::{self, parse};
use crate::{DraftArgs, EngineArgs, List, Res, SamplingArgs};

#[derive(clap::Args)]
pub struct Args {
    /// Request file
    #[arg(long)]
    requests: PathBuf,
    /// Output: one JSON line per request
    #[arg(long)]
    out: PathBuf,
    /// Keep the requests already in the output, and append the others
    #[arg(long)]
    resume: bool,
    /// Requests decoding at once
    #[arg(long, default_value_t = 16)]
    max_seqs: usize,
    /// Tokens a request generates at most, unless its line says
    #[arg(long, default_value_t = 32768)]
    max_tokens: usize,
    /// Tokens that end a generation: <|endoftext|> and <|im_end|>
    #[arg(long, default_value = "151643,151645")]
    stop: List<u32>,
    /// Prompt tokens a sequence adds per step
    #[arg(long, default_value_t = 512)]
    chunk: usize,
    /// KV pool of the global layers, in tokens of 12 KiB [default: every slot at the longest request]
    #[arg(long)]
    kv_tokens: Option<usize>,
    /// Test drafter: draft the tokens of this earlier output, for the requests it has
    #[arg(long, value_name = "OUTPUT")]
    replay: Option<PathBuf>,
    /// Share of the replay drafter's positions drafted wrong
    #[arg(long, default_value_t = 0.0)]
    replay_mistakes: f32,
    #[command(flatten)]
    draft: DraftArgs,
    #[command(flatten)]
    sampling: SamplingArgs,
    #[command(flatten)]
    engine: EngineArgs,
}

fn read_requests(a: &Args) -> Res<Vec<(String, Request)>> {
    let lines = files::requests(&a.requests)?.into_iter().enumerate();
    lines
        .map(|(i, line)| {
            let (mut sampling, mut max_tokens) = (a.sampling.with(0.6, i as u64), a.max_tokens);
            for kv in line.opts.split(',').filter(|o| *o != "-") {
                let (k, v) = kv.split_once('=').ok_or_else(|| format!("{}: option {kv} isn't key=value", line.name))?;
                match k {
                    "seed" => sampling.seed = parse(&line.name, k, v)?,
                    "max_tokens" => max_tokens = parse(&line.name, k, v)?,
                    "temperature" => sampling.temperature = parse(&line.name, k, v)?,
                    "top_p" => sampling.top_p = parse(&line.name, k, v)?,
                    _ => return Err(format!("{}: unknown option {k}", line.name)),
                }
            }
            let req = Request { prompt: Prompt::Tokens(line.prompt), max_tokens, stop: a.stop.0.clone(), sampling, draft: a.draft.draft() };
            Ok((line.name, req))
        })
        .collect()
}

pub fn generate(a: &Args) -> Res {
    let mut reqs = read_requests(a)?;
    let rows = a.draft.spec_rows;
    if let Some(p) = &a.replay {
        let outs = files::outputs(p)?;
        for (name, r) in &mut reqs {
            if let (Some(out), Prompt::Tokens(prompt)) = (outs.get(name), &r.prompt) {
                r.draft = Draft::Replay { rows, trace: prompt.iter().chain(out).copied().collect() };
            }
        }
    }
    let prompt_len = |r: &Request| match &r.prompt {
        Prompt::Tokens(p) => p.len(),
        Prompt::Resume(..) => unreachable!(),
    };
    let max_seqs = a.max_seqs.min(reqs.len()).max(1);
    let longest = reqs.iter().map(|(_, r)| prompt_len(r) + r.max_tokens).max().unwrap_or(1);
    let drafts = a.replay.is_some() || a.draft.drafter.is_some();
    let cfg = EngineConfig {
        max_seqs,
        max_step_tokens: a.chunk.max(max_seqs),
        max_chunk: a.chunk,
        global_kv_tokens: a.kv_tokens.unwrap_or(max_seqs * longest.next_multiple_of(128)),
        max_seq_len: longest,
        max_head_rows: max_seqs * if drafts { rows } else { 1 },
    };
    let mut eng = a.engine.load(cfg)?;
    a.draft.load(&mut eng)?;
    let resume = a.resume && a.out.exists();
    let finished = if resume { files::outputs(&a.out)? } else { HashMap::new() };
    let file = std::fs::OpenOptions::new().create(true).append(resume).write(true).truncate(!resume).open(&a.out);
    let mut out = std::io::BufWriter::new(file.map_err(|e| format!("{}: {e}", a.out.display()))?);
    let mut g = eng.generator()?;
    g.set_replay_mistakes(a.replay_mistakes, 1);
    let n = reqs.len();
    let (mut names, mut prompts) = (Vec::with_capacity(n), Vec::with_capacity(n));
    for (name, req) in reqs.into_iter().filter(|(name, _)| !finished.contains_key(name)) {
        prompts.push(prompt_len(&req));
        g.submit(req).map_err(|e| format!("{name}: {e}"))?;
        names.push(name);
    }
    let mut tokens: Vec<Vec<u32>> = vec![Vec::new(); n];
    let (t0, mut report) = (Instant::now(), 10.0);
    let (mut generated, mut done) = (0usize, 0usize);
    while !g.is_idle() {
        for ev in g.step()? {
            match ev {
                Event::Token(id, t) => {
                    tokens[id.0].push(t);
                    generated += 1;
                }
                Event::Done(id, finish) => {
                    let i = id.0;
                    let finish = if finish == Finish::Stop { "stop" } else { "length" };
                    let line = json!({"name": names[i], "finish": finish, "prompt_tokens": prompts[i], "tokens": std::mem::take(&mut tokens[i])});
                    writeln!(out, "{line}").and_then(|_| out.flush()).map_err(|e| format!("{}: {e}", a.out.display()))?;
                    done += 1;
                }
            }
        }
        let t = t0.elapsed().as_secs_f64();
        if t >= report {
            let (queued, running) = g.counts();
            eprintln!("{t:.0} s: {done}/{n} done, {running} running, {queued} queued; {generated} tokens, {:.0} tok/s", generated as f64 / t);
            report += 10.0;
        }
    }
    let wall = t0.elapsed().as_secs_f64();
    let steps = g.stats.len();
    let fed: usize = g.stats.iter().map(|s| s.tokens).sum();
    let gpu: f64 = g.stats.iter().map(|s| s.gpu_seconds).sum();
    eprintln!(
        "{n} requests, {generated} tokens generated in {wall:.1} s ({:.0} tok/s); {steps} steps of {:.1} tokens on average, GPU busy {:.0}%",
        generated as f64 / wall,
        fed as f64 / steps.max(1) as f64,
        100.0 * gpu / wall
    );
    // Decoding alone: the steps without a prompt chunk, which bring every token but each
    // request's first.
    let mut dec: Vec<f64> = g.stats.iter().filter(|s| s.decode).map(|s| s.gpu_seconds).collect();
    if !dec.is_empty() {
        let secs: f64 = dec.iter().sum();
        dec.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!(
            "decoding: {} small-run steps, GPU ms median {:.3}; {:.0} tok/s over their GPU time",
            dec.len(),
            dec[dec.len() / 2] * 1e3,
            generated.saturating_sub(n) as f64 / secs
        );
    }
    Ok(())
}
