use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use violetto::{EngineConfig, Rows, Run, StepOptions, HIDDEN};

use crate::{files, safetensors, EngineArgs, List, Res};

#[derive(clap::Args)]
pub struct Args {
    /// Request file the traces were generated from
    #[arg(long)]
    requests: PathBuf,
    /// Their `generate` output
    #[arg(long)]
    traces: PathBuf,
    /// Output directory, one file per trace; existing files are skipped
    #[arg(long, value_name = "DIR")]
    out: PathBuf,
    /// Layers whose outputs are kept
    #[arg(long, default_value = "35,47")]
    taps: List<usize>,
    /// Tokens per step
    #[arg(long, default_value_t = 512)]
    chunk: usize,
    #[command(flatten)]
    engine: EngineArgs,
}

pub fn features(a: &Args) -> Res {
    let reqs: HashMap<String, Vec<u32>> = files::requests(&a.requests)?.into_iter().map(|r| (r.name, r.prompt)).collect();
    let traces = files::outputs(&a.traces)?;
    std::fs::create_dir_all(&a.out).map_err(|e| format!("{}: {e}", a.out.display()))?;
    let (taps, chunk) = (&a.taps.0, a.chunk);
    let mut names: Vec<&String> = traces.keys().filter(|n| reqs.contains_key(*n)).collect();
    names.sort();
    let longest = names.iter().map(|n| reqs[*n].len() + traces[*n].len()).max().unwrap_or(1);
    let cfg = EngineConfig { max_seqs: 1, max_step_tokens: chunk, max_chunk: chunk, global_kv_tokens: longest.next_multiple_of(128) + 128, max_seq_len: longest, max_head_rows: 8 };
    let mut eng = a.engine.load(cfg)?;
    eng.set_taps(taps)?;
    let opts = StepOptions { head: Rows::None, hidden: Rows::None, read_logits: false, read_taps: true };
    let (t0, mut done_tokens) = (Instant::now(), 0usize);
    for (i, name) in names.iter().enumerate() {
        let path = a.out.join(format!("{name}.safetensors"));
        if path.exists() {
            continue;
        }
        let prompt = &reqs[*name];
        let ids: Vec<u32> = prompt.iter().chain(&traces[*name]).copied().collect();
        let n = ids.len();
        let id = eng.new_seq()?;
        // [position][tap][HIDDEN], from each step's [tap][token][HIDDEN].
        let mut feat = vec![0u16; n * taps.len() * HIDDEN];
        for start in (0..n).step_by(chunk) {
            let end = (start + chunk).min(n);
            let o = eng.step(&[Run { seq: id, tokens: &ids[start..end] }], &opts)?;
            for pos in start..end {
                for k in 0..taps.len() {
                    let (src, dst) = ((k * (end - start) + pos - start) * HIDDEN, (pos * taps.len() + k) * HIDDEN);
                    feat[dst..dst + HIDDEN].copy_from_slice(&o.taps[src..src + HIDDEN]);
                }
            }
        }
        eng.free_seq(id);
        let mut st = safetensors::Writer::default();
        st.u32("tokens", &[n], &ids);
        st.u32("prompt_len", &[1], &[prompt.len() as u32]);
        st.u32("taps", &[taps.len()], &taps.iter().map(|&t| t as u32).collect::<Vec<_>>());
        st.bf16("feat", &[n, taps.len(), HIDDEN], &feat);
        st.save(&path)?;
        done_tokens += n;
        eprintln!("{}/{} {name}: {n} tokens ({:.0} tok/s)", i + 1, names.len(), done_tokens as f64 / t0.elapsed().as_secs_f64());
    }
    Ok(())
}
