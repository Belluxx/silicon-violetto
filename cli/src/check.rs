use std::path::PathBuf;

use serde_json::json;
use violetto::{Draft, EngineConfig, Event, Prompt, Request, Sampling};

use crate::{files, load_drafter, EngineArgs, Res};

#[derive(clap::Args)]
pub struct DrafterArgs {
    /// Drafter to check, from tools/spec/export.py
    #[arg(long)]
    drafter: PathBuf,
    /// Request file: each prompt is prefilled alone
    #[arg(long)]
    requests: PathBuf,
    /// Output: per prompt its bonus token, the block's logits, and the drafts and their tree
    #[arg(long)]
    out: PathBuf,
    /// Rows of the block
    #[arg(long, default_value_t = 8)]
    rows: usize,
    #[command(flatten)]
    engine: EngineArgs,
}

pub fn drafter_check(a: &DrafterArgs) -> Res {
    let reqs = files::requests(&a.requests)?;
    let longest = reqs.iter().map(|r| r.prompt.len()).max().unwrap_or(1) + 16;
    let cfg = EngineConfig { max_seqs: 1, max_step_tokens: 512, max_chunk: 512, global_kv_tokens: longest.next_multiple_of(128) + 128, max_seq_len: longest, max_head_rows: 16 };
    let mut eng = a.engine.load(cfg)?;
    load_drafter(&mut eng, &a.drafter)?;
    let mut out = Vec::new();
    for r in &reqs {
        // The prompt draws one token, its bonus token, and the drafter then runs its block.
        let mut g = eng.generator()?;
        g.submit(Request { prompt: Prompt::Tokens(r.prompt.clone()), max_tokens: 1, stop: Vec::new(), sampling: Sampling::GREEDY, draft: Draft::Model { rows: a.rows } })?;
        let mut bonus = 0;
        while !g.is_idle() {
            for ev in g.step()? {
                if let Event::Token(_, t) = ev {
                    bonus = t;
                }
            }
        }
        drop(g);
        let d = eng.drafter_debug(a.rows, 0);
        out.push(json!({"name": r.name, "bonus": bonus, "feed": d.feed, "tree": d.tree, "vocab_size": d.vocab.len(), "logits": d.logits}));
    }
    files::write(&a.out, format!("{}\n", json!(out)))
}
