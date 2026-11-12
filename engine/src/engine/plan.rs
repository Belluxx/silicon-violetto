use super::buffers::PART_SLOTS_PER_TOKEN;
use super::kernels::{RunIn, FEED_ROW, FEED_W, VERIFY_T};
use super::{Engine, Rows, SeqId, DEC_TGS_PER_CORE, TILE_TGS_PER_CORE};
use crate::gpu;
use crate::model::{N_KV_HEADS, WINDOW};

/// One sequence's tokens in a step.
pub struct PlanRun<'a> {
    pub seq: SeqId,
    /// Input ids; FEED_ROW + i for the slot's GPU-fed token i.
    pub tokens: &'a [u32],
    /// The rows that get the head. The last of them sets the sequence's next input.
    pub head: Rows,
    /// Speculative verification: the head on every row, and rows 1.. are drafts (see
    /// `accept`).
    pub verify: bool,
}

pub struct Plan {
    pub n: usize,
    /// Every run has at most VERIFY_T tokens, so attn_verify applies.
    pub fused: bool,
    pub n_head: usize,
    pub n_hidden: usize,
    pub n_runs: usize,
    /// Some verify run has 3 rows or more, so its accepted path may need kv_compact.
    pub branches: bool,
    /// attn_mma's query tiles; with `fused`, one per run.
    pub tiles: usize,
    /// attn_verify's column groups (of the longest run) and simdgroups per group.
    pub verify_shape: (usize, usize),
    /// Tokens in the longest run.
    pub max_run: usize,
    /// (keys per split, splits) on local and global layers.
    splits: [(u32, u32); 2],
}

impl Plan {
    pub fn splits(&self, global: bool) -> (u32, u32) {
        self.splits[global as usize]
    }
}

impl Engine {
    /// Lays out a step in step set `set` and advances the sequences (verify runs: to the
    /// most they can reach).
    pub(super) fn plan(&mut self, set: usize, runs: &[PlanRun], hidden: Rows) -> Result<Plan, String> {
        let n: usize = runs.iter().map(|r| r.tokens.len()).sum();
        if n == 0 || n > self.cfg.max_step_tokens {
            return Err(format!("step has {n} tokens (max {})", self.cfg.max_step_tokens));
        }
        let mut tokens = Vec::with_capacity(n);
        let mut tin = Vec::with_capacity(2 * n);
        let mut vis = Vec::with_capacity(n);
        let (mut head_rows, mut hidden_rows, mut samples) = (Vec::new(), Vec::new(), Vec::new());
        let mut tiles: Vec<u32> = Vec::new();
        let mut run_in = Vec::with_capacity(runs.len());
        let (mut max_cached, mut tile_keys, mut max_count) = (0usize, [0usize; 2], 0usize);
        for (ri, run) in runs.iter().enumerate() {
            let (id, count) = (run.seq, run.tokens.len());
            if count == 0 || count > self.cfg.max_chunk {
                return Err(format!("{count} tokens for one sequence in a step (max_chunk {})", self.cfg.max_chunk));
            }
            if run.tokens.iter().any(|&t| t >= FEED_ROW && (t - FEED_ROW) as usize >= FEED_W) {
                return Err(format!("only {FEED_W} GPU-fed tokens per slot"));
            }
            if runs[..ri].iter().any(|r| r.seq == id) {
                return Err("a sequence appears twice in one step".into());
            }
            let seq = self.seqs[id.0 as usize].as_mut().ok_or("unknown sequence")?;
            let new_len = seq.len + count;
            if new_len > self.cfg.max_seq_len {
                return Err(format!("sequence would reach {new_len} tokens (max {})", self.cfg.max_seq_len));
            }
            self.kv.reserve(id.0, &mut seq.pages, new_len)?;
            let first_row = tokens.len();
            tokens.extend_from_slice(run.tokens);
            for row in 0..count as u32 {
                tin.extend([id.0, row]);
                // Causal within its tile of 8; `embed` overwrites a draft tree's.
                vis.push(u32::MAX >> (31 - row % 8));
            }
            let rows = |r: Rows, v: &mut Vec<u32>| match r {
                Rows::None => {}
                Rows::Last => v.push((first_row + count - 1) as u32),
                Rows::All => v.extend((first_row..first_row + count).map(|x| x as u32)),
            };
            let first_head = head_rows.len();
            rows(if run.verify { Rows::All } else { run.head }, &mut head_rows);
            rows(hidden, &mut hidden_rows);
            samples.extend(head_rows[first_head..].iter().map(|&row| seq.sampling.row_of(row)));
            run_in.push(RunIn {
                slot: id.0,
                first: first_row as u32,
                count: count as u32,
                head0: first_head as u32,
                nhead: (head_rows.len() - first_head) as u32,
                verify: run.verify as u32,
            });
            for i in (0..count).step_by(8) {
                let nt = (count - i).min(8);
                tiles.extend([(first_row + i) as u32, nt as u32]);
                let (p0, end) = (seq.len + i, seq.len + i + nt);
                tile_keys[0] = tile_keys[0].max(end - p0.saturating_sub(WINDOW));
                tile_keys[1] = tile_keys[1].max(end);
            }
            max_cached = max_cached.max(seq.len);
            max_count = max_count.max(count);
            seq.len = new_len;
        }
        if head_rows.len() > self.cfg.max_head_rows {
            return Err(format!("{} head rows (max {})", head_rows.len(), self.cfg.max_head_rows));
        }
        let st = &self.sets[set];
        gpu::write(&st.tokens, 0, &tokens);
        gpu::write(&st.tin, 0, &tin);
        gpu::write(&st.vis, 0, &vis);
        gpu::write(&st.head_rows, 0, &head_rows);
        gpu::write(&st.sample_rows, 0, &samples);
        gpu::write(&st.hidden_rows, 0, &hidden_rows);
        gpu::write(&st.tiles, 0, &tiles);
        gpu::write(&st.runs, 0, &run_in);

        let fused = max_count <= VERIFY_T;
        let n_tiles = tiles.len() / 2;
        let cap = (self.cfg.max_step_tokens * PART_SLOTS_PER_TOKEN / n).clamp(1, 256);
        let splits = |global: bool| {
            if fused {
                let cached = if global { max_cached } else { max_cached.min(WINDOW) };
                let forced = self.tuning.decode_split[global as usize];
                decode_splits(runs.len(), cached, cap, DEC_TGS_PER_CORE * self.cores, self.cores, forced)
            } else {
                tile_splits(n_tiles, tile_keys[global as usize], cap, TILE_TGS_PER_CORE * self.cores)
            }
        };
        // Column groups of 8 for max_count tokens x 5 heads.
        let groups = (5 * max_count).div_ceil(8).clamp(1, 5);
        Ok(Plan {
            n,
            fused,
            n_head: head_rows.len(),
            n_hidden: hidden_rows.len(),
            n_runs: runs.len(),
            branches: runs.iter().any(|r| r.verify && r.tokens.len() > 2),
            tiles: n_tiles,
            verify_shape: (groups, self.tuning.verify_ki[groups - 1]),
            max_run: max_count,
            splits: [splits(false), splits(true)],
        })
    }
}

/// attn_verify's keys per split and splits per run: the longest run's cached keys over
/// about `tgs` threadgroups per layer, in an even number of splits (odd counts measured
/// slower) of at most `max` per kv head, plus a split of its own for the new keys.
pub(super) fn decode_splits(runs: usize, cached: usize, cap: usize, tgs: usize, max: usize, forced: Option<usize>) -> (u32, u32) {
    let want = tgs.div_ceil(N_KV_HEADS * runs).next_multiple_of(2).min(max).clamp(1, cap);
    let mut split = forced.unwrap_or_else(|| cached.div_ceil(want).next_multiple_of(32).max(32));
    while cached.div_ceil(split) + 1 > cap {
        split += 32;
    }
    (split as u32, (cached.div_ceil(split) + 1) as u32)
}

/// attn_mma's keys per split and splits per tile, for about `tgs` threadgroups per layer.
fn tile_splits(tiles: usize, max_keys: usize, cap: usize, tgs: usize) -> (u32, u32) {
    let want = tgs.div_ceil(N_KV_HEADS * tiles).clamp(1, cap);
    let mut split = max_keys.div_ceil(want).next_multiple_of(32).max(32);
    while max_keys.div_ceil(split) > cap {
        split += 32;
    }
    (split as u32, max_keys.div_ceil(split) as u32)
}

/// Rounds to the nearest bf16, ties to even, as torch's .to(bfloat16) (finite inputs).
pub(super) fn round_bf16(x: f32) -> f32 {
    let u = x.to_bits();
    f32::from_bits((u + 0x7fff + ((u >> 16) & 1)) & 0xffff_0000)
}
