use std::path::Path;

use super::encode::{Epilogue, SmallAttn};
use super::kernels::{KvArgs, FEED_W, VERIFY_T};
use super::plan::{decode_splits, Plan};
use super::{Engine, DEC_TGS_PER_CORE};
use crate::gpu::{self, Buffer, Encoder, Gpu, Pipeline};
use crate::model::{matrix, Matrix, FFN, HEAD_DIM, HIDDEN, N_HEADS, N_KV_HEADS, QKV, WINDOW};
use crate::weights::{DType, Weights};

/// Draft rows per slot and step, at most (attn_verify's runs).
pub const MAX_ROWS: usize = VERIFY_T;
const PART_SLOTS: usize = 64;
/// Candidates per drafted row, at most (kernels.metal's CANDS). A block of R rows uses at
/// most R - 1.
const CANDS: usize = 4;
/// The temperature the drafter was trained at (tools/spec/drafter.py, TEMP), where its
/// probabilities are calibrated.
const TRAIN_TEMPERATURE: f32 = 0.6;

struct DLayer {
    qkv: Matrix,
    o: Matrix,
    gate_up: Matrix,
    down: Matrix,
    /// The two-tap convolutions before attention, after it, before the MLP and after it.
    convs: Option<[DConv; 4]>,
}

struct DConv {
    /// f32 [2][HIDDEN]: the kernels' base, for x[t] and x[t - 1].
    base: usize,
    /// [CONV_KC][HIDDEN]: the kernels' corrections per 16 channels (160 rows, then zeros).
    corr: Matrix,
}

/// The convolutions' correction rows, padded to the GEMM's 64-row tiles (draft_conv's).
const CONV_KC: usize = 192;

/// One step's block inputs, written by the CPU.
struct DSet {
    /// (slot, row in its block) per block row.
    tin: Buffer,
    vis: Buffer,
    /// (first row, rows) per block.
    tiles: Buffer,
}

pub struct Drafter {
    w: Buffer,
    fc: Vec<Matrix>,
    kv_all: Matrix,
    /// The projection of the context before the block, added to every block row's input.
    inject: Option<Matrix>,
    mask_emb: usize,
    layers: Vec<DLayer>,
    head: Matrix,
    vocab: Buffer,
    v: usize,
    pub taps: Vec<usize>,
    /// Rings per (layer, slot), as the target's local layers'.
    k_ring: Buffer,
    v_ring: Buffer,
    ring: usize,
    // Context, per token of the step.
    ctx: Buffer,
    ctx_inv: Buffer,
    kvp: Buffer,
    /// With inject: the normed context per (slot, position % ctx_ring_len), and each block
    /// row's copy of the one before its block.
    ctx_ring: Buffer,
    ctx_ring_len: usize,
    last: Buffer,
    // Block rows.
    x: Buffer,
    qkv: Buffer,
    inv: Buffer,
    attn: Buffer,
    mixed: Buffer,
    act: Buffer,
    hfin: Buffer,
    /// With conv: a convolution's output, the attention's and the MLP's outputs before
    /// theirs, and the corrections.
    cx: Buffer,
    co: Buffer,
    corr: Buffer,
    corr_inv: Buffer,
    logits: Buffer,
    /// Each drafted row's candidates (ids) and their probabilities, [row][CANDS].
    cand: Buffer,
    cand_q: Buffer,
    meta: Buffer,
    tokens: Buffer,
    part: Buffer,
    vdir: Buffer,
    gates: Buffer,
    sdpa: Buffer,
    iota: Buffer,
    /// Zeros: the attention gates' and XSA's weights.
    zeros: Buffer,
    sets: [DSet; 3],
    p: DPipes,
}

struct DPipes {
    draft_kv: Pipeline,
    draft_conv: Pipeline,
    draft_ctx: Pipeline,
    draft_embed: Pipeline,
    draft_top: Pipeline,
    draft_tree: Pipeline,
}

/// What `Engine::drafter_debug` reads back.
#[doc(hidden)]
pub struct DrafterDebug {
    /// Raw, over the draft vocabulary, [rows][vocab.len()].
    pub logits: Vec<f32>,
    pub vocab: Vec<u32>,
    /// The slot's next input, then its drafts.
    pub feed: Vec<u32>,
    /// Their nodes: depth | parent << 8 | the rows each sees << 16.
    pub tree: Vec<u32>,
}

/// A slot to draft for in a step, and its block's rows (1 + the drafts).
pub struct DraftRun {
    pub slot: u32,
    pub rows: usize,
}

impl Drafter {
    pub fn load(gpu: &Gpu, path: &Path, max_seqs: usize, max_tokens: usize, max_chunk: usize) -> Result<Self, String> {
        let w = Weights::load(gpu, path)?;
        let meta = w.f32s(w.tensor("meta")?);
        let (n_layers, n_taps, window, v) = (meta[0] as usize, meta[1] as usize, meta[3] as usize, meta[4] as usize);
        if window != WINDOW || meta.len() != 5 + n_taps {
            return Err(format!("{}: window {window} (the engine's is {WINDOW}), meta {meta:?}", path.display()));
        }
        let taps = meta[5..].iter().map(|&t| t as usize).collect();
        let m = |name: &str, rows: usize, k: usize| matrix(&w, name, rows, k);
        let fc = (0..n_taps).map(|k| m(&format!("fc.{k}"), HIDDEN, HIDDEN)).collect::<Result<_, _>>()?;
        let layers = (0..n_layers)
            .map(|i| {
                Ok(DLayer {
                    qkv: m(&format!("D{i:02}.qkv"), QKV, HIDDEN)?,
                    o: m(&format!("D{i:02}.o"), HIDDEN, HIDDEN)?,
                    gate_up: m(&format!("D{i:02}.gate_up"), 2 * FFN, HIDDEN)?,
                    down: m(&format!("D{i:02}.down"), HIDDEN, FFN)?,
                    convs: if w.has(&format!("D{i:02}.conv0.base")) {
                        let c = |c: usize| -> Result<DConv, String> {
                            Ok(DConv {
                                base: w.get(&format!("D{i:02}.conv{c}.base"), DType::F32, &[2, HIDDEN])?.offset,
                                corr: m(&format!("D{i:02}.conv{c}.corr"), CONV_KC, HIDDEN)?,
                            })
                        };
                        Some([c(0)?, c(1)?, c(2)?, c(3)?])
                    } else {
                        None
                    },
                })
            })
            .collect::<Result<_, String>>()?;
        let kv_all = m("kv_all", n_layers * 2 * N_KV_HEADS * HEAD_DIM, HIDDEN)?;
        let inject = if w.has("inject") { Some(m("inject", HIDDEN, HIDDEN)?) } else { None };
        let head = m("head", v, HIDDEN)?;
        let mask_emb = w.get("mask_emb", DType::F32, &[HIDDEN])?.offset;
        let ids: Vec<u32> = w.f32s(w.get("vocab", DType::F32, &[v])?).iter().map(|&x| x as u32).collect();
        let vocab = gpu.buffer(4 * v)?;
        gpu::write(&vocab, 0, &ids);
        let ring = WINDOW + max_chunk;
        let ring_bytes = n_layers * max_seqs * N_KV_HEADS * ring * HEAD_DIM * 2;
        let rows = max_seqs * MAX_ROWS;
        let f = |floats: usize| gpu.buffer(4 * floats);
        let set = || -> Result<DSet, String> { Ok(DSet { tin: f(2 * rows)?, vis: f(rows)?, tiles: f(2 * max_seqs)? }) };
        let iota = f(rows)?;
        // A step's run is at most max_chunk tokens, so its positions don't collide.
        let ctx_ring_len = if inject.is_some() { max_chunk } else { 0 };
        gpu::write(&iota, 0, &(0..rows as u32).collect::<Vec<_>>());
        Ok(Self {
            fc,
            kv_all,
            inject,
            mask_emb,
            layers,
            head,
            vocab,
            v,
            taps,
            k_ring: gpu.buffer(ring_bytes)?,
            v_ring: gpu.buffer(ring_bytes)?,
            ring,
            ctx: f(max_tokens * HIDDEN)?,
            ctx_inv: f(max_tokens)?,
            kvp: f(max_tokens * n_layers * 2 * N_KV_HEADS * HEAD_DIM)?,
            ctx_ring: f((max_seqs * ctx_ring_len).max(1) * HIDDEN)?,
            ctx_ring_len,
            last: f(rows * HIDDEN)?,
            x: f(rows * HIDDEN)?,
            qkv: f(rows * QKV)?,
            inv: f(rows)?,
            attn: f(rows * HIDDEN)?,
            mixed: f(rows * HIDDEN)?,
            act: f(rows * FFN)?,
            hfin: f(rows * HIDDEN)?,
            cx: f(rows * HIDDEN)?,
            co: f(rows * HIDDEN)?,
            corr: f(rows * CONV_KC)?,
            corr_inv: f(rows)?,
            logits: f(rows * v)?,
            cand: f(rows * CANDS)?,
            cand_q: f(rows * CANDS)?,
            meta: f(2 * rows)?,
            tokens: f(rows)?,
            part: f(rows * PART_SLOTS * N_HEADS * super::kernels::PART_FLOATS)?,
            vdir: f(rows * N_KV_HEADS * HEAD_DIM)?,
            gates: f(rows * N_HEADS)?,
            sdpa: f(rows * HIDDEN)?,
            iota,
            zeros: f(N_HEADS * HEAD_DIM)?,
            sets: [set()?, set()?, set()?],
            p: DPipes {
                draft_kv: gpu.pipeline("draft_kv")?,
                draft_conv: gpu.pipeline("draft_conv")?,
                draft_ctx: gpu.pipeline("draft_ctx")?,
                draft_embed: gpu.pipeline("draft_embed")?,
                draft_top: gpu.pipeline_for("draft_top", 1024)?,
                draft_tree: gpu.pipeline("draft_tree")?,
            },
            w: w.buffer,
        })
    }

    fn kv_args(&self, layer: usize, max_seqs: usize) -> KvArgs {
        KvArgs {
            is_global: 0,
            kv_layer: layer as u32,
            ring: self.ring as u32,
            max_seqs: max_seqs as u32,
            page_shift: 0,
            n_pages: 0,
            max_pages: 0,
            window: WINDOW as u32,
        }
    }
}

impl Engine {
    /// Loads a drafter (tools/spec/export.py) and taps the layers it reads.
    pub fn load_drafter(&mut self, path: &Path) -> Result<(), String> {
        let d = Drafter::load(&self.gpu, path, self.cfg.max_seqs, self.cfg.max_step_tokens, self.cfg.max_chunk)?;
        self.set_taps(&d.taps.clone())?;
        self.drafter = Some(d);
        Ok(())
    }

    pub fn has_drafter(&self) -> bool {
        self.drafter.is_some()
    }

    /// For checking the drafter: the last block pass's first `rows` rows of logits, and
    /// slot `slot`'s GPU-fed tokens and draft tree.
    #[doc(hidden)]
    pub fn drafter_debug(&self, rows: usize, slot: u32) -> DrafterDebug {
        let d = self.drafter.as_ref().expect("a drafter");
        let at = slot as usize * FEED_W;
        DrafterDebug {
            logits: gpu::read(&d.logits, 0, rows * d.v),
            vocab: gpu::read(&d.vocab, 0, d.v),
            feed: gpu::read(&self.s.feed, at, FEED_W),
            tree: gpu::read(&self.s.tree, at, FEED_W),
        }
    }

    /// The drafter's pass after the step `plan`: context for all of the step's tokens, then
    /// a block for each of `runs`. `max_len`: the most positions any of their sequences can
    /// reach.
    pub(super) fn encode_drafter(&self, e: &Encoder, set: usize, plan: &Plan, runs: &[DraftRun], max_len: usize) {
        let Some(d) = &self.drafter else { return };
        let n = plan.n;
        // 1. Context.
        for (k, fc) in d.fc.iter().enumerate() {
            let ra = if k == 0 { 0.0 } else { 1.0 };
            let epi = Epilogue::Resid { scale: 1.0, ra, rb: 1.0, base: &d.ctx, raw: None };
            self.matmul_in(e, &d.w, fc, n, &self.tap_bufs[k], &d.ctx, epi);
        }
        self.matmul_in(e, &d.w, &d.kv_all, n, &d.ctx, &d.kvp, Epilogue::Norm { scale: 1.0, inv: &d.ctx_inv });
        e.pipeline(&d.p.draft_kv);
        e.buf(0, &d.kvp, 0);
        e.buf(1, &self.s.meta, 0);
        e.buf(2, &self.rope_tab, 0);
        e.buf(3, &d.k_ring, 0);
        e.buf(4, &d.v_ring, 0);
        e.bytes(5, &d.kv_args(0, self.cfg.max_seqs));
        e.buf(6, &self.s.on_path, 0);
        e.dispatch((n, d.layers.len(), 1), 64);
        if d.inject.is_some() {
            e.pipeline(&d.p.draft_ctx);
            e.buf(0, &d.ctx, 0);
            e.buf(1, &d.ctx_inv, 0);
            e.buf(2, &self.s.meta, 0);
            e.buf(3, &d.ctx_ring, 0);
            e.bytes(4, &(d.ctx_ring_len as u32));
            e.buf(5, &self.s.on_path, 0);
            e.dispatch((n, 1, 1), 256);
        }
        if runs.is_empty() {
            return;
        }
        // 2. Block rows.
        let ds = &d.sets[set];
        let (mut tin, mut vis, mut tiles) = (Vec::new(), Vec::new(), Vec::new());
        for r in runs {
            tiles.extend([tin.len() as u32 / 2, r.rows as u32]);
            for i in 0..r.rows as u32 {
                tin.extend([r.slot, i]);
                vis.push((1u32 << r.rows) - 1);
            }
        }
        let rows = vis.len();
        gpu::write(&ds.tin, 0, &tin);
        gpu::write(&ds.vis, 0, &vis);
        gpu::write(&ds.tiles, 0, &tiles);
        e.pipeline(&d.p.draft_embed);
        e.buf(0, &self.model.buffer, self.model.embed.offset);
        e.buf(1, &ds.tin, 0);
        e.buf(2, &self.s.lens, 0);
        e.buf(3, &self.s.feed, 0);
        e.buf(4, &d.w, d.mask_emb);
        e.buf(5, &d.x, 0);
        e.buf(6, &d.meta, 0);
        e.buf(7, &d.tokens, 0);
        e.buf(8, &d.ctx_ring, 0);
        e.buf(9, &d.last, 0);
        e.bytes(10, &(d.ctx_ring_len as u32));
        e.dispatch((rows, 1, 1), 256);
        if let Some(inj) = &d.inject {
            let epi = Epilogue::Resid { scale: 1.0, ra: 1.0, rb: 1.0, base: &d.x, raw: None };
            self.matmul_in(e, &d.w, inj, rows, &d.last, &d.x, epi);
        }
        let max_run = runs.iter().map(|r| r.rows).max().unwrap_or(1);
        let groups = (5 * max_run).div_ceil(8).clamp(1, 5);
        // At most the window before the block: 33 splits at most, within PART_SLOTS.
        let cached = max_len.min(WINDOW);
        let splits = decode_splits(runs.len(), cached, PART_SLOTS, DEC_TGS_PER_CORE * self.cores, self.cores, None);
        // out = resid + conv(x) within each block (or conv(x) alone).
        let conv = |c: &DConv, x: &Buffer, resid: Option<&Buffer>, out: &Buffer| {
            self.matmul_in(e, &d.w, &c.corr, rows, x, &d.corr, Epilogue::Norm { scale: 1.0, inv: &d.corr_inv });
            e.pipeline(&d.p.draft_conv);
            e.buf(0, x, 0);
            e.buf(1, &d.corr, 0);
            e.buf(2, &d.w, c.base);
            e.buf(3, &ds.tin, 0);
            e.buf(4, resid.unwrap_or(x), 0);
            e.buf(5, out, 0);
            e.bytes(6, &(resid.is_some() as u32));
            e.dispatch((rows, 1, 1), 256);
        };
        for (li, l) in d.layers.iter().enumerate() {
            let xin = match &l.convs {
                Some(c) => {
                    conv(&c[0], &d.x, None, &d.cx);
                    &d.cx
                }
                None => &d.x,
            };
            self.matmul_in(e, &d.w, &l.qkv, rows, xin, &d.qkv, Epilogue::Norm { scale: 1.0, inv: &d.inv });
            let a = SmallAttn {
                qkv: &d.qkv,
                x: xin,
                inv: &d.inv,
                meta: &d.meta,
                tokens: &d.tokens,
                tiles: &ds.tiles,
                vis: &ds.vis,
                tin: &ds.tin,
                part: &d.part,
                vdir: &d.vdir,
                gates: &d.gates,
                out: &d.attn,
                sdpa: &d.sdpa,
                cap_q: &d.sdpa,
                k: &d.k_ring,
                v: &d.v_ring,
                kv: d.kv_args(li, self.cfg.max_seqs),
                w: &d.zeros,
                ve: None,
                attn_gate: 0,
                xsa: 0,
                rope: true,
                capture: false,
                splits,
                shape: (groups, self.tuning.verify_ki[groups - 1]),
                max_run,
                n_runs: runs.len(),
                n: rows,
            };
            self.encode_small_attention(e, &a);
            let resid = |base| Epilogue::Resid { scale: 1.0, ra: 1.0, rb: 1.0, base, raw: None };
            match &l.convs {
                Some(c) => {
                    let plain = Epilogue::Resid { scale: 1.0, ra: 0.0, rb: 1.0, base: &d.x, raw: None };
                    self.matmul_in(e, &d.w, &l.o, rows, &d.attn, &d.co, plain);
                    conv(&c[1], &d.co, Some(&d.x), &d.mixed);
                    conv(&c[2], &d.mixed, None, &d.cx);
                    self.matmul_in(e, &d.w, &l.gate_up, rows, &d.cx, &d.act, Epilogue::SwiGlu);
                    let plain = Epilogue::Resid { scale: 1.0, ra: 0.0, rb: 1.0, base: &d.x, raw: None };
                    self.matmul_in(e, &d.w, &l.down, rows, &d.act, &d.co, plain);
                    conv(&c[3], &d.co, Some(&d.mixed), &d.x);
                }
                None => {
                    self.matmul_in(e, &d.w, &l.o, rows, &d.attn, &d.mixed, resid(&d.x));
                    self.matmul_in(e, &d.w, &l.gate_up, rows, &d.mixed, &d.act, Epilogue::SwiGlu);
                    self.matmul_in(e, &d.w, &l.down, rows, &d.act, &d.x, resid(&d.mixed));
                }
            }
        }
        // 3. Tree.
        e.pipeline(&self.p.rmsnorm_rows);
        e.buf(0, &d.x, 0);
        e.buf(1, &d.iota, 0);
        e.buf(2, &d.hfin, 0);
        e.dispatch((rows, 1, 1), 256);
        // Raw logits: through the softcap, a trained drafter's largest would tie.
        self.matmul_in(e, &d.w, &d.head, rows, &d.hfin, &d.logits, Epilogue::Raw);
        let ncand = CANDS.min(max_run - 1) as u32;
        e.pipeline(&d.p.draft_top);
        e.buf(0, &d.logits, 0);
        e.buf(1, &d.vocab, 0);
        e.buf(2, &ds.tin, 0);
        e.buf(3, &d.cand, 0);
        e.buf(4, &d.cand_q, 0);
        e.bytes(5, &(d.v as u32));
        e.bytes(6, &(1.0 / TRAIN_TEMPERATURE));
        e.bytes(7, &ncand);
        e.dispatch((rows, 1, 1), 1024);
        e.pipeline(&d.p.draft_tree);
        e.buf(0, &ds.tiles, 0);
        e.buf(1, &ds.tin, 0);
        e.buf(2, &d.cand, 0);
        e.buf(3, &d.cand_q, 0);
        e.buf(4, &self.s.feed, 0);
        e.buf(5, &self.s.tree, 0);
        e.bytes(6, &(runs.len() as u32));
        e.bytes(7, &ncand);
        e.dispatch((runs.len().div_ceil(32), 1, 1), 32);
    }
}
