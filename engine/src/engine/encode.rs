use std::f32::consts::LOG2_E;

use super::buffers::KPART_FLOATS;
use super::kernels::{
    AttnArgs, KvArgs, MatArgs, PostArgs, ARGMAX_BLOCK, EPI_HEAD, EPI_NORM, EPI_RAW, EPI_RESID, EPI_SWIGLU, FLAG_WRITE_INV,
    FLAG_WRITE_RAW, MAX_KSPLIT, SAMPLE_THREADS,
};
use super::plan::Plan;
use super::{skip, Engine, SPLITK_TGS_PER_CORE};
use crate::gpu::{Buffer, Encoder, Pipeline};
use crate::model::{
    Layer, Matrix, ATTN_SCALE, HEAD_DIM, HIDDEN, MUDD_LAYERS, N_HEADS, N_KV_HEADS, N_LAYERS, VOCAB, VOCAB_REAL,
};

/// A matmul kernel. The engine picks one by step size; `Engine::bench_matmul` forces any.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Matmul {
    /// 1, 2 or 4 rows per simdgroup, at most 8 tokens; for one token, the rows' k may be split
    /// over 2 or 4 simdgroups.
    Gemv { rows: usize, splits: usize },
    /// A token tile of 16, 32 or 64, with 1, 2 or 4 k tiles in flight.
    Gemm { tile: usize, k_tiles: usize },
    /// On the matrix units with the weights streamed into registers: 1 or 2 blocks of 8 rows
    /// and 1 or 2 tiles of 8 tokens per simdgroup, k split over 4 or 8 simdgroups.
    Stream { rows: usize, tiles: usize, splits: usize },
}

impl Matmul {
    /// The engine's choice for `n` tokens of `w`: the GEMV
    /// for one token, the streaming GEMM up to 8 (16 for int8), then the smallest GEMM token
    /// tile that holds the step. Past 64 tokens, 32-token tiles when the last 64-token tile
    /// would be at most half full.
    pub(super) fn for_tokens(n: usize, w: &Matrix) -> Self {
        match n {
            1 if w.int8.is_some() && w.k > HIDDEN => Matmul::Gemv { rows: 2, splits: 4 },
            1 => Matmul::Gemv { rows: 2, splits: 1 },
            2..=8 => Matmul::Stream { rows: if w.k > HIDDEN { 2 } else { 1 }, tiles: 1, splits: 8 },
            9..=16 if w.int8.is_some() => Matmul::Stream { rows: 2, tiles: 2, splits: 8 },
            9..=16 => Matmul::Gemm { tile: 16, k_tiles: 2 },
            17..=32 => Matmul::Gemm { tile: 32, k_tiles: 1 },
            65..=384 if (1..=32).contains(&(n % 64)) => Matmul::Gemm { tile: 32, k_tiles: 1 },
            _ => Matmul::Gemm { tile: 64, k_tiles: 1 },
        }
    }
}

/// The buffers and shape of one small-run attention (attn_verify, then attn_dec_reduce):
/// the target's layers', or the drafter's.
pub(super) struct SmallAttn<'a> {
    pub qkv: &'a Buffer,
    /// The attention input before its norm (for the gates and value embeddings).
    pub x: &'a Buffer,
    pub inv: &'a Buffer,
    pub meta: &'a Buffer,
    pub tokens: &'a Buffer,
    pub tiles: &'a Buffer,
    pub vis: &'a Buffer,
    pub tin: &'a Buffer,
    pub part: &'a Buffer,
    pub vdir: &'a Buffer,
    pub gates: &'a Buffer,
    pub out: &'a Buffer,
    pub sdpa: &'a Buffer,
    pub cap_q: &'a Buffer,
    pub k: &'a Buffer,
    pub v: &'a Buffer,
    pub kv: KvArgs,
    /// The weight buffer holding ve (value-embedding gate), attn_gate and xsa.
    pub w: &'a Buffer,
    pub ve: Option<usize>,
    pub attn_gate: usize,
    pub xsa: usize,
    pub rope: bool,
    pub capture: bool,
    pub splits: (u32, u32),
    pub shape: (usize, usize),
    pub max_run: usize,
    pub n_runs: usize,
    pub n: usize,
}

/// What a matmul writes for each output, and the buffers that takes.
pub(super) enum Epilogue<'a> {
    /// scale * acc / rms(x), the rms_norm of the input folded in. Also writes 1/rms(x) per
    /// token to `inv`.
    Norm { scale: f32, inv: &'a Buffer },
    /// ra * base + rb * scale * acc; with `raw`, also scale * acc there.
    Resid { scale: f32, ra: f32, rb: f32, base: &'a Buffer, raw: Option<&'a Buffer> },
    /// silu(gate) * up over the interleaved rows, the rms_norm of the input folded in.
    SwiGlu,
    /// Softcapped logits, the padded ids at -inf.
    Head,
    /// The accumulator as it is.
    Raw,
}

impl Engine {
    /// The whole forward pass.
    pub(super) fn encode_forward(&self, e: &Encoder, set: usize, plan: &Plan) {
        self.encode_embed(e, set, plan);
        for li in 0..N_LAYERS {
            self.encode_layer(e, set, plan, li, false);
        }
        self.encode_tail(e, set, plan);
    }

    pub(super) fn resid_in(&self, li: usize) -> &Buffer {
        match li {
            0 => &self.s.hist0,
            12 => &self.s.hist12,
            23 => &self.s.hist23,
            _ => &self.s.hmain,
        }
    }

    fn resid_out(&self, li: usize) -> &Buffer {
        match li {
            11 => &self.s.hist12,
            22 => &self.s.hist23,
            _ => &self.s.hmain,
        }
    }

    fn on(&self, group: u32) -> bool {
        self.skip & group == 0
    }

    pub(super) fn encode_embed(&self, e: &Encoder, set: usize, plan: &Plan) {
        e.pipeline(&self.p.embed);
        e.buf(0, &self.model.buffer, self.model.embed.offset);
        e.buf(1, &self.sets[set].tokens, 0);
        e.buf(2, &self.s.hist0, 0);
        e.buf(3, &self.sets[set].tin, 0);
        e.buf(4, &self.s.meta, 0);
        e.buf(5, &self.s.lens, 0);
        e.buf(6, &self.s.feed, 0);
        e.buf(7, &self.s.tree, 0);
        e.buf(8, &self.sets[set].vis, 0);
        e.dispatch((plan.n, 1, 1), 256);
    }

    pub(super) fn encode_layer(&self, e: &Encoder, set: usize, plan: &Plan, li: usize, capture: bool) {
        let (l, s, n) = (&self.model.layers[li], &self.s, plan.n);
        let h = self.resid_in(li);
        let (x_attn, base) = match MUDD_LAYERS.iter().position(|&m| m == li) {
            Some(site) if self.on(skip::MUDD) => {
                e.pipeline(&self.p.mudd);
                e.buf(0, h, 0);
                e.buf(1, &s.hist0, 0);
                e.buf(2, if site == 0 { &s.hist12 } else { &s.hist23 }, 0);
                e.buf(3, &self.model.buffer, self.model.mudd_dense1);
                e.buf(4, &self.model.buffer, self.model.mudd_mix[site]);
                e.buf(5, &s.xattn, 0);
                e.buf(6, &s.base, 0);
                e.dispatch((n, 1, 1), 256);
                (&s.xattn, &s.base)
            }
            _ => (h, h),
        };
        if capture {
            self.rmsnorm_rows(e, x_attn, &s.iota, &s.cap_attn_in, n);
        }
        if self.on(skip::QKV) {
            self.matmul(e, &l.qkv, n, x_attn, &s.qkv, Epilogue::Norm { scale: l.qkv_scale, inv: &s.inv });
        }
        if plan.fused {
            self.encode_verify_attention(e, set, plan, l, x_attn, capture);
        } else {
            self.encode_chunk_attention(e, set, plan, l, x_attn, capture);
        }
        let raw = |b| capture.then_some(b);
        if self.on(skip::O) {
            let epi = Epilogue::Resid { scale: l.o_scale, ra: l.resid_attn, rb: l.post_attn, base, raw: raw(&s.cap_attn_out) };
            self.matmul(e, &l.o, n, &s.attn, &s.mixed, epi);
        }
        if capture {
            self.rmsnorm_rows(e, &s.mixed, &s.iota, &s.cap_mlp_in, n);
        }
        if self.on(skip::GATE_UP) {
            self.matmul(e, &l.gate_up, n, &s.mixed, &s.act, Epilogue::SwiGlu);
        }
        if self.on(skip::DOWN) {
            let epi = Epilogue::Resid { scale: 1.0, ra: l.resid_mlp, rb: l.post_mlp, base: &s.mixed, raw: raw(&s.cap_mlp_out) };
            self.matmul(e, &l.down, n, &s.act, self.resid_out(li), epi);
        }
        if let Some(k) = self.taps.iter().position(|&t| t == li) {
            e.pipeline(&self.p.tap_rows);
            e.buf(0, self.resid_out(li), 0);
            e.buf(1, &self.tap_bufs[k], 0);
            e.dispatch((n, 1, 1), 256);
        }
    }

    /// attn_verify, then attn_dec_reduce: everything from the qkv matmul to the attention
    /// output in two dispatches, for runs of at most 8 tokens.
    fn encode_verify_attention(&self, e: &Encoder, set: usize, plan: &Plan, l: &Layer, x: &Buffer, capture: bool) {
        let (s, st, w) = (&self.s, &self.sets[set], &self.model.buffer);
        let (k, v) = self.kv.buffers(l);
        let a = SmallAttn {
            qkv: &s.qkv,
            x,
            inv: &s.inv,
            meta: &s.meta,
            tokens: &st.tokens,
            tiles: &st.tiles,
            vis: &st.vis,
            tin: &st.tin,
            part: &s.part,
            vdir: &s.vdir,
            gates: &s.gates,
            out: &s.attn,
            sdpa: &s.cap_sdpa,
            cap_q: &s.cap_q,
            k,
            v,
            kv: self.kv.args(l),
            w,
            ve: l.ve_gate,
            attn_gate: l.attn_gate,
            xsa: l.xsa,
            rope: !l.global,
            capture,
            splits: plan.splits(l.global),
            shape: plan.verify_shape,
            max_run: plan.max_run,
            n_runs: plan.n_runs,
            n: plan.n,
        };
        self.encode_small_attention(e, &a);
    }

    /// attn_verify and attn_dec_reduce over the buffers of `a`.
    pub(super) fn encode_small_attention(&self, e: &Encoder, a: &SmallAttn) {
        let (split, max_splits) = a.splits;
        let (groups, ki) = a.shape;
        // Threadgroup memory: Q^T (16 x 64 floats per column group) and the new keys and
        // values (1024 floats), then the sums of the simdgroups that share a group: in one
        // round if they fit in 20 KB, else halving each round.
        let one_round = ki >= 4 && ki * 5 * a.max_run * HEAD_DIM * 4 <= 20480;
        // One round keeps only the valid columns, 5 per token.
        let cols = if one_round { ki * 5 * a.max_run } else { ki / 2 * groups * 8 };
        let tgm = (groups * 16 * 64 + 1024).max(cols * HEAD_DIM);
        let args = AttnArgs {
            split,
            max_splits,
            scale_log2: ATTN_SCALE * LOG2_E,
            rope: a.rope as u32,
            ve: a.ve.is_some() as u32,
            capture: a.capture as u32,
            fixed_splits: 0,
            col_sets: groups as u32,
            one_round: one_round as u32,
        };
        if self.on(skip::ATTN) {
            e.pipeline(&self.p.attn_verify[ki.trailing_zeros() as usize]);
            e.buf(0, a.qkv, 0);
            e.buf(1, a.x, 0);
            e.buf(2, a.inv, 0);
            e.buf(3, a.meta, 0);
            e.buf(4, a.tokens, 0);
            e.buf(5, &self.model.buffer, self.model.value_embed);
            match a.ve {
                Some(offset) => e.buf(6, a.w, offset),
                None => e.buf(6, &self.s.dummy, 0),
            }
            e.buf(7, a.w, a.attn_gate);
            e.buf(8, a.k, 0);
            e.buf(9, a.v, 0);
            e.buf(10, &self.kv.page_table, 0);
            e.buf(11, &self.rope_tab, 0);
            e.buf(12, a.part, 0);
            e.buf(13, a.vdir, 0);
            e.buf(14, a.gates, 0);
            e.bytes(15, &a.kv);
            e.bytes(16, &args);
            e.buf(17, a.tiles, 0);
            e.buf(18, a.vis, 0);
            e.buf(19, a.cap_q, 0);
            e.tg_mem(0, 4 * tgm);
            e.tg_mem(1, 4 * cols.max(1));
            e.dispatch((max_splits as usize, N_KV_HEADS, a.n_runs), 32 * groups * ki);
        }
        if self.on(skip::REDUCE) {
            e.pipeline(&self.p.attn_dec_reduce);
            e.buf(0, a.part, 0);
            e.buf(1, a.meta, 0);
            e.buf(2, a.vdir, 0);
            e.buf(3, a.gates, 0);
            e.buf(4, a.w, a.xsa);
            e.buf(5, a.out, 0);
            e.buf(6, a.sdpa, 0);
            e.bytes(7, &a.kv);
            e.bytes(8, &args);
            e.buf(9, a.tin, 0);
            e.dispatch((a.n, N_HEADS, 1), 512);
        }
    }

    /// qkv_post, attn_mma, then attn_sum_reduce.
    fn encode_chunk_attention(&self, e: &Encoder, set: usize, plan: &Plan, l: &Layer, x: &Buffer, capture: bool) {
        let (s, st, w) = (&self.s, &self.sets[set], &self.model.buffer);
        let kv = self.kv.args(l);
        let (k, v) = self.kv.buffers(l);
        if self.on(skip::POST) {
            e.pipeline(&self.p.qkv_post);
            e.buf(0, &s.qkv, 0);
            e.buf(1, x, 0);
            e.buf(2, &s.inv, 0);
            e.buf(3, &s.meta, 0);
            e.buf(4, &st.tokens, 0);
            e.buf(5, w, self.model.value_embed);
            self.bind_ve_gate(e, 6, l);
            e.buf(7, w, l.attn_gate);
            e.buf(8, k, 0);
            e.buf(9, v, 0);
            e.buf(10, &s.vdir, 0);
            e.buf(11, &s.gates, 0);
            e.buf(12, &self.kv.page_table, 0);
            e.buf(13, &self.rope_tab, 0);
            e.bytes(14, &kv);
            e.bytes(15, &PostArgs { rope: !l.global as u32, ve: l.ve_gate.is_some() as u32 });
            e.buf(16, &st.tin, 0);
            e.dispatch((plan.n, 1, 1), 32 * (N_HEADS + N_KV_HEADS));
        }
        let (split, max_splits) = plan.splits(l.global);
        let args = AttnArgs {
            split,
            max_splits,
            scale_log2: ATTN_SCALE * LOG2_E,
            rope: 0,
            ve: 0,
            capture: capture as u32,
            fixed_splits: max_splits,
            col_sets: 0,
            one_round: 0,
        };
        if self.on(skip::ATTN) {
            e.pipeline(&self.p.attn_mma);
            e.buf(0, &s.qkv, 0);
            e.buf(1, &s.meta, 0);
            e.buf(2, k, 0);
            e.buf(3, v, 0);
            e.buf(4, &self.kv.page_table, 0);
            e.buf(5, &s.part, 0);
            e.buf(6, &st.tiles, 0);
            e.bytes(7, &kv);
            e.bytes(8, &args);
            e.buf(9, &st.vis, 0);
            e.dispatch((plan.tiles, N_KV_HEADS, max_splits as usize), 32 * (N_HEADS / N_KV_HEADS));
        }
        if self.on(skip::REDUCE) {
            self.bind_attn_reduce(e, set, l, &kv, &args, &self.p.attn_sum_reduce);
            e.dispatch((plan.n, 1, 1), 32 * N_HEADS);
        }
    }

    fn bind_ve_gate(&self, e: &Encoder, index: usize, l: &Layer) {
        match l.ve_gate {
            Some(offset) => e.buf(index, &self.model.buffer, offset),
            None => e.buf(index, &self.s.dummy, 0),
        }
    }

    /// The two reduces take the same arguments: they sum the partials, then apply XSA and
    /// the gate.
    fn bind_attn_reduce(&self, e: &Encoder, set: usize, l: &Layer, kv: &KvArgs, args: &AttnArgs, p: &Pipeline) {
        let s = &self.s;
        e.pipeline(p);
        e.buf(0, &s.part, 0);
        e.buf(1, &s.meta, 0);
        e.buf(2, &s.vdir, 0);
        e.buf(3, &s.gates, 0);
        e.buf(4, &self.model.buffer, l.xsa);
        e.buf(5, &s.attn, 0);
        e.buf(6, &s.cap_sdpa, 0);
        e.bytes(7, kv);
        e.bytes(8, args);
        e.buf(9, &self.sets[set].tin, 0);
    }

    /// Final norm, head, each head row's token, then `accept`; if a verify run can branch,
    /// kv_compact then moves its path's K/V into place.
    pub(super) fn encode_tail(&self, e: &Encoder, set: usize, plan: &Plan) {
        let (s, st) = (&self.s, &self.sets[set]);
        let h = self.resid_out(N_LAYERS - 1);
        if plan.n_hidden > 0 {
            self.rmsnorm_rows(e, h, &st.hidden_rows, &s.hfin_all, plan.n_hidden);
        }
        let rows = plan.n_head;
        if rows > 0 && self.on(skip::HEAD) {
            self.rmsnorm_rows(e, h, &st.head_rows, &s.hfin_head, rows);
            self.matmul(e, &self.model.head, rows, &s.hfin_head, &s.logits, Epilogue::Head);
            self.encode_sample(e, set, rows);
        }
        e.pipeline(&self.p.accept);
        e.buf(0, &st.runs, 0);
        e.buf(1, &st.tokens, 0);
        e.buf(2, &st.out_ids, 0);
        e.buf(3, &s.lens, 0);
        e.buf(4, &s.feed, 0);
        e.buf(5, &s.gen, 0);
        e.buf(6, &st.emitted, 0);
        e.bytes(7, &(plan.n_runs as u32));
        e.buf(8, &s.tree, 0);
        e.buf(9, &s.path, 0);
        e.buf(10, &s.path_len, 0);
        e.buf(11, &s.on_path, 0);
        e.dispatch((plan.n_runs.div_ceil(64), 1, 1), 64);
        if plan.branches {
            self.encode_kv_compact(e, set, plan);
        }
    }

    /// kv_compact: each verify run's accepted path's K/V to its positions, on every layer.
    fn encode_kv_compact(&self, e: &Encoder, set: usize, plan: &Plan) {
        let layers = &self.model.layers;
        let local = layers.iter().find(|l| !l.global).expect("a local layer");
        let global = layers.iter().find(|l| l.global).expect("a global layer");
        let n_local = layers.iter().filter(|l| !l.global).count();
        let ((kl, vl), (kg, vg)) = (self.kv.buffers(local), self.kv.buffers(global));
        e.pipeline(&self.p.kv_compact);
        e.buf(0, &self.sets[set].runs, 0);
        e.buf(1, &self.s.path, 0);
        e.buf(2, &self.s.path_len, 0);
        e.buf(3, &self.s.lens, 0);
        e.buf(4, kl, 0);
        e.buf(5, vl, 0);
        e.buf(6, kg, 0);
        e.buf(7, vg, 0);
        e.buf(8, &self.kv.page_table, 0);
        e.bytes(9, &self.kv.args(local));
        e.bytes(10, &self.kv.args(global));
        e.bytes(11, &(n_local as u32));
        e.dispatch((plan.n_runs, layers.len(), 1), 32 * N_KV_HEADS);
    }

    /// The replay drafter, after `accept`: the next step's drafts of every slot with a
    /// recorded sequence.
    pub(super) fn encode_replay(&self, e: &Encoder, set: usize, plan: &Plan, args: &super::kernels::ReplayArgs) {
        let s = &self.s;
        e.pipeline(&self.p.replay_draft);
        e.buf(0, &self.sets[set].runs, 0);
        e.buf(1, &s.lens, 0);
        e.buf(2, &s.feed, 0);
        e.buf(3, &s.trace, 0);
        e.buf(4, &s.trace_len, 0);
        e.bytes(5, args);
        e.bytes(6, &(plan.n_runs as u32));
        e.buf(7, &s.tree, 0);
        e.dispatch((plan.n_runs, 1, 1), super::kernels::FEED_W);
    }

    /// argmax_partial, then sample: `rows` tokens from the logits.
    pub(super) fn encode_sample(&self, e: &Encoder, set: usize, rows: usize) {
        let (s, st) = (&self.s, &self.sets[set]);
        let blocks = VOCAB.div_ceil(ARGMAX_BLOCK);
        e.pipeline(&self.p.argmax_partial);
        e.buf(0, &s.logits, 0);
        e.buf(1, &s.argmax_part, 0);
        e.bytes(2, &(VOCAB as u32));
        e.dispatch((blocks, rows, 1), 256);
        e.pipeline(&self.p.sample);
        e.buf(0, &s.logits, 0);
        e.buf(1, &s.argmax_part, 0);
        e.buf(2, &st.sample_rows, 0);
        e.buf(3, &st.out_ids, 0);
        e.buf(4, &s.meta, 0);
        e.bytes(5, &(blocks as u32));
        e.bytes(6, &(VOCAB as u32));
        e.dispatch((rows, 1, 1), SAMPLE_THREADS);
    }

    fn rmsnorm_rows(&self, e: &Encoder, x: &Buffer, rows: &Buffer, out: &Buffer, n: usize) {
        e.pipeline(&self.p.rmsnorm_rows);
        e.buf(0, x, 0);
        e.buf(1, rows, 0);
        e.buf(2, out, 0);
        e.dispatch((n, 1, 1), 256);
    }

    /// y = epilogue(w . x) for `n` tokens.
    pub(super) fn matmul(&self, e: &Encoder, w: &Matrix, n: usize, x: &Buffer, y: &Buffer, epi: Epilogue) {
        self.matmul_in(e, &self.model.buffer, w, n, x, y, epi);
    }

    /// `matmul` with a matrix of another weight buffer (the drafter's).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn matmul_in(&self, e: &Encoder, wb: &Buffer, w: &Matrix, n: usize, x: &Buffer, y: &Buffer, epi: Epilogue) {
        let s = &self.s;
        let plain = MatArgs {
            n: n as u32,
            k: w.k as u32,
            m: w.m as u32,
            epi: 0,
            flags: 0,
            vocab_real: VOCAB_REAL as u32,
            scale: 1.0,
            ra: 0.0,
            rb: 0.0,
            ksplit: 1,
            kchunk: w.k as u32,
            gshift: w.int8.map_or(0, |q| q.group.trailing_zeros()),
        };
        let (args, base, aux) = match epi {
            Epilogue::Norm { scale, inv } => (MatArgs { epi: EPI_NORM, flags: FLAG_WRITE_INV, scale, ..plain }, &s.dummy, inv),
            Epilogue::Resid { scale, ra, rb, base, raw } => {
                let flags = if raw.is_some() { FLAG_WRITE_RAW } else { 0 };
                (MatArgs { epi: EPI_RESID, flags, scale, ra, rb, ..plain }, base, raw.unwrap_or(&s.dummy))
            }
            Epilogue::SwiGlu => (MatArgs { epi: EPI_SWIGLU, ..plain }, &s.dummy, &s.dummy),
            Epilogue::Head => (MatArgs { epi: EPI_HEAD, ..plain }, &s.dummy, &s.dummy),
            Epilogue::Raw => (MatArgs { epi: EPI_RAW, ..plain }, &s.dummy, &s.dummy),
        };
        let bind = |args: &MatArgs| {
            e.buf(0, wb, w.offset);
            e.buf(1, x, 0);
            e.buf(2, y, 0);
            e.buf(3, base, 0);
            e.buf(4, aux, 0);
            e.bytes(5, args);
            match w.int8 {
                Some(q) => e.buf(8, wb, q.scales),
                None => e.buf(8, &s.dummy, 0),
            }
        };
        let fi = w.int8.is_some() as usize;
        match self.force_matmul.unwrap_or(Matmul::for_tokens(n, w)) {
            Matmul::Gemv { rows, splits } => {
                let per_pass = n.next_power_of_two();
                let (ki, ri) = (if w.k == HIDDEN { 0 } else { 1 }, rows.trailing_zeros() as usize);
                e.pipeline(match splits {
                    1 => &self.p.gemv[fi][ki][ri][per_pass.trailing_zeros() as usize],
                    _ => &self.p.gemv_split[fi][ki][ri][splits.trailing_zeros() as usize - 1],
                });
                bind(&args);
                e.dispatch((w.m.div_ceil(rows * 8 / splits), n.div_ceil(per_pass), 1), 256);
            }
            Matmul::Stream { rows, tiles, splits } => {
                let ki = if w.k == HIDDEN { 0 } else { 1 };
                let (ri, ti, si) = (rows.trailing_zeros() as usize, tiles.trailing_zeros() as usize, splits.trailing_zeros() as usize - 2);
                e.pipeline(&self.p.gemm_stream[fi][ki][ri][ti][si]);
                bind(&args);
                e.dispatch((w.m.div_ceil(rows * 64 / splits), n.div_ceil(8 * tiles), 1), 256);
            }
            Matmul::Gemm { tile, k_tiles } => {
                // Split K when the tiles alone can't fill the GPU.
                let tiles = n.div_ceil(tile) * (w.m / 64);
                let units = w.k / (k_tiles * 16);
                let target = self.tuning.splitk_target.unwrap_or(SPLITK_TGS_PER_CORE * self.cores);
                let ks = if tiles < target { target.div_ceil(tiles) } else { 1 };
                let ks = ks.min(MAX_KSPLIT).min(units).min((KPART_FLOATS / (n * w.m)).max(1));
                let kchunk = units.div_ceil(ks) * k_tiles * 16;
                let ks = w.k.div_ceil(kchunk);
                let args = MatArgs { ksplit: ks as u32, kchunk: kchunk as u32, ..args };
                e.pipeline(&self.p.gemm[fi][(tile / 16).trailing_zeros() as usize][k_tiles.trailing_zeros() as usize]);
                bind(&args);
                e.buf(6, &s.kpart, 0);
                e.buf(7, &s.kss, 0);
                e.dispatch((n.div_ceil(tile), w.m / 64, ks), 128);
                if ks > 1 {
                    e.pipeline(&self.p.gemm_splitk_reduce);
                    e.buf(0, &s.kpart, 0);
                    e.buf(1, &s.kss, 0);
                    e.buf(2, y, 0);
                    e.buf(3, base, 0);
                    e.buf(4, aux, 0);
                    e.bytes(5, &args);
                    e.dispatch((w.m.div_ceil(512), n, 1), 256);
                }
            }
        }
    }
}
