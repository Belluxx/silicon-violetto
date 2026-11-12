use super::kernels::{chain_node, GenState, RunIn, SampleRow, ARGMAX_BLOCK, FEED_W, MAX_KSPLIT, PART_FLOATS};
use super::EngineConfig;
use crate::gpu::{self, Buffer, Gpu};
use crate::model::{FFN, HEAD_DIM, HIDDEN, N_HEADS, N_KV_HEADS, QKV, VOCAB};

/// Attention partials per head and token of a full step. Smaller steps give a token more.
pub const PART_SLOTS_PER_TOKEN: usize = 16;
/// Split-K partials, in floats.
pub const KPART_FLOATS: usize = 1 << 22;

/// Activations, shared by all steps: they run in order on one queue.
pub struct Scratch {
    /// The residual stream: layer 0's input, the outputs of layers 11 and 22 (kept for
    /// MUDD), and every other layer's.
    pub hist0: Buffer,
    pub hist12: Buffer,
    pub hist23: Buffer,
    pub hmain: Buffer,
    /// MUDD's attention input and residual base.
    pub xattn: Buffer,
    pub base: Buffer,
    pub qkv: Buffer,
    /// 1/rms of each token's attention input.
    pub inv: Buffer,
    /// XSA value directions and attention gates.
    pub vdir: Buffer,
    pub gates: Buffer,
    pub part: Buffer,
    /// Split-K partials ([ksplit][n][m]) and sums of squares ([ksplit][n]).
    pub kpart: Buffer,
    pub kss: Buffer,
    pub attn: Buffer,
    pub mixed: Buffer,
    pub act: Buffer,
    /// Final normed hidden states: head rows and kept rows.
    pub hfin_head: Buffer,
    pub hfin_all: Buffer,
    pub logits: Buffer,
    pub argmax_part: Buffer,
    /// (sequence slot, position) per token, which `embed` resolves from `tin` and `lens`.
    pub meta: Buffer,
    /// Sequence state on the GPU, per slot (see `embed` and `accept`): the position of its
    /// next input, its GPU-fed tokens and their draft tree, and its generation state.
    pub lens: Buffer,
    pub feed: Buffer,
    pub tree: Buffer,
    pub gen: Buffer,
    /// What `accept` leaves for kv_compact and the drafter: each verify run's accepted path
    /// and its length, and per token whether it's now in its sequence.
    pub path: Buffer,
    pub path_len: Buffer,
    pub on_path: Buffer,
    /// Each slot's recorded sequence for the replay drafter, and its length (0: none).
    pub trace: Buffer,
    pub trace_len: Buffer,
    pub iota: Buffer,
    pub cap_attn_in: Buffer,
    /// q after norm and RoPE, from attn_verify (which leaves qkv's q raw).
    pub cap_q: Buffer,
    pub cap_sdpa: Buffer,
    pub cap_attn_out: Buffer,
    pub cap_mlp_in: Buffer,
    pub cap_mlp_out: Buffer,
    pub dummy: Buffer,
}

impl Scratch {
    pub fn new(gpu: &Gpu, cfg: &EngineConfig) -> Result<Self, String> {
        let n = cfg.max_step_tokens;
        let rows = cfg.max_head_rows;
        let f = |floats: usize| gpu.buffer(floats * 4);
        let s = Self {
            hist0: f(n * HIDDEN)?,
            hist12: f(n * HIDDEN)?,
            hist23: f(n * HIDDEN)?,
            hmain: f(n * HIDDEN)?,
            xattn: f(n * HIDDEN)?,
            base: f(n * HIDDEN)?,
            qkv: f(n * QKV)?,
            inv: f(n)?,
            vdir: f(n * N_KV_HEADS * HEAD_DIM)?,
            gates: f(n * N_HEADS)?,
            part: f(n * PART_SLOTS_PER_TOKEN * N_HEADS * PART_FLOATS)?,
            kpart: f(KPART_FLOATS)?,
            kss: f(MAX_KSPLIT * n)?,
            attn: f(n * HIDDEN)?,
            mixed: f(n * HIDDEN)?,
            act: f(n * FFN)?,
            hfin_head: f(rows * HIDDEN)?,
            hfin_all: f(n * HIDDEN)?,
            logits: f(rows * VOCAB)?,
            argmax_part: f(rows * VOCAB.div_ceil(ARGMAX_BLOCK) * 2)?,
            meta: f(2 * n)?,
            lens: f(cfg.max_seqs)?,
            feed: f(cfg.max_seqs * FEED_W)?,
            tree: f(cfg.max_seqs * FEED_W)?,
            gen: gpu.buffer(cfg.max_seqs * size_of::<GenState>())?,
            path: f(cfg.max_seqs * FEED_W)?,
            path_len: f(cfg.max_seqs)?,
            on_path: f(n)?,
            trace: f(cfg.max_seqs * cfg.max_seq_len)?,
            trace_len: f(cfg.max_seqs)?,
            iota: f(n)?,
            cap_attn_in: f(n * HIDDEN)?,
            cap_q: f(n * HIDDEN)?,
            cap_sdpa: f(n * HIDDEN)?,
            cap_attn_out: f(n * HIDDEN)?,
            cap_mlp_in: f(n * HIDDEN)?,
            cap_mlp_out: f(n * HIDDEN)?,
            dummy: f(64)?,
        };
        gpu::write(&s.iota, 0, &(0..n as u32).collect::<Vec<_>>());
        let chains: Vec<u32> = (0..cfg.max_seqs * FEED_W).map(|i| chain_node((i % FEED_W) as u32)).collect();
        gpu::write(&s.tree, 0, &chains);
        Ok(s)
    }
}

/// One step's inputs, written by the CPU, and its outputs.
pub struct StepSet {
    /// Input ids; FEED_ROW + i for the slot's GPU-fed token i.
    pub tokens: Buffer,
    /// (sequence slot, row in its run) per token.
    pub tin: Buffer,
    /// Per token, the keys of its tile it sees (bit j: the tile's token j).
    pub vis: Buffer,
    pub head_rows: Buffer,
    pub hidden_rows: Buffer,
    /// Each head row's sampling settings, and the token it got.
    pub sample_rows: Buffer,
    pub out_ids: Buffer,
    /// Query tiles: (first token, tokens <= 8), never across sequences.
    pub tiles: Buffer,
    /// The runs, for `accept`, and the tokens each emitted.
    pub runs: Buffer,
    pub emitted: Buffer,
}

impl StepSet {
    pub fn new(gpu: &Gpu, cfg: &EngineConfig) -> Result<Self, String> {
        let n = cfg.max_step_tokens;
        let f = |words: usize| gpu.buffer(words * 4);
        Ok(Self {
            tokens: f(n)?,
            tin: f(2 * n)?,
            vis: f(n)?,
            head_rows: f(cfg.max_head_rows)?,
            hidden_rows: f(n)?,
            sample_rows: gpu.buffer(cfg.max_head_rows * size_of::<SampleRow>())?,
            out_ids: f(cfg.max_head_rows)?,
            tiles: f(2 * (n / 8 + cfg.max_seqs + 1))?,
            runs: gpu.buffer(cfg.max_seqs * size_of::<RunIn>())?,
            emitted: f(cfg.max_seqs)?,
        })
    }
}
