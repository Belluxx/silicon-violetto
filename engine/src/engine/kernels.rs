use super::Tuning;
use crate::gpu::{Gpu, Pipeline};

pub const EPI_NORM: u32 = 1;
pub const EPI_RESID: u32 = 2;
pub const EPI_SWIGLU: u32 = 3;
pub const EPI_HEAD: u32 = 4;
pub const EPI_RAW: u32 = 5;
pub const FLAG_WRITE_INV: u32 = 1;
pub const FLAG_WRITE_RAW: u32 = 2;
pub const PART_FLOATS: usize = 132;
pub const ARGMAX_BLOCK: usize = 4096;
pub const SAMPLE_THREADS: usize = 1024;
/// Input token `FEED_ROW + i` is the GPU-fed token i of its sequence slot: 0 is the slot's
/// next input, 1.. are drafts (see `embed`).
pub const FEED_ROW: u32 = 0xffff_0000;
/// GPU-fed tokens per slot.
pub const FEED_W: usize = 16;

/// Row `row` of a chain, as a draft tree's node (see `embed`).
pub fn chain_node(row: u32) -> u32 {
    let parent = if row == 0 { 0xff } else { row - 1 };
    row | parent << 8 | ((2 << row) - 1) << 16
}

/// `SampleRow::tok` for a row that isn't one of the step's tokens.
pub const NO_TOK: u32 = u32::MAX;
/// Tokens per run that attn_verify takes.
pub const VERIFY_T: usize = 8;
/// Stop tokens per slot that `accept` checks.
pub const MAX_STOP: usize = 8;
pub const MAX_KSPLIT: usize = 8;
/// Weight formats, as the kernel names spell them.
const FORMATS: [&str; 2] = ["bf16", "i8"];

#[repr(C)]
#[derive(Clone, Copy)]
pub struct MatArgs {
    pub n: u32,
    pub k: u32,
    pub m: u32,
    pub epi: u32,
    pub flags: u32,
    pub vocab_real: u32,
    pub scale: f32,
    pub ra: f32,
    pub rb: f32,
    pub ksplit: u32,
    pub kchunk: u32,
    pub gshift: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct KvArgs {
    pub is_global: u32,
    pub kv_layer: u32,
    pub ring: u32,
    pub max_seqs: u32,
    pub page_shift: u32,
    pub n_pages: u32,
    pub max_pages: u32,
    pub window: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PostArgs {
    pub rope: u32,
    pub ve: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AttnArgs {
    pub split: u32,
    pub max_splits: u32,
    pub scale_log2: f32,
    pub rope: u32,
    pub ve: u32,
    pub capture: u32,
    pub fixed_splits: u32,
    pub col_sets: u32,
    pub one_round: u32,
}

/// One head row's sampling settings.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SampleRow {
    pub seed_lo: u32,
    pub seed_hi: u32,
    /// The drawn token's position, if `tok` is NO_TOK.
    pub pos: u32,
    pub temperature: f32,
    pub top_p: f32,
    /// The step's token whose head row this is: the draw is at its position + 1.
    pub tok: u32,
}

/// One run of a step, for `accept`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RunIn {
    pub slot: u32,
    pub first: u32,
    pub count: u32,
    pub head0: u32,
    pub nhead: u32,
    pub verify: u32,
}

/// A slot's generation state on the GPU, which `accept` keeps.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GenState {
    pub remaining: u32,
    pub done: u32,
    pub nstop: u32,
    pub pad: u32,
    pub stop: [u32; MAX_STOP],
}

impl GenState {
    /// Never done: what plain steps (parity, bench) run under.
    pub const OPEN: GenState = GenState { remaining: u32::MAX, done: 0, nstop: 0, pad: 0, stop: [0; MAX_STOP] };
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ReplayArgs {
    pub stride: u32,
    pub seed: u32,
    pub rate: f32,
    pub vocab: u32,
}

pub struct Pipes {
    pub embed: Pipeline,
    pub rmsnorm_rows: Pipeline,
    pub tap_rows: Pipeline,
    /// [format: bf16, int8][k: 1280, 3328][rows per simdgroup: 1, 2, 4][tokens per pass: 1, 2, 4, 8]
    pub gemv: [[[[Pipeline; 4]; 3]; 2]; 2],
    /// One token, k split over simdgroups: [format][k][rows per simdgroup][splits: 2, 4]
    pub gemv_split: [[[[Pipeline; 2]; 3]; 2]; 2],
    /// [format: bf16, int8][token tile: 16, 32, 64][k tiles in flight: 1, 2, 4]
    pub gemm: [[[Pipeline; 3]; 3]; 2],
    /// [format][k: 1280, 3328][row blocks per simdgroup: 1, 2][tiles of 8 tokens: 1, 2][k splits: 4, 8]
    pub gemm_stream: [[[[[Pipeline; 2]; 2]; 2]; 2]; 2],
    pub gemm_splitk_reduce: Pipeline,
    pub qkv_post: Pipeline,
    /// Simdgroups per column group: 1, 2, 4.
    pub attn_verify: [Pipeline; 3],
    pub attn_dec_reduce: Pipeline,
    pub attn_mma: Pipeline,
    pub attn_sum_reduce: Pipeline,
    pub mudd: Pipeline,
    pub argmax_partial: Pipeline,
    pub sample: Pipeline,
    pub accept: Pipeline,
    pub kv_compact: Pipeline,
    pub replay_draft: Pipeline,
}

impl Pipes {
    pub fn new(gpu: &Gpu, tuning: &Tuning) -> Result<Self, String> {
        let p = |name: &str| gpu.pipeline(name);
        Ok(Self {
            embed: p("embed")?,
            rmsnorm_rows: p("rmsnorm_rows")?,
            tap_rows: p("tap_rows")?,
            gemv: array(|fi| {
                let f = FORMATS[fi];
                array(|ki| array(|ri| array(|bi| p(&format!("gemv_{f}_r{}_b{}_k{}_s1", 1 << ri, 1 << bi, [1280, 3328][ki])))))
            })?,
            gemv_split: array(|fi| {
                let f = FORMATS[fi];
                array(|ki| array(|ri| array(|si| p(&format!("gemv_{f}_r{}_b1_k{}_s{}", 1 << ri, [1280, 3328][ki], 2 << si)))))
            })?,
            gemm: array(|fi| array(|ti| array(|pi| p(&format!("gemm_{}_m{}_p{}", FORMATS[fi], 16 << ti, 1 << pi)))))?,
            gemm_stream: array(|fi| {
                let f = FORMATS[fi];
                array(|ki| {
                    array(|ri| array(|ti| array(|si| p(&format!("gemms_{f}_r{}_t{}_k{}_s{}", 1 << ri, 1 << ti, [1280, 3328][ki], 4 << si)))))
                })
            })?,
            gemm_splitk_reduce: p("gemm_splitk_reduce")?,
            qkv_post: p("qkv_post")?,
            // Each variant needs only the threads of the most column groups tuned to it: some
            // chips cap a pipeline at 512.
            attn_verify: array(|k| {
                let ki = 1 << k;
                let groups = (1..=5).rfind(|g| tuning.verify_ki[g - 1] == ki).unwrap_or(1);
                gpu.pipeline_for(&format!("attn_verify_k{ki}"), 32 * groups * ki)
            })?,
            attn_dec_reduce: p("attn_dec_reduce")?,
            attn_mma: p("attn_mma")?,
            attn_sum_reduce: p("attn_sum_reduce")?,
            mudd: p("mudd")?,
            argmax_partial: p("argmax_partial")?,
            sample: gpu.pipeline_for("sample", SAMPLE_THREADS)?,
            accept: p("accept")?,
            kv_compact: p("kv_compact")?,
            replay_draft: p("replay_draft")?,
        })
    }
}

fn array<T, const N: usize>(f: impl Fn(usize) -> Result<T, String>) -> Result<[T; N], String> {
    let v = (0..N).map(f).collect::<Result<Vec<T>, String>>()?;
    Ok(v.try_into().ok().expect("N elements"))
}
