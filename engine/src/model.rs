use std::path::Path;

use crate::gpu::{Buffer, Gpu};
use crate::weights::{DType, Weights};

pub const HIDDEN: usize = 1280;
pub const HEAD_DIM: usize = 128;
pub const N_HEADS: usize = 10;
pub const N_KV_HEADS: usize = 2;
/// Width of the fused q | k | v projection.
pub const QKV: usize = (N_HEADS + 2 * N_KV_HEADS) * HEAD_DIM;
pub const FFN: usize = 3328;
pub const N_LAYERS: usize = 48;
pub const VOCAB: usize = 151680;
/// Ids from here on are padding rows of the embedding; their logits are masked.
pub const VOCAB_REAL: usize = 151667;
/// Keys a local-layer query sees besides itself.
pub const WINDOW: usize = 1024;
pub const ATTN_SCALE: f32 = 0.1;
pub const MUDD_LAYERS: [usize; 2] = [24, 47];

/// A weight matrix, [m][k] row-major, `offset` bytes into the weight buffer.
#[derive(Clone, Copy)]
pub struct Matrix {
    pub offset: usize,
    pub m: usize,
    pub k: usize,
    /// None: bf16.
    pub int8: Option<Int8>,
}

/// An int8 matrix's scales: bf16 [m][k / group], `scales` bytes into the weight buffer.
#[derive(Clone, Copy)]
pub struct Int8 {
    pub scales: usize,
    pub group: usize,
}

impl Matrix {
    /// Bytes one pass over the matrix reads.
    pub fn bytes(&self) -> usize {
        match self.int8 {
            None => 2 * self.m * self.k,
            Some(q) => self.m * self.k + 2 * self.m * (self.k / q.group),
        }
    }
}

pub struct Layer {
    pub qkv: Matrix,
    pub o: Matrix,
    /// Rows interleaved: gate_0, up_0, gate_1, up_1, ...
    pub gate_up: Matrix,
    pub down: Matrix,
    pub attn_gate: usize,
    pub ve_gate: Option<usize>,
    pub xsa: usize,
    pub qkv_scale: f32,
    pub o_scale: f32,
    pub resid_attn: f32,
    pub post_attn: f32,
    pub resid_mlp: f32,
    pub post_mlp: f32,
    pub global: bool,
    /// Index among the global or among the local layers.
    pub kv_index: usize,
}

pub struct Model {
    pub buffer: Buffer,
    /// bf16, for the lookups.
    pub embed: Matrix,
    /// The embedding itself (tied), unless the file has a quantized head.
    pub head: Matrix,
    pub value_embed: usize,
    pub rope_freq: Vec<f32>,
    pub mudd_dense1: usize,
    /// Per MUDD site: [attention input, residual base][tap][32 weights + bias].
    pub mudd_mix: [usize; 2],
    pub layers: Vec<Layer>,
}

/// A bf16 matrix of the weight file.
fn bf16_matrix(w: &Weights, name: &str, m: usize, k: usize) -> Result<Matrix, String> {
    Ok(Matrix { offset: w.get(name, DType::Bf16, &[m, k])?.offset, m, k, int8: None })
}

/// A matrix of the weight file: bf16, or int8 with `<name>.scale` (tools/convert.py).
pub fn matrix(w: &Weights, name: &str, m: usize, k: usize) -> Result<Matrix, String> {
    if w.tensor(name)?.dtype != DType::I8 {
        return bf16_matrix(w, name, m, k);
    }
    let offset = w.get(name, DType::I8, &[m, k])?.offset;
    let scale = format!("{name}.scale");
    let groups = w.tensor(&scale)?.shape.get(1).copied().unwrap_or(0);
    let group = k / groups.max(1);
    // The kernels load 8 weights of one group at a time, and shift for the group.
    if !(8..=256).contains(&group) || !group.is_power_of_two() || group * groups != k {
        return Err(format!("{scale}: {groups} groups per row of {k}"));
    }
    let scales = w.get(&scale, DType::Bf16, &[m, groups])?.offset;
    Ok(Matrix { offset, m, k, int8: Some(Int8 { scales, group }) })
}

impl Model {
    pub fn load(gpu: &Gpu, path: &Path) -> Result<Self, String> {
        let w = Weights::load(gpu, path)?;
        let bf16 = |name: &str, m: usize, k: usize| bf16_matrix(&w, name, m, k);
        let matrix = |name: &str, m: usize, k: usize| matrix(&w, name, m, k);
        let f32s = |name: &str, shape: &[usize]| -> Result<usize, String> { Ok(w.get(name, DType::F32, shape)?.offset) };

        let mut layers = Vec::with_capacity(N_LAYERS);
        let (mut n_global, mut n_local) = (0, 0);
        for i in 0..N_LAYERS {
            let name = |k: &str| format!("L{i:02}.{k}");
            let global = i % 4 == 3;
            let count = if global { &mut n_global } else { &mut n_local };
            let kv_index = *count;
            *count += 1;
            let s = w.f32s(w.get(&name("scalars"), DType::F32, &[8])?);
            layers.push(Layer {
                qkv: matrix(&name("qkv"), QKV, HIDDEN)?,
                o: matrix(&name("o"), HIDDEN, HIDDEN)?,
                gate_up: matrix(&name("gate_up"), 2 * FFN, HIDDEN)?,
                down: matrix(&name("down"), HIDDEN, FFN)?,
                attn_gate: f32s(&name("attn_gate"), &[N_HEADS, HEAD_DIM])?,
                ve_gate: if i % 3 == 1 { Some(f32s(&name("ve_gate"), &[N_KV_HEADS, 12])?) } else { None },
                xsa: f32s(&name("xsa"), &[N_HEADS])?,
                qkv_scale: s[0],
                o_scale: s[1],
                resid_attn: s[2],
                post_attn: s[3],
                resid_mlp: s[4],
                post_mlp: s[5],
                global,
                kv_index,
            });
        }
        let embed = bf16("embed", VOCAB, HIDDEN)?;
        let head = if w.has("head") { matrix("head", VOCAB, HIDDEN)? } else { embed };
        let value_embed = bf16("value_embed", VOCAB, N_KV_HEADS * HEAD_DIM)?.offset;
        let rope_freq = w.f32s(w.get("rope_freq", DType::F32, &[32])?);
        let mudd_dense1 = f32s("mudd.dense1", &[32, HIDDEN])?;
        let mudd_mix = f32s("mudd.mix", &[2, 2, 3, 33])?;
        Ok(Self {
            buffer: w.buffer,
            embed,
            head,
            value_embed,
            rope_freq,
            mudd_dense1,
            mudd_mix: [mudd_mix, mudd_mix + 2 * 3 * 33 * 4],
            layers,
        })
    }
}
