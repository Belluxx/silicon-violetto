use super::encode::Epilogue;
use super::kernels::SampleRow;
use super::{Engine, Matmul, Sampling, SeqId};
use crate::gpu;
use crate::model::{N_LAYERS, VOCAB};

/// Kernel groups `Engine::set_profile_skip` can leave out, for timing: the results are
/// then wrong.
pub mod skip {
    pub const QKV: u32 = 1;
    pub const POST: u32 = 2;
    pub const ATTN: u32 = 4;
    pub const REDUCE: u32 = 8;
    pub const O: u32 = 16;
    pub const GATE_UP: u32 = 32;
    pub const DOWN: u32 = 64;
    pub const MUDD: u32 = 128;
    pub const HEAD: u32 = 256;
}

impl Engine {
    /// Leaves kernel groups out of every step (see `skip`).
    #[doc(hidden)]
    pub fn set_profile_skip(&mut self, mask: u32) {
        self.skip = mask;
    }

    /// Advances a sequence by `len` positions without computing them: decode timing
    /// doesn't depend on the KV values.
    #[doc(hidden)]
    pub fn bench_fill(&mut self, id: SeqId, len: usize) -> Result<(), String> {
        let seq = self.seqs[id.0 as usize].as_mut().ok_or("unknown sequence")?;
        seq.len += len;
        gpu::write(&self.s.lens, id.0 as usize, &[seq.len as u32]);
        self.kv.reserve(id.0, &mut seq.pages, seq.len)
    }

    /// The tokens the sampler picks from rows of logits (rows x VOCAB; None: the previous
    /// call's), each row with its own settings and drawn token position.
    #[doc(hidden)]
    pub fn sample_logits(&mut self, logits: Option<&[f32]>, rows: &[(Sampling, usize)]) -> Result<Vec<u32>, String> {
        let n = rows.len();
        if n == 0 || n > self.cfg.max_head_rows || logits.is_some_and(|l| l.len() != n * VOCAB) {
            return Err(format!("{n} rows of logits (max {} rows)", self.cfg.max_head_rows));
        }
        rows.iter().try_for_each(|r| r.0.check())?;
        let samples: Vec<SampleRow> = rows.iter().map(|&(s, pos)| s.row_at(pos)).collect();
        if let Some(l) = logits {
            gpu::write(&self.s.logits, 0, l);
        }
        gpu::write(&self.sets[0].sample_rows, 0, &samples);
        self.gpu.run(|e| self.encode_sample(e, 0, n))?;
        Ok(gpu::read(&self.sets[0].out_ids, 0, n))
    }

    /// GPU seconds (best of `reps`) of one matmul kind (qkv, o, gate_up, down or head) over
    /// all layers' matrices, for `n` tokens with `variant`, and the weight bytes that reads.
    /// `same` reuses layer 0's matrix, so it comes from the cache.
    #[doc(hidden)]
    pub fn bench_matmul(
        &mut self,
        kind: &str,
        n: usize,
        variant: Matmul,
        reps: usize,
        same: bool,
    ) -> Result<(f64, usize), String> {
        if !["qkv", "o", "gate_up", "down", "head"].contains(&kind) {
            return Err(format!("unknown matmul {kind}"));
        }
        let gemv_ok = |rows: usize, splits: usize| n <= 8 && (splits == 1 || n == 1) && rows * 8 / splits > 0;
        let ok = match variant {
            Matmul::Gemv { rows, splits } => gemv_ok(rows, splits),
            Matmul::Stream { tiles, .. } => n <= 8 * tiles,
            Matmul::Gemm { .. } => true,
        };
        if !ok || (kind == "head" && n > self.cfg.max_head_rows) {
            return Err(format!("{kind} with {variant:?} can't take {n} tokens"));
        }
        self.force_matmul = Some(variant);
        let s = &self.s;
        let best = (0..reps).try_fold(f64::MAX, |best, _| {
            let t = self.gpu.run(|e| {
                if kind == "head" {
                    self.matmul(e, &self.model.head, n, &s.hfin_head, &s.logits, Epilogue::Head);
                    return;
                }
                for li in 0..N_LAYERS {
                    let l = &self.model.layers[if same { 0 } else { li }];
                    let resid = |base| Epilogue::Resid { scale: 1.0, ra: 1.0, rb: 1.0, base, raw: None };
                    match kind {
                        "qkv" => self.matmul(e, &l.qkv, n, &s.hmain, &s.qkv, Epilogue::Norm { scale: 1.0, inv: &s.inv }),
                        "o" => self.matmul(e, &l.o, n, &s.attn, &s.mixed, resid(&s.hmain)),
                        "gate_up" => self.matmul(e, &l.gate_up, n, &s.mixed, &s.act, Epilogue::SwiGlu),
                        _ => self.matmul(e, &l.down, n, &s.act, &s.hmain, resid(&s.mixed)),
                    }
                }
            })?;
            Ok::<f64, String>(best.min(t))
        });
        self.force_matmul = None;
        let layers = &self.model.layers;
        let bytes = match kind {
            "head" => self.model.head.bytes(),
            "qkv" => layers.iter().map(|l| l.qkv.bytes()).sum(),
            "o" => layers.iter().map(|l| l.o.bytes()).sum(),
            "gate_up" => layers.iter().map(|l| l.gate_up.bytes()).sum(),
            _ => layers.iter().map(|l| l.down.bytes()).sum(),
        };
        Ok((best?, bytes))
    }
}
