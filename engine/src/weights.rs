use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::gpu::{self, Buffer, Gpu};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    Bf16,
    I8,
}

pub struct Tensor {
    pub dtype: DType,
    pub shape: Vec<usize>,
    /// Byte offset into the weight buffer.
    pub offset: usize,
    pub nbytes: usize,
}

/// A weight file, laid out as tools/convert.py's write() documents.
pub struct Weights {
    pub buffer: Buffer,
    tensors: HashMap<String, Tensor>,
}

fn u32_at(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
}

fn u64_at(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}

impl Weights {
    pub fn load(gpu: &Gpu, path: &Path) -> Result<Self, String> {
        let err = |e: std::io::Error| format!("{}: {e}", path.display());
        let mut f = File::open(path).map_err(err)?;
        let mut head = [0u8; 32];
        f.read_exact(&mut head).map_err(err)?;
        if &head[0..8] != b"VIOLETTO" || u32_at(&head, 8) != 1 {
            return Err(format!("{}: not a version-1 violetto weight file", path.display()));
        }
        let n = u32_at(&head, 12) as usize;
        let data_offset = u64_at(&head, 16);
        let mut records = vec![0u8; 128 * n];
        f.read_exact(&mut records).map_err(err)?;
        let mut tensors = HashMap::new();
        let mut end = 0;
        for r in records.chunks_exact(128) {
            let name_len = r[..64].iter().position(|&c| c == 0).unwrap_or(64);
            let name = String::from_utf8_lossy(&r[..name_len]).into_owned();
            let dtype = match u32_at(r, 64) {
                0 => DType::F32,
                1 => DType::Bf16,
                2 => DType::I8,
                d => return Err(format!("{name}: unknown dtype {d}")),
            };
            let ndim = u32_at(r, 68) as usize;
            let shape = (0..ndim).map(|i| u32_at(r, 72 + 4 * i) as usize).collect();
            let t = Tensor { dtype, shape, offset: u64_at(r, 88) as usize, nbytes: u64_at(r, 96) as usize };
            end = end.max(t.offset + t.nbytes);
            tensors.insert(name, t);
        }
        // One buffer for everything, read straight into its (unified) memory.
        let buffer = gpu.buffer(end)?;
        f.seek(SeekFrom::Start(data_offset)).map_err(err)?;
        // Safety: the buffer holds `end` bytes and nothing else uses it yet.
        let dst = unsafe { std::slice::from_raw_parts_mut(gpu::contents::<u8>(&buffer), end) };
        f.read_exact(dst).map_err(err)?;
        Ok(Self { buffer, tensors })
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn tensor(&self, name: &str) -> Result<&Tensor, String> {
        self.tensors.get(name).ok_or_else(|| format!("weight file has no tensor {name}"))
    }

    pub fn get(&self, name: &str, dtype: DType, shape: &[usize]) -> Result<&Tensor, String> {
        let t = self.tensor(name)?;
        if t.dtype != dtype || t.shape != shape {
            return Err(format!("{name}: expected {dtype:?} {shape:?}, found {:?} {:?}", t.dtype, t.shape));
        }
        Ok(t)
    }

    pub fn f32s(&self, t: &Tensor) -> Vec<f32> {
        assert_eq!(t.dtype, DType::F32);
        gpu::read::<f32>(&self.buffer, t.offset / 4, t.nbytes / 4)
    }
}
