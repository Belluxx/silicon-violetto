use super::kernels::KvArgs;
use super::EngineConfig;
use crate::gpu::{self, Buffer, Gpu};
use crate::model::{Layer, Model, HEAD_DIM, N_KV_HEADS, WINDOW};

const PAGE_SHIFT: u32 = 7;
pub const PAGE: usize = 1 << PAGE_SHIFT;

pub struct KvCache {
    k_local: Buffer,
    v_local: Buffer,
    k_global: Buffer,
    v_global: Buffer,
    pub page_table: Buffer,
    free_pages: Vec<u32>,
    ring: usize,
    n_pages: usize,
    max_pages: usize,
    max_seqs: usize,
}

impl KvCache {
    pub fn new(gpu: &Gpu, cfg: &EngineConfig, model: &Model) -> Result<Self, String> {
        let n_global = model.layers.iter().filter(|l| l.global).count();
        let n_local = model.layers.len() - n_global;
        // The window plus a whole chunk: its earlier queries still need the keys its later
        // tokens overwrite.
        let ring = WINDOW + cfg.max_chunk;
        let n_pages = cfg.global_kv_tokens.div_ceil(PAGE);
        let max_pages = cfg.max_seq_len.div_ceil(PAGE);
        let row = N_KV_HEADS * HEAD_DIM * 2;
        let local = n_local * cfg.max_seqs * ring * row;
        let global = n_global * n_pages * PAGE * row;
        Ok(Self {
            k_local: gpu.buffer(local)?,
            v_local: gpu.buffer(local)?,
            k_global: gpu.buffer(global)?,
            v_global: gpu.buffer(global)?,
            page_table: gpu.buffer(cfg.max_seqs * max_pages * 4)?,
            free_pages: (0..n_pages as u32).rev().collect(),
            ring,
            n_pages,
            max_pages,
            max_seqs: cfg.max_seqs,
        })
    }

    /// Adds pages to slot `slot`, whose pages are `pages`, until they hold `len` positions.
    pub fn reserve(&mut self, slot: u32, pages: &mut Vec<u32>, len: usize) -> Result<(), String> {
        while pages.len() * PAGE < len {
            let page = self.free_pages.pop().ok_or("global KV pool exhausted")?;
            gpu::write(&self.page_table, slot as usize * self.max_pages + pages.len(), &[page]);
            pages.push(page);
        }
        Ok(())
    }

    pub fn release(&mut self, pages: &[u32]) {
        self.free_pages.extend(pages.iter().rev());
    }

    pub fn free_pages(&self) -> usize {
        self.free_pages.len()
    }

    pub fn total_pages(&self) -> usize {
        self.n_pages
    }

    pub fn buffers(&self, l: &Layer) -> (&Buffer, &Buffer) {
        if l.global {
            (&self.k_global, &self.v_global)
        } else {
            (&self.k_local, &self.v_local)
        }
    }

    pub fn args(&self, l: &Layer) -> KvArgs {
        KvArgs {
            is_global: l.global as u32,
            kv_layer: l.kv_index as u32,
            ring: self.ring as u32,
            max_seqs: self.max_seqs as u32,
            page_shift: PAGE_SHIFT,
            n_pages: self.n_pages as u32,
            max_pages: self.max_pages as u32,
            window: WINDOW as u32,
        }
    }
}
