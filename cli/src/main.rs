mod bench;
mod bench_spec;
mod check;
mod features;
mod files;
mod generate;
mod profile;
mod safetensors;
mod serve;
mod text;

use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;

use clap::{Parser, Subcommand};
use violetto::{Draft, Engine, EngineConfig, Sampling, Tuning};

/// Errors are messages for the terminal.
pub type Res<T = ()> = Result<T, String>;

#[derive(Parser)]
#[command(name = "violetto", version, about = "Limite 1B Violetto on Apple Silicon", arg_required_else_help = true)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Serve an OpenAI-compatible API
    Serve(serve::Args),
    /// Run a file of requests, continuously batched
    ///
    /// A request line is `name opts id id id ...`, the prompt as token ids. opts is `-`, or
    /// comma-separated key=value pairs that override the flags for that request: seed (default:
    /// the line's index), max_tokens, temperature and top_p. The output has one JSON line per
    /// request, in the order they finish: name, finish ("stop" or "length"), prompt_tokens and
    /// tokens.
    Generate(generate::Args),
    /// Measure single-stream decode, prefill and batched decode
    ///
    /// Random prompts, tokens fed back on the GPU. An empty list skips its table.
    Bench(bench::Args),
    /// Measure speculative decoding by rows per step
    ///
    /// With --drafter, the engine's drafter; without, the replay drafter, which drafts a first
    /// decode's tokens with a share of them wrong. Speeds are tokens over the GPU time of the
    /// steps that ran every sequence.
    BenchSpec(bench_spec::Args),
    /// Time each kernel group of a step
    ///
    /// GPU time of a decode step (or with --chunk, of a prefill step) with each kernel group
    /// left out, and with that group alone.
    Profile(profile::Args),
    /// Time every matmul kernel variant
    ///
    /// Every matmul kind, kernel variant and token count, over all layers' matrices back to
    /// back.
    Matmul(profile::MatmulArgs),
    /// Dump one block of the drafter, for tools/spec/check_engine.py
    DrafterCheck(check::DrafterArgs),
    /// Dump the drafter's training features of generated traces
    ///
    /// Each trace of a `generate` run is teacher-forced, and DIR/<name>.safetensors holds its
    /// tokens, prompt_len, taps and feat: bf16(rms_norm(output)) of the tapped layers at every
    /// position.
    Features(features::Args),
}

/// A comma-separated list, empty for none.
#[derive(Clone, Debug)]
pub struct List<T>(pub Vec<T>);

impl<T: FromStr<Err: Display>> FromStr for List<T> {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        let items = s.split(',').map(str::trim).filter(|x| !x.is_empty());
        items.map(|x| x.parse().map_err(|e| format!("{x}: {e}"))).collect::<Result<_, _>>().map(List)
    }
}

impl<T: Copy + Ord> List<T> {
    pub fn max(&self) -> Option<T> {
        self.0.iter().copied().max()
    }
}

/// The weights, and the tuning overrides.
#[derive(clap::Args)]
pub struct EngineArgs {
    /// Weight file, from tools/convert.py
    #[arg(long, default_value = "models/limite-bf16.vlt")]
    pub weights: PathBuf,
    /// GPU cores to plan threadgroups for [default: the chip's]
    #[arg(long, help_heading = "Tuning")]
    gpu_cores: Option<usize>,
    /// Threadgroups a GEMM splits K to reach, 0 for never [default: 8 per GPU core]
    #[arg(long, help_heading = "Tuning")]
    splitk: Option<usize>,
    /// attn_verify's keys per split on local and global layers, 0 for the default
    #[arg(long, help_heading = "Tuning", value_name = "LOCAL,GLOBAL")]
    dec_split: Option<List<usize>>,
    /// attn_verify's simdgroups per column group (1, 2 or 4), for 1 to 5 column groups
    #[arg(long, help_heading = "Tuning", default_value = "4,4,4,2,2")]
    verify_ki: List<usize>,
    /// Rows policy: verify tokens a step holds at long context, 0 for no policy [default: 32; bench-spec: 0]
    #[arg(long, help_heading = "Tuning")]
    pub spec_tokens: Option<usize>,
    /// Rows policy: verify tokens a step holds, shared by its decoding requests
    #[arg(long, help_heading = "Tuning", default_value_t = Tuning::default().spec_tokens_short)]
    spec_tokens_short: usize,
    /// Rows policy: tokens of context across the decoding requests where long context starts
    #[arg(long, help_heading = "Tuning", default_value_t = Tuning::default().spec_kv)]
    spec_kv: usize,
    /// Rows policy: rows a request verifies at long context, at most
    #[arg(long, help_heading = "Tuning", default_value_t = Tuning::default().spec_rows_long)]
    spec_rows_long: usize,
}

impl EngineArgs {
    pub fn tuning(&self) -> Res<Tuning> {
        let mut t = Tuning { gpu_cores: self.gpu_cores, splitk_target: self.splitk, ..Tuning::default() };
        if let Some(List(d)) = &self.dec_split {
            let &[local, global] = &d[..] else { return Err("--dec-split takes LOCAL,GLOBAL".into()) };
            t.decode_split = [Some(local).filter(|&x| x > 0), Some(global).filter(|&x| x > 0)];
        }
        if self.verify_ki.0.len() > 5 || self.verify_ki.0.iter().any(|k| ![1, 2, 4].contains(k)) {
            return Err("--verify-ki takes up to 5 values of 1, 2 or 4".into());
        }
        t.verify_ki[..self.verify_ki.0.len()].copy_from_slice(&self.verify_ki.0);
        t.spec_tokens = self.spec_tokens.unwrap_or(t.spec_tokens);
        (t.spec_tokens_short, t.spec_kv, t.spec_rows_long) = (self.spec_tokens_short, self.spec_kv, self.spec_rows_long);
        Ok(t)
    }

    pub fn load(&self, cfg: EngineConfig) -> Res<Engine> {
        self.load_with(cfg, self.tuning()?)
    }

    pub fn load_with(&self, cfg: EngineConfig, tuning: Tuning) -> Res<Engine> {
        let t0 = Instant::now();
        let e = Engine::new(&self.weights, cfg, tuning)?;
        let cores = match e.gpu_cores_detected() {
            Some(n) if n == e.gpu_cores() => format!("{n} GPU cores"),
            Some(n) => format!("planned for {} GPU cores, {n} detected", e.gpu_cores()),
            None => format!("GPU cores not found, planned for {}", e.gpu_cores()),
        };
        eprintln!("loaded {} on {} ({cores}) in {:.2} s", self.weights.display(), e.device_name(), t0.elapsed().as_secs_f64());
        Ok(e)
    }

    /// The weight file's name without its extension.
    pub fn weights_name(&self) -> String {
        self.weights.file_stem().unwrap_or_default().to_string_lossy().into_owned()
    }
}

pub fn load_drafter(eng: &mut Engine, path: &Path) -> Res {
    eng.load_drafter(path).map_err(|e| format!("{}: {e}", path.display()))
}

/// The most rows per step a request asks for by default: the rows policy takes it from there.
pub const SPEC_ROWS: usize = 8;

/// Speculative decoding, for the commands that generate.
#[derive(clap::Args)]
pub struct DraftArgs {
    /// Drafter for speculative decoding, from tools/spec/export.py
    #[arg(long)]
    pub drafter: Option<PathBuf>,
    /// The most rows a request verifies per step
    #[arg(long, default_value_t = SPEC_ROWS)]
    pub spec_rows: usize,
}

impl DraftArgs {
    /// Every request's drafter: the engine's if --drafter was given.
    pub fn draft(&self) -> Draft {
        if self.drafter.is_some() { Draft::Model { rows: self.spec_rows } } else { Draft::None }
    }

    pub fn load(&self, eng: &mut Engine) -> Res {
        self.drafter.as_deref().map_or(Ok(()), |d| load_drafter(eng, d))
    }
}

#[derive(clap::Args)]
pub struct SamplingArgs {
    /// Sampling temperature, 0 for greedy [default: 0.6; bench and bench-spec: 0]
    #[arg(long)]
    temperature: Option<f32>,
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,
}

impl SamplingArgs {
    pub fn with(&self, temperature: f32, seed: u64) -> Sampling {
        Sampling { temperature: self.temperature.unwrap_or(temperature), top_p: self.top_p, seed }
    }
}

fn main() {
    let done = match Cli::parse().command {
        Command::Serve(a) => serve::serve(&a),
        Command::Generate(a) => generate::generate(&a),
        Command::Bench(a) => bench::bench(&a),
        Command::BenchSpec(a) => bench_spec::bench_spec(&a),
        Command::Profile(a) => profile::profile(&a),
        Command::Matmul(a) => profile::matmul(&a),
        Command::DrafterCheck(a) => check::drafter_check(&a),
        Command::Features(a) => features::features(&a),
    };
    if let Err(e) = done {
        eprintln!("violetto: {e}");
        std::process::exit(1);
    }
}
