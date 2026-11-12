use std::collections::VecDeque;
use std::time::Instant;

use super::drafter::DraftRun;
use super::kernels::{GenState, ReplayArgs, FEED_ROW, FEED_W, MAX_STOP};
use super::kv::PAGE;
use super::plan::PlanRun;
use super::{Engine, Rows, Sampling, SeqId};
use crate::gpu::{self, CommandBuffer};
use crate::model::{VOCAB, VOCAB_REAL};

pub struct Request {
    pub prompt: Prompt,
    /// Tokens to generate at most.
    pub max_tokens: usize,
    /// Tokens that end the generation (at most 8); the stop token is its last output.
    pub stop: Vec<u32>,
    pub sampling: Sampling,
    pub draft: Draft,
}

/// Where a request's drafts come from.
#[derive(Clone, Debug, PartialEq)]
pub enum Draft {
    /// None: one token per step.
    None,
    /// A test drafter: it drafts a recorded sequence (prompt and continuation), wrong at
    /// the generator's replay mistake rate.
    Replay { rows: usize, trace: Vec<u32> },
    /// The engine's drafter (`Engine::load_drafter`), verifying at most `rows` (up to 8)
    /// tokens a step.
    Model { rows: usize },
}

impl Draft {
    fn rows(&self) -> usize {
        match self {
            Draft::None => 1,
            Draft::Replay { rows, .. } | Draft::Model { rows } => *rows,
        }
    }
}

/// GPU-fed input rows: the slot's next input, then its drafts.
const FEED_ROWS: [u32; FEED_W] = {
    let mut r = [0; FEED_W];
    let mut i = 0;
    while i < FEED_W {
        r[i] = FEED_ROW + i as u32;
        i += 1;
    }
    r
};

pub enum Prompt {
    /// Prefilled into a new sequence.
    Tokens(Vec<u32>),
    /// A sequence already in the engine, continued by feeding it this token. The generator
    /// takes it over, and frees it at the end.
    Resume(SeqId, u32),
}

/// Requests are numbered from 0 in the order they were submitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReqId(pub usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// Its last token was a stop token.
    Stop,
    /// It generated max_tokens.
    Length,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    Token(ReqId, u32),
    Done(ReqId, Finish),
}

#[derive(Clone, Copy, Debug)]
pub struct StepStats {
    pub tokens: usize,
    /// Every run had at most 8 tokens (the small-run attention ran).
    pub decode: bool,
    /// Tokens the step emitted, for requests still running.
    pub emitted: usize,
    pub gpu_seconds: f64,
    /// CPU time spent encoding it.
    pub encode_seconds: f64,
}

/// Continuous batching, pipelined: the CPU encodes step t+1 while step t runs, and collects
/// each step's results (and so the lengths its drafts reach) one step behind.
pub struct Generator<'e> {
    eng: &'e mut Engine,
    queue: VecDeque<Queued>,
    active: Vec<Active>,
    in_flight: Option<InFlight>,
    submitted: usize,
    /// Steps encoded, and the last one collected.
    steps: usize,
    collected: Option<usize>,
    /// The replay drafter's mistakes: the share of positions, and the hash's seed.
    replay: (f32, u32),
    pub stats: Vec<StepStats>,
    /// While set, requests whose prompt is in wait instead of decoding, so that a benchmark
    /// can start them all at the same context.
    pub hold: bool,
}

struct Queued {
    id: ReqId,
    req: Request,
    /// The sequence's length once its last token is fed (the final token never is).
    kv_len: usize,
}

struct Active {
    id: ReqId,
    seq: SeqId,
    /// Tokens the CPU feeds: the prompt, or the resumed token. `fed` of them are planned.
    feed: Vec<u32>,
    fed: usize,
    max_tokens: usize,
    stop: Vec<u32>,
    kv_len: usize,
    /// Rows per decoding step (1: no drafter), and who drafts them.
    rows: usize,
    model: bool,
    replay: bool,
    /// Tokens received, and the most the step in flight can still bring.
    received: usize,
    inflight: usize,
    done: bool,
    /// The last step that uses the sequence. It's freed once that step is collected.
    last_step: Option<usize>,
}

/// A request's run in the step being planned: its tokens from the CPU (None: `rows`
/// GPU-fed rows), and whether it draws a token.
struct Pick {
    active: usize,
    feed: Option<(usize, usize)>,
    rows: usize,
    draws: bool,
}

/// A drawing run of a step in flight.
struct Drawn {
    id: ReqId,
    seq: SeqId,
    run: usize,
    count: usize,
    head0: usize,
    /// Verify rows (0: one draw, from a prompt chunk or a one-token step).
    verify: usize,
}

struct InFlight {
    cb: CommandBuffer,
    set: usize,
    step: usize,
    drawn: Vec<Drawn>,
    stats: StepStats,
}

impl Engine {
    /// A generator over this engine's free sequence slots and KV pages.
    pub fn generator(&mut self) -> Result<Generator<'_>, String> {
        let c = &self.cfg;
        if c.max_step_tokens < c.max_seqs || c.max_head_rows < c.max_seqs {
            return Err(format!(
                "generation needs max_step_tokens ({}) and max_head_rows ({}) of at least max_seqs ({})",
                c.max_step_tokens, c.max_head_rows, c.max_seqs
            ));
        }
        Ok(Generator {
            eng: self,
            queue: VecDeque::new(),
            active: Vec::new(),
            in_flight: None,
            submitted: 0,
            steps: 0,
            collected: None,
            replay: (0.0, 0),
            stats: Vec::new(),
            hold: false,
        })
    }
}

impl Generator<'_> {
    /// The replay drafter's mistakes: a share `rate` of the positions, picked by a hash of
    /// (seed, position).
    pub fn set_replay_mistakes(&mut self, rate: f32, seed: u32) {
        self.replay = (rate, seed);
    }

    /// Queues a request; it starts once a slot and its KV pages are free.
    pub fn submit(&mut self, req: Request) -> Result<ReqId, String> {
        let eng = &*self.eng;
        req.sampling.check()?;
        if req.stop.len() > MAX_STOP {
            return Err(format!("{} stop tokens (max {MAX_STOP})", req.stop.len()));
        }
        let rows = req.draft.rows();
        if rows == 0 || rows > FEED_W.min(eng.cfg.max_chunk).min(eng.cfg.max_head_rows) {
            return Err(format!("{rows} rows per step (max {FEED_W}, max_chunk and max_head_rows)"));
        }
        if let Draft::Replay { trace, .. } = &req.draft {
            if trace.len() > eng.cfg.max_seq_len || trace.iter().any(|&t| t as usize >= VOCAB) {
                return Err("replay trace longer than max_seq_len, or with ids out of range".into());
            }
        }
        if let Draft::Model { rows } = req.draft {
            if !eng.has_drafter() || rows > super::drafter::MAX_ROWS {
                return Err(format!("Draft::Model needs a loaded drafter and at most {} rows", super::drafter::MAX_ROWS));
            }
        }
        let len = match req.prompt {
            Prompt::Tokens(ref p) if p.is_empty() => return Err("empty prompt".into()),
            Prompt::Tokens(ref p) if p.iter().any(|&t| t as usize >= VOCAB) => return Err("token id out of range".into()),
            Prompt::Tokens(ref p) => p.len(),
            Prompt::Resume(_, t) if t as usize >= VOCAB => return Err("token id out of range".into()),
            Prompt::Resume(seq, _) => {
                let queued = self.queue.iter().any(|q| matches!(q.req.prompt, Prompt::Resume(s, _) if s == seq));
                if queued || self.active.iter().any(|a| a.seq == seq) {
                    return Err(format!("{seq:?} is already generating"));
                }
                eng.seqs.get(seq.0 as usize).and_then(Option::as_ref).ok_or("unknown sequence")?.len + 1
            }
        };
        if req.max_tokens == 0 {
            return Err("max_tokens is 0".into());
        }
        let kv_len = len + req.max_tokens - 1;
        if kv_len > eng.cfg.max_seq_len || kv_len.div_ceil(PAGE) > eng.kv.total_pages() {
            return Err(format!("a request reaching {kv_len} positions doesn't fit max_seq_len or the KV pool"));
        }
        let id = ReqId(self.submitted);
        self.submitted += 1;
        self.queue.push_back(Queued { id, req, kv_len });
        Ok(id)
    }

    /// Nothing queued, running or in flight.
    pub fn is_idle(&self) -> bool {
        self.queue.is_empty() && self.active.is_empty() && self.in_flight.is_none()
    }

    /// Requests waiting for a slot, and running.
    pub fn counts(&self) -> (usize, usize) {
        (self.queue.len(), self.active.len())
    }

    /// Drops a request. Queued, it never starts; running, it gets no more events, and its
    /// sequence is freed once no step in flight uses it.
    pub fn cancel(&mut self, id: ReqId) {
        if let Some(i) = self.queue.iter().position(|q| q.id == id) {
            if let Some(Queued { req: Request { prompt: Prompt::Resume(seq, _), .. }, .. }) = self.queue.remove(i) {
                self.eng.free_seq(seq);
            }
        } else if let Some(a) = self.active.iter_mut().find(|a| a.id == id) {
            a.done = true;
        }
    }

    /// Starts what it can of the queue, commits the next step, and returns what the previous
    /// step produced.
    pub fn step(&mut self) -> Result<Vec<Event>, String> {
        self.admit()?;
        let next = self.encode_next()?;
        let mut events = Vec::new();
        if let Some(prev) = std::mem::replace(&mut self.in_flight, next) {
            self.collect(prev, &mut events)?;
        }
        self.release();
        Ok(events)
    }

    /// Starts queued requests, in order, while a slot is free and the KV pool can hold
    /// them at their longest beside the running ones.
    fn admit(&mut self) -> Result<(), String> {
        while let Some(q) = self.queue.front() {
            let eng = &*self.eng;
            let held = |seq: SeqId| eng.seqs[seq.0 as usize].as_ref().map_or(0, |s| s.pages.len());
            let growth: usize = self.active.iter().map(|a| a.kv_len.div_ceil(PAGE).saturating_sub(held(a.seq))).sum();
            let (slot_free, need) = match q.req.prompt {
                Prompt::Tokens(_) => (eng.seqs.iter().any(Option::is_none), q.kv_len.div_ceil(PAGE)),
                Prompt::Resume(seq, _) => (true, q.kv_len.div_ceil(PAGE).saturating_sub(held(seq))),
            };
            if !slot_free || eng.kv.free_pages() < growth + need {
                if self.active.is_empty() && self.in_flight.is_none() {
                    return Err(format!("{:?} can't start: no free slot or KV pages, and nothing running", q.id));
                }
                return Ok(());
            }
            let Queued { id, req, kv_len } = self.queue.pop_front().unwrap();
            let (seq, feed) = match req.prompt {
                Prompt::Tokens(p) => (self.eng.new_seq()?, p),
                Prompt::Resume(seq, t) => (seq, vec![t]),
            };
            self.eng.set_sampling(seq, req.sampling)?;
            // No step in flight uses the slot (a new one, or a resumed idle sequence).
            let slot = seq.0 as usize;
            let mut gs = GenState { remaining: req.max_tokens as u32, nstop: req.stop.len() as u32, ..GenState::OPEN };
            gs.stop[..req.stop.len()].copy_from_slice(&req.stop);
            self.eng.set_gen(slot, gs);
            let rows = req.draft.rows();
            let model = matches!(req.draft, Draft::Model { .. });
            let replay = matches!(req.draft, Draft::Replay { .. });
            let trace = match req.draft {
                Draft::Replay { trace, .. } => trace,
                _ => Vec::new(),
            };
            gpu::write(&self.eng.s.trace, slot * self.eng.cfg.max_seq_len, &trace);
            gpu::write(&self.eng.s.trace_len, slot, &[trace.len() as u32]);
            self.active.push(Active {
                id,
                seq,
                feed,
                fed: 0,
                max_tokens: req.max_tokens,
                stop: req.stop,
                kv_len,
                rows,
                model,
                replay,
                received: 0,
                inflight: 0,
                done: false,
                last_step: None,
            });
        }
        Ok(())
    }

    /// Plans, encodes and commits a step, if any request has a token to add: a run for
    /// every decoding request, then prompt chunks in the order the requests started.
    fn encode_next(&mut self) -> Result<Option<InFlight>, String> {
        let (step, cfg) = (self.steps, &self.eng.cfg);
        let mut budget = cfg.max_step_tokens;
        let hold = self.hold;
        let decoding = |a: &Active| !hold && !a.done && a.fed == a.feed.len() && a.received + a.inflight < a.max_tokens;
        // Rows per decoding request: what fits the step, then the rows policy.
        let n_dec = self.active.iter().filter(|a| decoding(a)).count();
        let row_cap = (cfg.max_step_tokens.min(cfg.max_head_rows) / n_dec.max(1)).max(1);
        let t = &self.eng.tuning;
        let kv: usize = self.active.iter().filter(|a| decoding(a)).map(|a| self.eng.seq_len(a.seq)).sum();
        let short = (t.spec_tokens_short / n_dec.max(1)).max(2);
        let spec_cap = match t.spec_tokens {
            0 => usize::MAX,
            _ if kv < t.spec_kv => short,
            tokens => short.max((tokens / n_dec.max(1)).clamp(2, t.spec_rows_long.max(2))),
        };
        let mut picks = Vec::new();
        for (i, a) in self.active.iter_mut().enumerate() {
            if decoding(a) {
                // Every row may emit a token: stay within max_tokens.
                let rows = a.rows.min(row_cap).min(spec_cap).min(a.max_tokens - a.received - a.inflight);
                picks.push(Pick { active: i, feed: None, rows, draws: true });
                a.inflight += rows;
                a.last_step = Some(step);
                budget -= rows;
            }
        }
        for (i, a) in self.active.iter_mut().enumerate() {
            if budget > 0 && a.fed < a.feed.len() {
                let k = (a.feed.len() - a.fed).min(cfg.max_chunk).min(budget);
                let draws = a.fed + k == a.feed.len();
                picks.push(Pick { active: i, feed: Some((a.fed, a.fed + k)), rows: k, draws });
                a.fed += k;
                a.inflight += draws as usize;
                a.last_step = Some(step);
                budget -= k;
            }
        }
        if picks.is_empty() {
            return Ok(None);
        }
        let runs: Vec<PlanRun> = picks
            .iter()
            .map(|p| {
                let a = &self.active[p.active];
                let tokens = match p.feed {
                    Some((s, e)) => &a.feed[s..e],
                    None => &FEED_ROWS[..p.rows],
                };
                let head = if p.draws { Rows::Last } else { Rows::None };
                PlanRun { seq: a.seq, tokens, head, verify: p.feed.is_none() && p.rows > 1 }
            })
            .collect();
        let mut drawn = Vec::new();
        let mut head0 = 0;
        for (run, (p, r)) in picks.iter().zip(&runs).enumerate() {
            if p.draws {
                let a = &self.active[p.active];
                let verify = if r.verify { p.rows } else { 0 };
                drawn.push(Drawn { id: a.id, seq: a.seq, run, count: r.tokens.len(), head0, verify });
                head0 += verify.max(1);
            }
        }
        let replay = self.active.iter().any(|a| a.rows > 1 && a.replay);
        let modeled = self.active.iter().any(|a| a.model);
        let draft_runs: Vec<DraftRun> = drawn
            .iter()
            .filter_map(|d| {
                let a = self.active.iter().find(|a| a.id == d.id)?;
                a.model.then(|| DraftRun { slot: d.seq.0, rows: a.rows.min(spec_cap) })
            })
            .collect();
        let set = step % 3;
        let eng = &mut *self.eng;
        let plan = eng.plan(set, &runs, Rows::None)?;
        let max_len = draft_runs.iter().map(|r| eng.seq_len(SeqId(r.slot)) + r.rows).max().unwrap_or(0);
        let t0 = Instant::now();
        let cb = eng.gpu.command_buffer();
        let e = cb.encoder();
        eng.encode_forward(&e, set, &plan);
        if modeled {
            eng.encode_drafter(&e, set, &plan, &draft_runs, max_len);
        }
        if replay {
            let args = ReplayArgs {
                stride: eng.cfg.max_seq_len as u32,
                seed: self.replay.1,
                rate: self.replay.0,
                vocab: VOCAB_REAL as u32,
            };
            eng.encode_replay(&e, set, &plan, &args);
        }
        e.end();
        let stats = StepStats {
            tokens: plan.n,
            decode: plan.fused,
            emitted: 0,
            gpu_seconds: 0.0,
            encode_seconds: t0.elapsed().as_secs_f64(),
        };
        cb.commit();
        self.steps += 1;
        Ok(Some(InFlight { cb, set, step, drawn, stats }))
    }

    fn collect(&mut self, f: InFlight, events: &mut Vec<Event>) -> Result<(), String> {
        let gpu_seconds = f.cb.wait()?;
        let before = events.len();
        self.collected = Some(f.step);
        let st = &self.eng.sets[f.set];
        let n_head = f.drawn.last().map_or(0, |d| d.head0 + d.verify.max(1));
        let ids = gpu::read::<u32>(&st.out_ids, 0, n_head);
        let n_runs = f.drawn.last().map_or(0, |d| d.run + 1);
        let emitted = gpu::read::<u32>(&st.emitted, 0, n_runs);
        for d in &f.drawn {
            let e = emitted[d.run] as usize;
            // The plan advanced the sequence by its whole run: a verify run advances by what
            // it emits, and a slot already done on the GPU not at all.
            if d.verify > 0 {
                self.eng.settle(d.seq, d.count - e);
            } else if e == 0 {
                self.eng.settle(d.seq, d.count);
            }
            let a = self.active.iter_mut().find(|a| a.id == d.id).expect("a drawing request is active");
            a.inflight -= d.verify.max(1);
            if a.done {
                continue;
            }
            for &tok in &ids[d.head0..d.head0 + e] {
                a.received += 1;
                events.push(Event::Token(d.id, tok));
                let finish = if a.stop.contains(&tok) {
                    Finish::Stop
                } else if a.received == a.max_tokens {
                    Finish::Length
                } else {
                    continue;
                };
                a.done = true;
                events.push(Event::Done(d.id, finish));
                break;
            }
        }
        let emitted = events[before..].iter().filter(|e| matches!(e, Event::Token(..))).count();
        self.stats.push(StepStats { gpu_seconds, emitted, ..f.stats });
        Ok(())
    }

    /// Frees the sequences of finished requests that no step in flight uses.
    fn release(&mut self) {
        let collected = self.collected;
        let eng = &mut *self.eng;
        self.active.retain(|a| {
            let free = a.done && a.last_step.is_none_or(|s| collected.is_some_and(|c| s <= c));
            if free {
                eng.free_seq(a.seq);
            }
            !free
        });
    }
}

impl Drop for Generator<'_> {
    /// Waits for the step in flight, and frees every sequence the generator holds.
    fn drop(&mut self) {
        if let Some(f) = self.in_flight.take() {
            let _ = f.cb.wait();
        }
        for a in self.active.drain(..) {
            self.eng.free_seq(a.seq);
        }
        for q in self.queue.drain(..) {
            if let Prompt::Resume(seq, _) = q.req.prompt {
                self.eng.free_seq(seq);
            }
        }
    }
}
