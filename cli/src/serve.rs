use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use tokenizers::Tokenizer;
use violetto::{Draft, EngineConfig, Event, Finish, Prompt, ReqId, Request, Sampling, VOCAB_REAL};

use crate::text::{self, Message, Output, ENDOFTEXT, IM_END};
use crate::{DraftArgs, EngineArgs, Res};

#[derive(clap::Args)]
pub struct Args {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8000)]
    port: u16,
    /// Requests decoding at once
    #[arg(long, default_value_t = 1)]
    max_seqs: usize,
    /// Positions a request can reach, its prompt and its reply together
    #[arg(long, default_value_t = 131072)]
    max_seq_len: usize,
    /// KV pool of the global layers, in tokens of 12 KiB [default: every slot at --max-seq-len]
    #[arg(long)]
    kv_tokens: Option<usize>,
    /// Prompt tokens a sequence adds per step
    #[arg(long, default_value_t = 512)]
    chunk: usize,
    #[arg(long, default_value = "models/limite-1b-violetto/tokenizer.json")]
    tokenizer: PathBuf,
    /// The model's id in replies and /v1/models
    #[arg(long, default_value = "limite-1b-violetto")]
    model_name: String,
    #[command(flatten)]
    draft: DraftArgs,
    #[command(flatten)]
    engine: EngineArgs,
}

const MAX_BODY: usize = 64 << 20;

/// One choice of a request, for the engine thread.
struct Job {
    req: Request,
    reply: Reply,
}

/// Where a choice's tokens go, and the flag that cancels it.
struct Reply {
    choice: usize,
    tx: Sender<(usize, Out)>,
    cancel: Arc<AtomicBool>,
}

enum Out {
    Token(u32),
    Done(Finish),
    Error(String),
}

struct Ctx {
    tok: Tokenizer,
    jobs: Sender<Job>,
    /// Every request's drafter.
    draft: Draft,
    model: String,
    max_seq_len: usize,
    started: u64,
    requests: AtomicU64,
}

struct HttpError(u16, String);

fn bad(msg: impl Into<String>) -> HttpError {
    HttpError(400, msg.into())
}

pub fn serve(a: &Args) -> Res {
    let tok = Tokenizer::from_file(&a.tokenizer).map_err(|e| format!("{}: {e}", a.tokenizer.display()))?;
    let (max_seqs, max_seq_len) = (a.max_seqs, a.max_seq_len);
    let cfg = EngineConfig {
        max_seqs,
        max_step_tokens: a.chunk.max(max_seqs),
        max_chunk: a.chunk,
        global_kv_tokens: a.kv_tokens.unwrap_or(max_seqs * max_seq_len),
        max_seq_len,
        max_head_rows: max_seqs * if a.draft.drafter.is_some() { a.draft.spec_rows } else { 1 },
    };
    let addr = format!("{}:{}", a.host, a.port);
    let (jobs, jobs_rx) = mpsc::channel();
    let (ready, ready_rx) = mpsc::channel();
    let ctx = Ctx { tok, jobs, draft: a.draft.draft(), model: a.model_name.clone(), max_seq_len, started: now(), requests: AtomicU64::new(0) };
    std::thread::scope(|sc| {
        // The engine's Metal objects stay on the thread that made them.
        let engine = sc.spawn(|| run_engine(a, cfg, jobs_rx, ready));
        if ready_rx.recv().is_err() {
            return engine.join().expect("the engine thread");
        }
        let listener = TcpListener::bind(&addr).map_err(|e| format!("{addr}: {e}"))?;
        eprintln!("serving {} at http://{addr}/v1 ({max_seqs} slots of {max_seq_len} positions)", ctx.model);
        for conn in listener.incoming().flatten() {
            let ctx = &ctx;
            sc.spawn(move || handle(conn, ctx));
        }
        Ok(())
    })
}

/// The engine thread: until it sends `ready`, its error is the command's.
fn run_engine(a: &Args, cfg: EngineConfig, jobs: Receiver<Job>, ready: Sender<()>) -> Res {
    let mut eng = a.engine.load(cfg)?;
    a.draft.load(&mut eng)?;
    let mut g = eng.generator()?;
    let _ = ready.send(());
    let mut live: HashMap<ReqId, Reply> = HashMap::new();
    loop {
        let mut new = Vec::new();
        if g.is_idle() {
            match jobs.recv() {
                Ok(j) => new.push(j),
                Err(_) => return Ok(()),
            }
        }
        new.extend(jobs.try_iter());
        for j in new {
            match g.submit(j.req) {
                Ok(id) => drop(live.insert(id, j.reply)),
                Err(e) => drop(j.reply.tx.send((j.reply.choice, Out::Error(e)))),
            }
        }
        live.retain(|&id, r| {
            let keep = !r.cancel.load(Ordering::Relaxed);
            if !keep {
                g.cancel(id);
            }
            keep
        });
        if g.is_idle() {
            continue;
        }
        let events = g.step().unwrap_or_else(|e| {
            eprintln!("engine failed: {e}");
            std::process::exit(1)
        });
        for ev in events {
            let (id, out) = match ev {
                Event::Token(id, t) => (id, Out::Token(t)),
                Event::Done(id, f) => (id, Out::Done(f)),
            };
            let done = matches!(out, Out::Done(_));
            let Some(r) = live.get(&id) else { continue };
            // A closed channel: the client is gone.
            if r.tx.send((r.choice, out)).is_err() {
                g.cancel(id);
                live.remove(&id);
            } else if done {
                live.remove(&id);
            }
        }
    }
}

fn handle(mut s: TcpStream, ctx: &Ctx) {
    let _ = s.set_read_timeout(Some(Duration::from_secs(60)));
    let res = read_request(&s).and_then(|(method, path, body)| {
        let parse = || serde_json::from_slice::<Value>(&body).map_err(|e| bad(format!("invalid JSON: {e}")));
        match (method.as_str(), path.as_str()) {
            ("GET", "/health") => Ok(Some(json!({}))),
            ("GET", "/v1/models") => Ok(Some(models(ctx))),
            ("POST", "/v1/chat/completions") => complete(ctx, &mut s, &parse()?, true),
            ("POST", "/v1/completions") => complete(ctx, &mut s, &parse()?, false),
            ("POST", "/tokenize") => tokenize(ctx, &parse()?).map(Some),
            ("POST", "/detokenize") => detokenize(ctx, &parse()?).map(Some),
            (_, "/health" | "/v1/models" | "/v1/chat/completions" | "/v1/completions" | "/tokenize" | "/detokenize") => {
                Err(HttpError(405, format!("{method} not allowed on {path}")))
            }
            _ => Err(HttpError(404, format!("no route {path}"))),
        }
    });
    match res {
        Ok(Some(v)) => send_json(&mut s, 200, &v),
        Ok(None) => {} // streamed
        Err(HttpError(0, _)) => {} // the client is gone
        Err(HttpError(code, msg)) => {
            let kind = if code < 500 { "invalid_request_error" } else { "server_error" };
            send_json(&mut s, code, &json!({"error": {"message": msg, "type": kind, "param": null, "code": null}}));
        }
    }
}

/// (method, path without the query, body).
fn read_request(s: &TcpStream) -> Result<(String, String, Vec<u8>), HttpError> {
    let mut r = BufReader::new(s);
    let mut line = String::new();
    let read_line = |r: &mut BufReader<&TcpStream>, line: &mut String| {
        line.clear();
        r.by_ref().take(1 << 16).read_line(line).map_err(|e| bad(format!("reading the request: {e}")))
    };
    read_line(&mut r, &mut line)?;
    let mut f = line.split_whitespace();
    let (method, target) = (f.next().unwrap_or_default().to_string(), f.next().unwrap_or_default());
    let path = target.split('?').next().unwrap_or_default().to_string();
    let (mut len, mut expect) = (0, false);
    loop {
        if read_line(&mut r, &mut line)? == 0 {
            return Err(bad("the request ended in its headers"));
        }
        let l = line.trim_end();
        if l.is_empty() {
            break;
        }
        let Some((k, v)) = l.split_once(':') else { continue };
        let v = v.trim();
        match k.trim().to_ascii_lowercase().as_str() {
            "content-length" => len = v.parse().map_err(|_| bad("bad Content-Length"))?,
            "expect" => expect = v.eq_ignore_ascii_case("100-continue"),
            "transfer-encoding" if !v.eq_ignore_ascii_case("identity") => {
                return Err(HttpError(411, "send a Content-Length".into()))
            }
            _ => {}
        }
    }
    if len > MAX_BODY {
        return Err(HttpError(413, format!("bodies over {MAX_BODY} bytes aren't taken")));
    }
    if expect && len > 0 {
        let mut w = *r.get_ref();
        let _ = w.write_all(b"HTTP/1.1 100 Continue\r\n\r\n");
    }
    let mut body = vec![0; len];
    r.read_exact(&mut body).map_err(|e| bad(format!("reading the body: {e}")))?;
    Ok((method, path, body))
}

fn send_json(s: &mut TcpStream, code: u16, v: &Value) {
    let body = v.to_string();
    let reason = match code {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        411 => "Length Required",
        413 => "Payload Too Large",
        _ => "Internal Server Error",
    };
    let head = format!("HTTP/1.1 {code} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
    let _ = s.write_all(head.as_bytes()).and_then(|_| s.write_all(body.as_bytes()));
}

/// One server-sent event; a failed write means the client is gone.
fn send_event(s: &mut TcpStream, data: &str) -> Result<(), HttpError> {
    s.write_all(format!("data: {data}\n\n").as_bytes()).and_then(|_| s.flush()).map_err(|e| HttpError(0, e.to_string()))
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn models(ctx: &Ctx) -> Value {
    json!({"object": "list", "data": [{"id": ctx.model, "object": "model", "created": ctx.started,
        "owned_by": "violetto", "max_model_len": ctx.max_seq_len}]})
}

fn tokenize(ctx: &Ctx, body: &Value) -> Result<Value, HttpError> {
    let text = match (body.get("messages"), body.get("prompt").and_then(Value::as_str)) {
        (Some(m), _) => {
            let add = body.get("add_generation_prompt").and_then(Value::as_bool).unwrap_or(true);
            text::render(&messages(m)?, add).map_err(bad)?
        }
        (None, Some(p)) => p.to_string(),
        _ => return Err(bad("give a prompt (text) or messages")),
    };
    let ids = encode(ctx, &text)?;
    Ok(json!({"count": ids.len(), "max_model_len": ctx.max_seq_len, "tokens": ids}))
}

fn detokenize(ctx: &Ctx, body: &Value) -> Result<Value, HttpError> {
    let ids = token_ids(body.get("tokens").ok_or_else(|| bad("give tokens"))?)?;
    Ok(json!({"prompt": ctx.tok.decode(&ids, false).map_err(|e| bad(e.to_string()))?}))
}

fn encode(ctx: &Ctx, text: &str) -> Result<Vec<u32>, HttpError> {
    Ok(ctx.tok.encode(text, false).map_err(|e| bad(e.to_string()))?.get_ids().to_vec())
}

fn token_ids(v: &Value) -> Result<Vec<u32>, HttpError> {
    let ids = v.as_array().ok_or_else(|| bad("token ids must be an array"))?;
    ids.iter()
        .map(|x| x.as_u64().filter(|&t| t < VOCAB_REAL as u64).map(|t| t as u32))
        .collect::<Option<_>>()
        .ok_or_else(|| bad(format!("token ids must be integers below {VOCAB_REAL}")))
}

fn messages(v: &Value) -> Result<Vec<Message>, HttpError> {
    let text = |v: Option<&Value>, what: &str| -> Result<String, HttpError> {
        match v {
            None | Some(Value::Null) => Ok(String::new()),
            Some(Value::String(s)) => Ok(s.clone()),
            // Content parts: only text, joined.
            Some(Value::Array(parts)) => parts
                .iter()
                .map(|p| match (p.get("type").and_then(Value::as_str), p.get("text").and_then(Value::as_str)) {
                    (Some("text"), Some(t)) => Ok(t),
                    _ => Err(bad(format!("{what}: only text content is supported"))),
                })
                .collect(),
            _ => Err(bad(format!("{what} must be a string"))),
        }
    };
    v.as_array()
        .ok_or_else(|| bad("messages must be an array"))?
        .iter()
        .map(|m| {
            if m.get("tool_calls").is_some_and(|t| !t.is_null()) || m.get("function_call").is_some_and(|t| !t.is_null()) {
                return Err(bad("Limite Violetto supports text-only chat; tool calls are not supported."));
            }
            let reasoning = match m.get("reasoning_content") {
                None | Some(Value::Null) => None,
                r => Some(text(r, "reasoning_content")?),
            };
            Ok(Message {
                role: m.get("role").and_then(Value::as_str).ok_or_else(|| bad("a message has no role"))?.to_string(),
                content: text(m.get("content"), "content")?,
                reasoning,
            })
        })
        .collect()
}

/// The generation settings of a request.
struct Settings {
    n: usize,
    max_tokens: usize,
    sampling: Sampling,
    stop_ids: Vec<u32>,
    stops: Vec<String>,
    stream: bool,
    usage_chunk: bool,
    token_ids: bool,
}

fn settings(ctx: &Ctx, body: &Value, prompt_len: usize) -> Result<Settings, HttpError> {
    let num = |k: &str| match body.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v.as_f64().map(Some).ok_or_else(|| bad(format!("{k} must be a number"))),
    };
    let uint = |k: &str| match body.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v.as_u64().map(Some).ok_or_else(|| bad(format!("{k} must be a non-negative integer"))),
    };
    let flag = |k: &str| body.get(k).and_then(Value::as_bool).unwrap_or(false);
    // Accepted only at values that don't change the output.
    let unsupported = |k: &str, v: &Value| match k {
        "logprobs" | "echo" => v.as_bool() != Some(false) && v.as_u64() != Some(0),
        "presence_penalty" | "frequency_penalty" => v.as_f64() != Some(0.0),
        "best_of" => v.as_u64() != Some(1),
        "logit_bias" => v.as_object().is_none_or(|o| !o.is_empty()),
        "tools" | "functions" => v.as_array().is_none_or(|a| !a.is_empty()),
        "response_format" => v.get("type").and_then(Value::as_str) != Some("text"),
        _ => true,
    };
    for k in ["logprobs", "top_logprobs", "echo", "presence_penalty", "frequency_penalty", "best_of", "logit_bias", "tools", "functions", "response_format", "suffix"] {
        if body.get(k).is_some_and(|v| !v.is_null() && unsupported(k, v)) {
            return Err(bad(format!("{k} is not supported")));
        }
    }
    let n = uint("n")?.unwrap_or(1) as usize;
    if !(1..=128).contains(&n) {
        return Err(bad("n must be 1 to 128"));
    }
    let room = ctx.max_seq_len.saturating_sub(prompt_len);
    let max_tokens = match uint("max_completion_tokens")?.or(uint("max_tokens")?) {
        Some(m) => m as usize,
        None => room,
    };
    if prompt_len == 0 || max_tokens == 0 || max_tokens > room {
        return Err(bad(format!(
            "This model's maximum context length is {} tokens. However, you requested {} tokens ({prompt_len} in the prompt, {max_tokens} for the completion).",
            ctx.max_seq_len,
            prompt_len + max_tokens
        )));
    }
    let temperature = num("temperature")?.unwrap_or(0.6);
    let top_p = num("top_p")?.unwrap_or(0.95);
    if !(temperature == 0.0 || (1e-4..=100.0).contains(&temperature)) || !(top_p > 0.0 && top_p <= 1.0) {
        return Err(bad("temperature must be 0 or 1e-4 to 100, and top_p in (0, 1]"));
    }
    // No seed: a fresh one, so requests differ.
    let seed = match body.get("seed") {
        None | Some(Value::Null) => {
            let t = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
            t ^ ctx.requests.load(Ordering::Relaxed).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        }
        Some(v) => v.as_i64().map(|s| s as u64).or(v.as_u64()).ok_or_else(|| bad("seed must be an integer"))?,
    };
    let stops = match body.get("stop") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().map(|s| s.as_str().map(str::to_string)).collect::<Option<_>>().ok_or_else(|| bad("stop must be strings"))?,
        _ => return Err(bad("stop must be a string or an array of strings")),
    };
    if stops.iter().any(String::is_empty) {
        return Err(bad("stop strings can't be empty"));
    }
    let mut stop_ids = vec![ENDOFTEXT, IM_END];
    if let Some(v) = body.get("stop_token_ids").filter(|v| !v.is_null()) {
        stop_ids.extend(token_ids(v)?);
    }
    Ok(Settings {
        n,
        max_tokens,
        sampling: Sampling { temperature: temperature as f32, top_p: top_p as f32, seed },
        stop_ids,
        stops,
        stream: flag("stream"),
        usage_chunk: body.get("stream_options").and_then(|o| o.get("include_usage")).and_then(Value::as_bool).unwrap_or(false),
        token_ids: flag("return_token_ids"),
    })
}

/// Sets the cancel flags of a request's jobs when its handler returns, however it does.
struct CancelOnDrop(Vec<Arc<AtomicBool>>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        for c in &self.0 {
            c.store(true, Ordering::Relaxed);
        }
    }
}

/// /v1/chat/completions (`chat`) or /v1/completions. Returns the response, or None once it
/// has been streamed.
fn complete(ctx: &Ctx, s: &mut TcpStream, body: &Value, chat: bool) -> Result<Option<Value>, HttpError> {
    let t0 = Instant::now();
    let prompt = if chat {
        let m = messages(body.get("messages").ok_or_else(|| bad("give messages"))?)?;
        encode(ctx, &text::render(&m, true).map_err(bad)?)?
    } else {
        match body.get("prompt") {
            Some(Value::Array(a)) if a.len() == 1 && !a[0].is_number() => match &a[0] {
                Value::String(p) => encode(ctx, p)?,
                v => token_ids(v)?,
            },
            Some(Value::String(p)) => encode(ctx, p)?,
            Some(v @ Value::Array(_)) => token_ids(v).map_err(|_| bad("one prompt per request: text, or an array of token ids"))?,
            _ => return Err(bad("give a prompt: text, or an array of token ids")),
        }
    };
    let st = settings(ctx, body, prompt.len())?;
    let num = ctx.requests.fetch_add(1, Ordering::Relaxed);
    let id = if chat { format!("chatcmpl-{num}") } else { format!("cmpl-{num}") };
    let (object, created) = (if chat { "chat.completion" } else { "text_completion" }, now());

    let (tx, rx) = mpsc::channel();
    let cancels: Vec<Arc<AtomicBool>> = (0..st.n).map(|_| Arc::new(AtomicBool::new(false))).collect();
    let _cancel = CancelOnDrop(cancels.clone());
    for (choice, cancel) in cancels.iter().enumerate() {
        let req = Request {
            prompt: Prompt::Tokens(prompt.clone()),
            max_tokens: st.max_tokens,
            stop: st.stop_ids.clone(),
            sampling: Sampling { seed: st.sampling.seed.wrapping_add(choice as u64), ..st.sampling },
            draft: ctx.draft.clone(),
        };
        let job = Job { req, reply: Reply { choice, tx: tx.clone(), cancel: cancel.clone() } };
        ctx.jobs.send(job).map_err(|_| HttpError(500, "the engine is gone".into()))?;
    }
    drop(tx);

    let mut outs: Vec<Output> = (0..st.n).map(|_| Output::new(&ctx.tok, chat, st.stops.clone())).collect();
    let mut finish: Vec<Option<&str>> = vec![None; st.n];
    let chunk = |choices: Value, usage: Option<Value>| {
        let mut c = json!({"id": id, "object": format!("{object}.chunk"), "created": created, "model": ctx.model, "choices": choices});
        if !chat {
            c["object"] = json!("text_completion");
        }
        if st.usage_chunk {
            c["usage"] = usage.unwrap_or(Value::Null);
        }
        c.to_string()
    };
    if st.stream {
        let head = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
        s.write_all(head.as_bytes()).map_err(|e| HttpError(0, e.to_string()))?;
        if chat {
            let first: Vec<Value> = (0..st.n).map(|i| json!({"index": i, "delta": {"role": "assistant", "content": ""}, "logprobs": null, "finish_reason": null})).collect();
            send_event(s, &chunk(json!(first), None))?;
        }
    }
    while finish.iter().any(Option::is_none) {
        let Ok((i, out)) = rx.recv() else { return Err(HttpError(500, "the engine is gone".into())) };
        if finish[i].is_some() {
            continue; // cut by a stop string, and cancelled
        }
        match out {
            Out::Token(t) => {
                if outs[i].push(t) {
                    finish[i] = Some("stop");
                    cancels[i].store(true, Ordering::Relaxed);
                }
            }
            Out::Done(f) => {
                outs[i].finish();
                finish[i] = Some(if f == Finish::Length { "length" } else { "stop" });
            }
            // The generator refused the request.
            Out::Error(e) if st.stream => {
                send_event(s, &json!({"error": {"message": e, "type": "invalid_request_error"}}).to_string())?;
                return Ok(None);
            }
            Out::Error(e) => return Err(bad(e)),
        }
        if st.stream {
            let [reasoning, content] = outs[i].take(finish[i].is_some());
            let mut choices = Vec::new();
            if chat {
                if let Some(r) = reasoning {
                    choices.push(json!({"index": i, "delta": {"reasoning_content": r}, "logprobs": null, "finish_reason": null}));
                }
                if let Some(c) = content {
                    choices.push(json!({"index": i, "delta": {"content": c}, "logprobs": null, "finish_reason": null}));
                }
                if let Some(f) = finish[i] {
                    let mut last = json!({"index": i, "delta": {}, "logprobs": null, "finish_reason": f});
                    if st.token_ids {
                        last["token_ids"] = json!(outs[i].tokens);
                    }
                    choices.push(last);
                }
            } else if content.is_some() || finish[i].is_some() {
                let mut c = json!({"index": i, "text": content.unwrap_or_default(), "logprobs": null, "finish_reason": finish[i]});
                if st.token_ids && finish[i].is_some() {
                    c["token_ids"] = json!(outs[i].tokens);
                }
                choices.push(c);
            }
            for c in choices {
                send_event(s, &chunk(json!([c]), None))?;
            }
        }
    }

    let completion: usize = outs.iter().map(|o| o.tokens.len()).sum();
    let usage = json!({"prompt_tokens": prompt.len(), "completion_tokens": completion, "total_tokens": prompt.len() + completion});
    let secs = t0.elapsed().as_secs_f64();
    eprintln!("{id}: {} prompt + {completion} generated tokens, n={}, {:?}, {secs:.1} s", prompt.len(), st.n, finish.iter().flatten().collect::<Vec<_>>());
    if st.stream {
        if st.usage_chunk {
            send_event(s, &chunk(json!([]), Some(usage)))?;
        }
        send_event(s, "[DONE]")?;
        return Ok(None);
    }
    let choices: Vec<Value> = outs
        .iter()
        .zip(&finish)
        .enumerate()
        .map(|(i, (o, f))| {
            let mut c = if chat {
                let reasoning = (!o.text[0].is_empty()).then(|| o.text[0].clone());
                json!({"index": i, "message": {"role": "assistant", "content": o.text[1], "reasoning_content": reasoning}, "logprobs": null, "finish_reason": f})
            } else {
                json!({"index": i, "text": o.text[1], "logprobs": null, "finish_reason": f})
            };
            if st.token_ids {
                c["token_ids"] = json!(o.tokens);
            }
            c
        })
        .collect();
    let mut r = json!({"id": id, "object": object, "created": created, "model": ctx.model, "choices": choices, "usage": usage});
    if st.token_ids {
        r["prompt_token_ids"] = json!(prompt);
    }
    Ok(Some(r))
}
