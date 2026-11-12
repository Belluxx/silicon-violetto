use tokenizers::Tokenizer;

pub const ENDOFTEXT: u32 = 151643;
pub const IM_END: u32 = 151645;
const THINK: u32 = 151665;
const END_THINK: u32 = 151666;

/// The fixed system prompt of models/limite-1b-violetto/chat_template.jinja.
const SYSTEM: &str = "You are a helpful assistant.\nPlease reason step by step, and put your final answer within \\boxed{}.";

pub struct Message {
    pub role: String,
    pub content: String,
    pub reasoning: Option<String>,
}

/// What chat_template.jinja renders, with the same refusals.
pub fn render(messages: &[Message], add_generation_prompt: bool) -> Result<String, String> {
    let mut s = format!("<|endoftext|><|im_start|>system\n{SYSTEM}<|im_end|>\n");
    for m in messages {
        match m.role.as_str() {
            "system" if m.content == SYSTEM => {}
            "system" => {
                return Err("Limite Violetto requires its fixed mathematical system prompt; custom system messages are not supported.".into())
            }
            "user" => s += &format!("<|im_start|>user\n{}<|im_end|>\n", m.content),
            "assistant" => {
                s += "<|im_start|>assistant\n";
                if let Some(r) = m.reasoning.as_deref().filter(|r| !r.is_empty()) {
                    s += &format!("<think>\n{r}\n</think>\n\n");
                }
                s += &format!("{}<|im_end|>\n", m.content);
            }
            _ => {
                return Err("Limite Violetto supports only user, assistant, and the exact canonical system message; other roles are not supported.".into())
            }
        }
    }
    if add_generation_prompt {
        s += "<|im_start|>assistant\n";
    }
    Ok(s)
}

/// Tokens to text a few at a time, so a character split over tokens comes out whole (as
/// vLLM's incremental detokenizer). Special tokens decode to nothing.
#[derive(Default)]
struct Decoder {
    ids: Vec<u32>,
    prefix: usize,
    read: usize,
}

impl Decoder {
    fn push(&mut self, tok: &Tokenizer, id: u32) -> String {
        self.ids.push(id);
        let decode = |ids: &[u32]| tok.decode(ids, true).unwrap_or_default();
        let (old, new) = (decode(&self.ids[self.prefix..self.read]), decode(&self.ids[self.prefix..]));
        match new.get(old.len()..) {
            Some(delta) if new.len() > old.len() && new.starts_with(&old) && !new.ends_with('\u{FFFD}') => {
                let delta = delta.to_string();
                (self.prefix, self.read) = (self.read, self.ids.len());
                delta
            }
            _ => String::new(),
        }
    }

    /// What's held back, even an incomplete character.
    fn flush(&mut self, tok: &Tokenizer) -> String {
        let decode = |ids: &[u32]| tok.decode(ids, true).unwrap_or_default();
        let (old, new) = (decode(&self.ids[self.prefix..self.read]), decode(&self.ids[self.prefix..]));
        self.prefix = self.ids.len();
        self.read = self.ids.len();
        new.get(old.len()..).filter(|_| new.starts_with(&old)).unwrap_or_default().to_string()
    }
}

/// One choice's generated text, as reasoning and content. With `split`, a generation that
/// opens with <think> puts everything up to </think> in reasoning, and the content after it
/// starts at its first non-blank. The text ends at the first stop string, and only text no
/// stop string can still begin in is released to a stream.
pub struct Output<'t> {
    tok: &'t Tokenizer,
    split: bool,
    stops: Vec<String>,
    /// Bytes a stop string could still need from the end of the text.
    hold: usize,
    dec: Decoder,
    pub tokens: Vec<u32>,
    /// Reasoning, content.
    pub text: [String; 2],
    sent: [usize; 2],
    part: usize,
    thought: bool,
    pub stopped: bool,
}

impl<'t> Output<'t> {
    pub fn new(tok: &'t Tokenizer, split: bool, stops: Vec<String>) -> Self {
        let hold = stops.iter().map(String::len).max().unwrap_or(1) - 1;
        let (dec, text) = (Decoder::default(), [String::new(), String::new()]);
        Self { tok, split, stops, hold, dec, tokens: Vec::new(), text, sent: [0, 0], part: 1, thought: false, stopped: false }
    }

    /// Adds a generated token; true once a stop string has ended the text.
    pub fn push(&mut self, id: u32) -> bool {
        self.tokens.push(id);
        if self.split && self.tokens.len() == 1 && id == THINK {
            self.part = 0;
        } else if self.split && self.part == 0 && id == END_THINK {
            let rest = self.dec.flush(self.tok);
            self.add(rest);
            (self.part, self.thought, self.dec) = (1, true, Decoder::default());
        } else {
            let delta = self.dec.push(self.tok, id);
            self.add(delta);
        }
        self.stopped
    }

    /// The generation ended: adds what the decoder held back.
    pub fn finish(&mut self) {
        if !self.stopped {
            let rest = self.dec.flush(self.tok);
            self.add(rest);
        }
    }

    fn add(&mut self, delta: String) {
        if self.stopped || delta.is_empty() {
            return;
        }
        let t = &mut self.text[self.part];
        let delta = if self.part == 1 && self.thought && t.is_empty() { delta.trim_start() } else { &delta[..] };
        let from = floor_char(t, t.len().saturating_sub(self.hold));
        t.push_str(delta);
        if let Some(p) = self.stops.iter().filter_map(|s| t[from..].find(s.as_str()).map(|p| p + from)).min() {
            t.truncate(p);
            self.stopped = true;
        }
    }

    /// New text for a stream, (reasoning, content); `all` at the end.
    pub fn take(&mut self, all: bool) -> [Option<String>; 2] {
        let mut out = [None, None];
        for (i, o) in out.iter_mut().enumerate() {
            let t = &self.text[i];
            let end = if all || self.stopped { t.len() } else { floor_char(t, t.len().saturating_sub(self.hold)) };
            if end > self.sent[i] {
                *o = Some(t[self.sent[i]..end].to_string());
                self.sent[i] = end;
            }
        }
        out
    }
}

fn floor_char(s: &str, mut i: usize) -> usize {
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}
