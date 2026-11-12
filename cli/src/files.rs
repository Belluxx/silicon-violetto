use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;

use crate::Res;

pub fn read(path: &Path) -> Res<String> {
    std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn write(path: &Path, bytes: impl AsRef<[u8]>) -> Res {
    std::fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// The fields of each line that isn't blank, with its number for error messages.
pub fn lines<'a>(path: &'a Path, text: &'a str) -> impl Iterator<Item = (String, std::str::SplitWhitespace<'a>)> {
    let lines = text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty());
    lines.map(move |(i, l)| (format!("{}:{}", path.display(), i + 1), l.split_whitespace()))
}

pub fn parse<T: std::str::FromStr>(at: &str, what: &str, v: &str) -> Res<T> {
    v.parse().map_err(|_| format!("{at}: bad {what} {v}"))
}

pub fn ids<'a>(at: &str, fields: impl Iterator<Item = &'a str>) -> Res<Vec<u32>> {
    fields.map(|x| parse(at, "token id", x)).collect()
}

/// One line of a request file: `name opts id id id ...`.
pub struct RequestLine {
    pub name: String,
    /// `-`, or comma-separated key=value pairs.
    pub opts: String,
    pub prompt: Vec<u32>,
}

pub fn requests(path: &Path) -> Res<Vec<RequestLine>> {
    let text = read(path)?;
    lines(path, &text)
        .map(|(at, mut f)| {
            let name = f.next().unwrap().to_string();
            let opts = f.next().ok_or_else(|| format!("{at}: no options (`-` for none)"))?.to_string();
            Ok(RequestLine { name, opts, prompt: ids(&at, f)? })
        })
        .collect()
}

/// The prompts of a request file, cut at `ctx` tokens.
pub fn prompts(path: &Path, ctx: usize) -> Res<Vec<Vec<u32>>> {
    let cut = |r: RequestLine| match r.prompt.len() >= ctx {
        true => Ok(r.prompt[..ctx].to_vec()),
        false => Err(format!("{}: {} has {} tokens, under the context of {ctx}", path.display(), r.name, r.prompt.len())),
    };
    requests(path)?.into_iter().map(cut).collect()
}

/// Name -> tokens of a `generate` output.
pub fn outputs(path: &Path) -> Res<HashMap<String, Vec<u32>>> {
    let text = read(path)?;
    let lines = text.lines().enumerate().filter(|(_, l)| !l.trim().is_empty());
    lines
        .map(|(i, l)| {
            let bad = |what: &str| format!("{}:{}: {what}", path.display(), i + 1);
            let v: Value = serde_json::from_str(l).map_err(|e| bad(&e.to_string()))?;
            let name = v["name"].as_str().ok_or_else(|| bad("no name"))?.to_string();
            let tokens = v["tokens"].as_array().ok_or_else(|| bad("no tokens"))?;
            let tokens = tokens.iter().map(|t| t.as_u64().map(|t| t as u32)).collect::<Option<_>>();
            Ok((name, tokens.ok_or_else(|| bad("tokens aren't ids"))?))
        })
        .collect()
}
