from __future__ import annotations

import argparse
import contextlib
import json
import readline  # noqa: F401 (line editing for input())
import shutil
import sys
import threading
import time
from collections import deque

import httpx

DIM, BAR, PLAIN = "\x1b[2m", "\x1b[7m", "\x1b[0m"
SAVE, RESTORE = "\x1b7", "\x1b8"
REFRESH = 0.1
WINDOW = 1.0  # the live rate's span, seconds: a speculative step lands several tokens at once

lock = threading.Lock()


def out(text: str) -> None:
    with lock:
        sys.stdout.write(text)
        sys.stdout.flush()


class Meter:
    """A reply's tokens as they arrive."""

    def __init__(self) -> None:
        self.start = time.perf_counter()
        self.first = self.last = 0.0
        self.tokens = 0
        self.phase = "prefill"

    def tick(self, phase: str) -> None:
        self.last = time.perf_counter()
        self.first = self.first or self.last
        self.tokens += 1
        self.phase = phase

    def rate(self, tokens: int) -> float:
        """Tokens a second from the first to the last, which the prefill doesn't slow."""
        return (tokens - 1) / (self.last - self.first) if self.last > self.first else 0.0


class Status:
    """The meter on the terminal's last row while a reply streams; a scroll region keeps the
    reply above it, so it never mixes with the text."""

    def __init__(self, meter: Meter) -> None:
        self.meter = meter
        self.rows = 0
        self.done = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def __enter__(self) -> Status:
        self.thread.start()
        return self

    def __exit__(self, *_) -> None:
        self.done.set()
        self.thread.join()
        out(f"{SAVE}\x1b[r\x1b[{self.rows};1H\x1b[2K{RESTORE}")

    def pin(self, rows: int) -> None:
        # Two newlines and one up: the cursor is on a new line inside the region, whichever
        # row it was on. Setting a region homes the cursor, hence the save and restore.
        out(f"{SAVE}\x1b[r{RESTORE}\n\n{SAVE}\x1b[1;{rows - 1}r{RESTORE}\x1b[1A")
        self.rows = rows

    def run(self) -> None:
        m = self.meter
        samples: deque[tuple[float, int]] = deque()
        while True:
            size = shutil.get_terminal_size()
            if size.lines != self.rows:
                self.pin(size.lines)
            now = time.perf_counter()
            samples.append((now, m.tokens))
            while now - samples[0][0] > WINDOW:
                samples.popleft()
            if m.first:
                since, before = samples[0]
                live = (m.tokens - before) / max(now - max(since, m.first), REFRESH)
                avg = m.rate(m.tokens)
                text = f" {m.phase:<9} {live:5.0f} tok/s   avg {avg:.0f}   {m.tokens} tokens   {now - m.first:.1f} s"
            else:
                text = f" {m.phase:<9} {now - m.start:.1f} s"
            width = size.columns - 1
            out(f"{SAVE}\x1b[{self.rows};1H{BAR}{text[:width].ljust(width)}{PLAIN}{RESTORE}")
            if self.done.wait(REFRESH):
                return


def ask(client: httpx.Client, body: dict, show_reasoning: bool) -> dict | None:
    """Streams one reply to the terminal; returns it as an assistant message, or None if
    there's no whole reply to keep."""
    meter = Meter()
    text = {"reasoning_content": "", "content": ""}
    usage, finish = None, None
    status = Status(meter) if sys.stdout.isatty() else contextlib.nullcontext()
    try:
        with status, client.stream("POST", "/v1/chat/completions", json=body) as r:
            if r.status_code != 200:
                r.read()
                out(f"error: {r.json()['error']['message']}\n")
                return None
            for line in r.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                if "error" in event:
                    out(f"\nerror: {event['error']['message']}\n")
                    return None
                usage = event.get("usage") or usage
                for choice in event["choices"]:
                    finish = choice["finish_reason"] or finish
                    delta = choice["delta"]
                    if piece := delta.get("reasoning_content"):
                        meter.tick("thinking")
                        text["reasoning_content"] += piece
                        if show_reasoning:
                            out(f"{DIM}{piece}{PLAIN}")
                    if piece := delta.get("content"):
                        meter.tick("answering")
                        if not text["content"] and show_reasoning and text["reasoning_content"]:
                            out("\n\n")
                        text["content"] += piece
                        out(piece)
    except KeyboardInterrupt:
        out("\n(stopped)\n")
        return None
    except httpx.TransportError as e:
        out(f"\nthe connection failed: {e}\n")
        return None
    if not usage:  # it comes last, so the stream was cut short
        out("\nthe server closed the stream before the reply ended\n")
        return None
    tokens, prompt = usage["completion_tokens"], usage["prompt_tokens"]
    cut = ", cut at max_tokens" if finish == "length" else ""
    out(f"\n{DIM}{meter.rate(tokens):.0f} tok/s, {tokens} tokens in {meter.last - meter.first:.1f} s, "
        f"{prompt} prompt tokens in {meter.first - meter.start:.2f} s{cut}{PLAIN}\n")
    return {"role": "assistant", **text}


def main() -> None:
    ap = argparse.ArgumentParser(description="Chat with a running `violetto serve`, with live tok/s.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--history", action="store_true", help="send earlier turns too, which the model handles badly")
    ap.add_argument("--hide-reasoning", action="store_true", help="show only the answer")
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--top-p", type=float)
    ap.add_argument("--max-tokens", type=int)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}"
    client = httpx.Client(base_url=base, timeout=httpx.Timeout(None, connect=5))
    try:
        model = client.get("/v1/models").json()["data"][0]
    except httpx.TransportError:
        sys.exit(f"nothing answers at {base}: start `violetto serve`, or give its --port")
    print(f"{model['id']} at {base}, {model['max_model_len']} tokens of context. Ctrl-C stops a reply, /quit leaves.")
    if args.history:
        print("Keeping history: expect the model to answer the first question again. /reset clears it.")

    params = {k: getattr(args, k) for k in ("temperature", "top_p", "max_tokens", "seed")}
    body = {k: v for k, v in params.items() if v is not None}
    body |= {"stream": True, "stream_options": {"include_usage": True}}
    messages: list[dict] = []
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if line == "/quit":
            return
        if line == "/reset":
            messages.clear()
        elif line:
            messages.append({"role": "user", "content": line})
            reply = ask(client, {**body, "messages": messages}, not args.hide_reasoning)
            if reply and args.history:
                messages.append(reply)
            elif reply:
                messages.clear()
            else:
                messages.pop()


if __name__ == "__main__":
    main()
