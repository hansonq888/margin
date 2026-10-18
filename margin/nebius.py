"""Thin client for Nebius Token Factory (OpenAI-compatible) with JSON helpers and timing."""

import base64
import io
import json
import re
import time
from dataclasses import dataclass, field

import openai
from openai import OpenAI
from PIL import Image

from margin import config

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass
class Result:
    data: dict | None
    text: str
    latency_s: float
    model: str
    usage: dict = field(default_factory=dict)
    error: str | None = None


class Nebius:
    def __init__(self, cfg: config.Config | None = None):
        self.cfg = cfg or config.load()
        if not self.cfg.has_api_key:
            raise RuntimeError("NEBIUS_API_KEY is not set. Copy .env.example to .env and add your key.")
        self.client = OpenAI(api_key=self.cfg.api_key, base_url=self.cfg.base_url)
        # Remember which optional parameters a model rejected so we only pay for the retry once.
        self._unsupported: dict[str, set[str]] = {}

    def chat_json(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        reasoning_off: bool = True,
    ) -> Result:
        """Call a chat model and parse a JSON object out of its reply."""
        messages = list(messages)
        extra: dict = {}
        mode = self.cfg.reasoning_off if reasoning_off else "none"
        if mode == "kwargs":
            extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        elif mode == "system":
            messages = [{"role": "system", "content": "/no_think"}, *messages]
        extra["response_format"] = {"type": "json_object"}

        skip = self._unsupported.setdefault(model, set())
        for key in list(extra):
            if key in skip:
                extra.pop(key)

        start = time.perf_counter()
        while True:
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **extra,
                )
                break
            except openai.BadRequestError as e:
                # Some endpoints reject response_format or chat_template_kwargs; drop one and retry.
                dropped = next((k for k in ("response_format", "extra_body") if k in extra), None)
                if dropped is None:
                    return Result(None, "", time.perf_counter() - start, model, error=str(e))
                extra.pop(dropped)
                skip.add(dropped)
            except openai.OpenAIError as e:
                return Result(None, "", time.perf_counter() - start, model, error=str(e))

        latency = time.perf_counter() - start
        text = resp.choices[0].message.content or ""
        usage = resp.usage.model_dump() if resp.usage else {}
        data = parse_json(text)
        err = None if data is not None else "reply was not valid JSON"
        return Result(data, text, latency, model, usage, err)


    def chat_stream(self, model: str, messages: list[dict], max_tokens: int = 300, temperature: float = 0.3):
        """Stream a reply as text pieces (reasoning off, <think> blocks dropped)."""
        extra: dict = {}
        if self.cfg.reasoning_off == "kwargs" and "extra_body" not in self._unsupported.get(model, set()):
            extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        elif self.cfg.reasoning_off == "system":
            messages = [{"role": "system", "content": "/no_think"}, *messages]
        stream = self.client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, stream=True, **extra
        )
        in_think = False
        for chunk in stream:
            if not chunk.choices:
                continue
            piece = chunk.choices[0].delta.content or ""
            # Drop any <think>...</think> the model emits despite reasoning being off.
            while piece:
                if in_think:
                    end = piece.find("</think>")
                    if end == -1:
                        piece = ""
                    else:
                        in_think, piece = False, piece[end + len("</think>") :]
                else:
                    start = piece.find("<think>")
                    if start == -1:
                        yield piece
                        piece = ""
                    else:
                        if start:
                            yield piece[:start]
                        in_think, piece = True, piece[start + len("<think>") :]


def parse_json(text: str) -> dict | None:
    """Extract the first JSON object from a model reply (tolerates <think> blocks and code fences)."""
    text = _THINK_RE.sub("", text).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def image_part(img: Image.Image, max_side: int = 1280) -> dict:
    """Encode an image as an OpenAI-style image_url content part."""
    img = img.convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
