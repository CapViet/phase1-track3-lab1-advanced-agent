"""Provider-agnostic chat LLM client used by the real (non-mock) runtime.

The lab allows any backend (Ollama, OpenAI, Gemini, Anthropic). This module exposes a
single `chat()` entry point that returns the generated text together with real usage
metrics (token counts + measured latency) so the agent can report true costs instead of
the hardcoded estimates in the scaffold.

Provider/model are selected via environment variables (a .env file is auto-loaded):
    LLM_PROVIDER   one of: ollama | openai | gemini | anthropic   (default: ollama)
    LLM_MODEL      model name                                     (default per provider)
    OLLAMA_HOST    base url for ollama                            (default: http://localhost:11434)
plus the usual OPENAI_API_KEY / GEMINI_API_KEY / ANTHROPIC_API_KEY for hosted providers.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

try:  # optional, but present in this lab's requirements
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is best-effort
    pass


DEFAULT_MODELS = {
    "ollama": "qwen2.5:3b",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "anthropic": "claude-haiku-4-5",
}


@dataclass
class Usage:
    """Real usage metrics for a single LLM call."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMResponse:
    text: str
    usage: Usage


@dataclass
class LLMClient:
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "ollama").lower())
    model: Optional[str] = field(default_factory=lambda: os.getenv("LLM_MODEL"))
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not self.model:
            self.model = DEFAULT_MODELS.get(self.provider, DEFAULT_MODELS["ollama"])

    def chat(self, system: str, user: str) -> LLMResponse:
        start = time.perf_counter()
        if self.provider == "ollama":
            text, usage = self._ollama(system, user)
        elif self.provider == "openai":
            text, usage = self._openai(system, user)
        elif self.provider == "gemini":
            text, usage = self._gemini(system, user)
        elif self.provider == "anthropic":
            text, usage = self._anthropic(system, user)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {self.provider!r}")
        # Always overwrite latency with the wall-clock measurement around the call.
        usage.latency_ms = int((time.perf_counter() - start) * 1000)
        return LLMResponse(text=text.strip(), usage=usage)

    # ---- providers -------------------------------------------------------
    def _ollama(self, system: str, user: str) -> tuple[str, Usage]:
        import ollama

        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        client = ollama.Client(host=host)
        resp = client.chat(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            options={"temperature": self.temperature},
        )
        text = resp["message"]["content"]
        usage = Usage(
            prompt_tokens=int(resp.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(resp.get("eval_count", 0) or 0),
        )
        return text, usage

    def _openai(self, system: str, user: str) -> tuple[str, Usage]:
        from openai import OpenAI

        client = OpenAI()
        resp = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        text = resp.choices[0].message.content or ""
        u = resp.usage
        usage = Usage(
            prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(u, "completion_tokens", 0) or 0,
        )
        return text, usage

    def _gemini(self, system: str, user: str) -> tuple[str, Usage]:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        resp = client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(system_instruction=system, temperature=self.temperature),
        )
        text = resp.text or ""
        meta = getattr(resp, "usage_metadata", None)
        usage = Usage(
            prompt_tokens=getattr(meta, "prompt_token_count", 0) or 0,
            completion_tokens=getattr(meta, "candidates_token_count", 0) or 0,
        )
        return text, usage

    def _anthropic(self, system: str, user: str) -> tuple[str, Usage]:
        import anthropic

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")
        usage = Usage(
            prompt_tokens=resp.usage.input_tokens,
            completion_tokens=resp.usage.output_tokens,
        )
        return text, usage
