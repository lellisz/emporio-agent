"""Mel, the agent loop, using an OpenAI-compatible chat API (NVIDIA hosted API or local NIM)."""
from __future__ import annotations

import json
import os
from typing import Any

from .history import HistoryStore, window
from .prompts import SYSTEM_PROMPT
from .tools import TOOL_SCHEMAS, Tools

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "mistralai/mistral-large-4-1t-a52b"
MAX_TOOL_ROUNDS = 6
TEMPERATURE = 0.2
FALLBACK_MESSAGE = ("Sorry, I'm having trouble answering right now. Please try again in a moment, "
                    "or reach us on WhatsApp at (67) 3321-4500.")


def make_client(api_key: str | None = None, base_url: str | None = None):
    """Build an OpenAI SDK client pointed at NVIDIA (hosted) or a local NIM."""
    from openai import OpenAI  # imported lazily so tests never need the SDK configured
    return OpenAI(api_key=api_key or os.environ.get("NVIDIA_API_KEY", "not-needed"),
                  base_url=base_url or os.environ.get("NVIDIA_BASE_URL", DEFAULT_BASE_URL))


class Agent:
    def __init__(self, client: Any = None, model: str | None = None, tools: Tools | None = None,
                 history: HistoryStore | None = None):
        self.client = client or make_client()
        self.model = model or os.environ.get("EMPORIO_MODEL", DEFAULT_MODEL)
        self.tools = tools or Tools()
        self.history = history or HistoryStore()

    # ---- helpers -------------------------------------------------------------
    def _complete(self, messages: list[dict]):
        return self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, *window(messages)],
            tools=TOOL_SCHEMAS, tool_choice="auto", temperature=TEMPERATURE)

    def _run_tool_call(self, call) -> dict:
        name = call.function.name
        raw = call.function.arguments or "{}"
        try:
            args = json.loads(raw)
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
            content = self.tools.run(name, args)
        except (json.JSONDecodeError, ValueError) as exc:
            # Malformed arguments: tell the model, let it retry. Never crash.
            content = json.dumps({"error": f"Invalid JSON arguments for {name}: {exc}. "
                                           "Call the tool again with a valid JSON object."})
        return {"role": "tool", "tool_call_id": call.id, "content": content}

    @staticmethod
    def _assistant_message(msg) -> dict:
        out: dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            out["tool_calls"] = [{"id": c.id, "type": "function",
                                  "function": {"name": c.function.name,
                                               "arguments": c.function.arguments or "{}"}}
                                 for c in msg.tool_calls]
        return out

    # ---- public API ----------------------------------------------------------
    def reply(self, session_id: str, user_text: str) -> str:
        messages = self.history.load(session_id)
        checkpoint = len(messages)
        messages.append({"role": "user", "content": user_text})
        try:
            for _ in range(MAX_TOOL_ROUNDS):
                choice = self._complete(messages).choices[0]
                msg = choice.message
                messages.append(self._assistant_message(msg))
                if choice.finish_reason == "tool_calls" and msg.tool_calls:
                    # Parallel calls: answer every tool_call_id before the next request.
                    messages.extend(self._run_tool_call(c) for c in msg.tool_calls)
                    continue
                text = (msg.content or "").strip()
                if not text:
                    raise RuntimeError("empty model response")
                self.history.save(session_id, messages)
                return text
            raise RuntimeError("too many tool rounds")
        except Exception:
            # Roll back so a failed turn leaves no dangling tool_calls in the history.
            del messages[checkpoint:]
            self.history.save(session_id, messages)
            return FALLBACK_MESSAGE
