"""Conversation history: persistence and safe windowing (OpenAI message format).

Messages are plain dicts:
  {"role": "user", "content": str}
  {"role": "assistant", "content": str | None, "tool_calls": [...]}   (optional tool_calls)
  {"role": "tool", "tool_call_id": str, "content": str}
The system prompt is NOT stored; the agent prepends it on every request.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

MAX_MESSAGES = 40  # window size sent to the model


def is_user_text(msg: dict) -> bool:
    """True for a real user turn (not a tool result)."""
    return msg.get("role") == "user" and isinstance(msg.get("content"), str)


def window(messages: list[dict], max_messages: int = MAX_MESSAGES) -> list[dict]:
    """Return the last ``max_messages`` messages, starting on a user turn.

    Starting on a user message guarantees a window never begins with an orphan
    ``tool`` message and never splits an assistant ``tool_calls`` message from
    its ``tool`` replies (they always sit between two user turns).
    """
    if len(messages) <= max_messages:
        return list(messages)
    tail = messages[-max_messages:]
    for i, msg in enumerate(tail):
        if is_user_text(msg):
            return tail[i:]
    # No user turn inside the window: fall back to the latest user turn.
    for i in range(len(messages) - 1, -1, -1):
        if is_user_text(messages[i]):
            return messages[i:]
    return []


def _safe_id(session_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", session_id) or "default"


class HistoryStore:
    def __init__(self, directory: Path | str = "sessions"):
        self.dir = Path(directory)

    def path(self, session_id: str) -> Path:
        return self.dir / f"{_safe_id(session_id)}.json"

    def load(self, session_id: str) -> list[dict]:
        p = self.path(session_id)
        if not p.exists():
            return []
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []  # corrupt file: start fresh instead of crashing the chat
        return data if isinstance(data, list) else []

    def save(self, session_id: str, messages: list[dict]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path(session_id).with_suffix(".tmp")
        tmp.write_text(json.dumps(messages, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path(session_id))  # atomic on POSIX
