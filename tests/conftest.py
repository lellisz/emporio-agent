"""Shared fixtures. All tests run offline: no network, no API key."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from emporio.data import load_store


@pytest.fixture(scope="session")
def store():
    return load_store()


# ---- OpenAI-shaped fake responses -----------------------------------------

def _resp(message, finish_reason):
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])


def text_resp(text: str):
    """A final assistant text reply (finish_reason == "stop")."""
    return _resp(SimpleNamespace(role="assistant", content=text, tool_calls=None), "stop")


def tool_resp(*calls, content=None):
    """An assistant reply requesting tool calls.

    Each call is ``(name, arguments)`` where arguments is a dict (serialised to
    JSON) or a raw string (to simulate malformed JSON from the model).
    """
    tool_calls = []
    for i, (name, args) in enumerate(calls):
        raw = args if isinstance(args, str) else json.dumps(args)
        tool_calls.append(SimpleNamespace(
            id=f"call_{i}", type="function",
            function=SimpleNamespace(name=name, arguments=raw)))
    return _resp(SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls), "tool_calls")


class FakeClient:
    """Mimics ``openai.OpenAI`` for ``client.chat.completions.create``."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        # Snapshot the messages: the agent keeps mutating the same list.
        self.calls.append({**kwargs, "messages": json.loads(json.dumps(kwargs["messages"], default=str))})
        if not self._responses:
            raise RuntimeError("FakeClient ran out of scripted responses")
        r = self._responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
