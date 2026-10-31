import json

import pytest

from emporio.agent import FALLBACK_MESSAGE, MAX_TOOL_ROUNDS, Agent
from emporio.history import HistoryStore
from emporio.prompts import SYSTEM_PROMPT
from emporio.tools import TOOL_SCHEMAS, Tools

from .conftest import FakeClient, text_resp, tool_resp


@pytest.fixture
def make(store, tmp_path):
    def _make(responses):
        client = FakeClient(responses)
        agent = Agent(client=client, model="test-model", tools=Tools(store=store, sessions_dir=tmp_path),
                      history=HistoryStore(tmp_path))
        return agent, client
    return _make


def test_schemas_are_openai_function_format():
    for s in TOOL_SCHEMAS:
        assert s["type"] == "function"
        assert {"name", "description", "parameters"} <= set(s["function"])
        assert s["function"]["parameters"]["type"] == "object"
        assert "input_schema" not in s


def test_plain_text_reply_and_request_shape(make):
    agent, client = make([text_resp("Hello!")])
    assert agent.reply("s", "oi") == "Hello!"
    call = client.calls[0]
    assert call["model"] == "test-model" and call["temperature"] == 0.2
    assert call["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert call["messages"][1] == {"role": "user", "content": "oi"}


def test_tool_roundtrip_uses_openai_messages(make):
    agent, client = make([tool_resp(("get_product", {"product": "Takamine GD20"})), text_resp("It costs R$ 2.199,00.")])
    assert "2.199,00" in agent.reply("s", "Quanto custa o Takamine GD20?")
    second = client.calls[1]["messages"]
    assistant, tool = second[-2], second[-1]
    assert assistant["role"] == "assistant" and assistant["tool_calls"][0]["id"] == "call_0"
    assert isinstance(assistant["tool_calls"][0]["function"]["arguments"], str)  # JSON string
    assert tool["role"] == "tool" and tool["tool_call_id"] == "call_0"
    assert json.loads(tool["content"])["products"][0]["final_price_brl"] == 2199.0


def test_parallel_tool_calls_all_answered(make):
    agent, client = make([
        tool_resp(("get_product", {"product": "Takamine GD20"}), ("store_info", {}),
                  ("calculate_installments", {"amount": 2199})),
        text_resp("Done.")])
    assert agent.reply("s", "hi") == "Done."
    msgs = client.calls[1]["messages"]
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_0", "call_1", "call_2"]
    assert all("error" not in json.loads(m["content"]) for m in tool_msgs)


def test_malformed_json_arguments_return_error_and_model_retries(make):
    agent, client = make([
        tool_resp(("get_product", '{"product": "Takamine GD20"')),   # truncated JSON
        tool_resp(("get_product", {"product": "Takamine GD20"})),
        text_resp("R$ 2.199,00")])
    assert agent.reply("s", "price?") == "R$ 2.199,00"
    first_tool = [m for m in client.calls[1]["messages"] if m["role"] == "tool"][0]
    assert "Invalid JSON" in json.loads(first_tool["content"])["error"]


def test_non_object_arguments_do_not_crash(make):
    agent, client = make([tool_resp(("get_product", "[1, 2]")), text_resp("ok")])
    assert agent.reply("s", "x") == "ok"


def test_unknown_tool_and_bad_args_are_reported_to_model(make):
    agent, client = make([tool_resp(("fly_to_moon", {}), ("get_product", {"wrong": 1})), text_resp("ok")])
    assert agent.reply("s", "x") == "ok"
    contents = [json.loads(m["content"]) for m in client.calls[1]["messages"] if m["role"] == "tool"]
    assert all("error" in c for c in contents)


def test_history_persists_across_turns(make):
    agent, client = make([text_resp("one"), text_resp("two")])
    agent.reply("s", "first"); agent.reply("s", "second")
    sent = client.calls[1]["messages"]
    assert [m["content"] for m in sent if m["role"] in ("user", "assistant")] == ["first", "one", "second"]


def test_max_tool_rounds_falls_back_and_rolls_back(make):
    agent, client = make([tool_resp(("store_info", {}))] * (MAX_TOOL_ROUNDS + 2))
    assert agent.reply("s", "loop") == FALLBACK_MESSAGE
    assert len(client.calls) == MAX_TOOL_ROUNDS
    assert agent.history.load("s") == []  # nothing dangling


def test_api_error_rolls_back_only_failed_turn(make):
    agent, _ = make([text_resp("fine"), RuntimeError("boom")])
    agent.reply("s", "ok turn")
    assert agent.reply("s", "bad turn") == FALLBACK_MESSAGE
    assert [m["content"] for m in agent.history.load("s")] == ["ok turn", "fine"]


def test_empty_model_response_falls_back(make):
    agent, _ = make([text_resp("")])
    assert agent.reply("s", "x") == FALLBACK_MESSAGE


def test_prompt_enforces_english_and_keeps_rules():
    p = SYSTEM_PROMPT
    assert "ALWAYS reply in English" in p
    assert "R$ 1.299,00" in p
    assert "not cumulative" in p and "PIX" in p
    assert p.index("TOOL RULES") < p.index("BUSINESS RULES")


def test_prompt_forbids_inventing_return_procedures():
    assert "ONLY the options, deadlines and conditions" in SYSTEM_PROMPT
