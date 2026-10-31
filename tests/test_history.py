from emporio.history import HistoryStore, is_user_text, window


def user(t="hi"):
    return {"role": "user", "content": t}


def assistant_calls(*ids):
    return {"role": "assistant", "content": None,
            "tool_calls": [{"id": i, "type": "function", "function": {"name": "x", "arguments": "{}"}} for i in ids]}


def tool(i):
    return {"role": "tool", "tool_call_id": i, "content": "{}"}


def final(t="ok"):
    return {"role": "assistant", "content": t}


def test_is_user_text():
    assert is_user_text(user())
    assert not is_user_text(tool("a"))
    assert not is_user_text(final())


def test_short_history_untouched():
    msgs = [user(), final()]
    assert window(msgs, 10) == msgs


def test_window_never_starts_on_tool_message():
    msgs = [user("1"), assistant_calls("a", "b"), tool("a"), tool("b"), final(), user("2"), final()]
    for size in range(1, len(msgs)):
        w = window(msgs, size)
        assert not w or is_user_text(w[0]), (size, w)


def test_window_never_splits_tool_pairs():
    msgs = [user("1"), assistant_calls("a"), tool("a"), final(), user("2"), assistant_calls("b"), tool("b"), final()]
    for size in range(1, len(msgs)):
        w = window(msgs, size)
        calls = {tc["id"] for m in w for tc in m.get("tool_calls", [])}
        replies = {m["tool_call_id"] for m in w if m["role"] == "tool"}
        assert calls == replies, (size, w)


def test_window_falls_back_to_latest_user_turn():
    msgs = [user("1")] + [assistant_calls("a"), tool("a")] * 5 + [final()]
    assert window(msgs, 3) == [] or is_user_text(window(msgs, 3)[0])
    msgs2 = [user("old"), final(), user("new")] + [assistant_calls("a"), tool("a")] * 5
    assert window(msgs2, 4)[0] == user("new")


def test_store_roundtrip_and_corruption(tmp_path):
    s = HistoryStore(tmp_path)
    assert s.load("a/b") == []
    s.save("a/b", [user("olá"), final("hi")])
    assert s.load("a/b")[0]["content"] == "olá"
    s.path("a/b").write_text("{broken")
    assert s.load("a/b") == []
