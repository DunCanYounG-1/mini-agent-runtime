"""LLM 客户端：两种接口格式的请求拼装 / 响应解析 / 重试策略（不发真实请求）。"""
import io
import json
import urllib.error

import pytest

from mini_agent import AnthropicMessagesLLM, LLMError, OpenAICompatLLM, llm_from_env


def _anthropic():
    return AnthropicMessagesLLM("https://api.deepseek.com/anthropic/", "k", "deepseek-chat", backoff_base=0)


def test_anthropic_payload_moves_system_and_merges_same_roles():
    llm = _anthropic()
    msgs = [
        {"role": "system", "content": "你是助手"},
        {"role": "user", "content": "查天气"},
        {"role": "assistant", "content": "<tool_call>{}</tool_call>"},
        {"role": "user", "content": "<observation>晴</observation>"},
        {"role": "user", "content": "<observation>格式提示</observation>"},
    ]
    p = llm.build_payload(msgs, ["<observation"], 0.2)
    assert p["system"] == "你是助手"
    assert [m["role"] for m in p["messages"]] == ["user", "assistant", "user"]   # 严格交替
    assert "晴" in p["messages"][2]["content"] and "格式提示" in p["messages"][2]["content"]
    assert p["stop_sequences"] == ["<observation"] and p["max_tokens"] > 0 and "stop" not in p
    assert llm.url == "https://api.deepseek.com/anthropic/v1/messages"
    assert llm.headers() == {"x-api-key": "k", "anthropic-version": "2023-06-01"}
    assert msgs[3]["content"] == "<observation>晴</observation>"  # 不改动调用方的 messages


def test_anthropic_payload_must_start_with_user():
    _, out = AnthropicMessagesLLM.convert_messages([{"role": "system", "content": "s"},
                                                    {"role": "assistant", "content": "a"}])
    assert out[0]["role"] == "user"


def test_anthropic_parse_text_and_thinking_blocks():
    r = _anthropic().parse_response({
        "model": "deepseek-chat",
        "content": [{"type": "thinking", "thinking": "想一想"}, {"type": "text", "text": "<final_answer>"},
                    {"type": "text", "text": "好</final_answer>"}],
        "usage": {"input_tokens": 10, "output_tokens": 3},
    })
    assert r.content == "<final_answer>好</final_answer>" and r.reasoning == "想一想"
    assert r.usage["prompt_tokens"] == 10 and r.usage["completion_tokens"] == 3


def test_openai_payload_and_parse():
    llm = OpenAICompatLLM("https://api.deepseek.com/v1", "k", "deepseek-chat")
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    p = llm.build_payload(msgs, ["<observation"], 0)
    assert p["messages"] == msgs and p["stop"] == ["<observation"]
    assert llm.url.endswith("/chat/completions") and llm.headers()["Authorization"] == "Bearer k"
    r = llm.parse_response({"choices": [{"message": {"content": "hi", "reasoning_content": "r"}}], "usage": {}})
    assert r.content == "hi" and r.reasoning == "r"


class _FakeHTTPError(urllib.error.HTTPError):
    def __init__(self, code):
        super().__init__("u", code, "err", {}, io.BytesIO(b'{"error": "x"}'))


def _patch_urlopen(monkeypatch, outcomes):
    calls = []

    def fake(req, timeout):
        calls.append(json.loads(req.data))
        item = outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return io.BytesIO(json.dumps(item).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake)
    return calls


def test_retry_on_overloaded_then_success(monkeypatch):
    ok = {"content": [{"type": "text", "text": "好"}], "usage": {}}
    calls = _patch_urlopen(monkeypatch, [_FakeHTTPError(529), urllib.error.URLError("reset"), ok])
    r = _anthropic().chat([{"role": "user", "content": "hi"}])
    assert r.content == "好" and len(calls) == 3


def test_no_retry_on_auth_error(monkeypatch):
    calls = _patch_urlopen(monkeypatch, [_FakeHTTPError(401)])
    with pytest.raises(LLMError) as e:
        _anthropic().chat([{"role": "user", "content": "hi"}])
    assert e.value.status == 401 and not e.value.retryable and len(calls) == 1


def test_empty_content_is_error(monkeypatch):
    _patch_urlopen(monkeypatch, [{"content": [], "usage": {}}] * 3)
    with pytest.raises(LLMError, match="空内容"):
        _anthropic().chat([{"role": "user", "content": "hi"}])


def test_llm_from_env_selects_provider(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert isinstance(llm_from_env(), AnthropicMessagesLLM)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert isinstance(llm_from_env(), OpenAICompatLLM)
    monkeypatch.setenv("LLM_PROVIDER", "foo")
    with pytest.raises(LLMError):
        llm_from_env()
