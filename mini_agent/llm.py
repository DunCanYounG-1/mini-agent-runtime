"""LLM 客户端。

支持两种接口格式（LLM_PROVIDER 切换），都只用纯文本收发：
  - anthropic：Anthropic Messages API（/v1/messages）。DeepSeek 提供兼容端点 https://api.deepseek.com/anthropic
  - openai   ：OpenAI-compatible /chat/completions。DeepSeek / GLM / 通义 / Kimi / 豆包都支持

不使用任何一方的原生工具调用字段（tools / tool_use），工具调用协议由 protocol.py 自己定义和解析，
所以换接口格式只影响本文件：请求怎么拼、响应怎么拆。
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable


class LLMError(Exception):
    """LLM 调用失败。retryable 表示是否值得重试（网络、限流、5xx）。"""

    def __init__(self, message: str, retryable: bool = False, status: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


@dataclass
class LLMResponse:
    content: str
    reasoning: str = ""             # 推理内容：openai 格式的 reasoning_content / anthropic 格式的 thinking 块
    usage: dict = field(default_factory=dict)   # 统一成 prompt_tokens / completion_tokens，另保留原始字段
    latency_ms: int = 0
    model: str = ""


class BaseLLM:
    model: str = "base"

    def chat(self, messages: list[dict], **kwargs: Any) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError


class _HTTPLLM(BaseLLM):
    """两种接口共用的部分：HTTP、超时、错误分类、指数退避重试。子类只负责拼请求和拆响应。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        max_retries: int = 3,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        backoff_base: float = 1.0,
    ):
        if not api_key:
            raise LLMError("缺少 API Key：请在 .env 中设置 LLM_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.backoff_base = backoff_base

    # 子类实现
    url: str
    def headers(self) -> dict: ...  # noqa: E704
    def build_payload(self, messages: list[dict], stop: list[str] | None, temperature: float) -> dict: ...  # noqa: E704
    def parse_response(self, data: dict) -> LLMResponse: ...  # noqa: E704

    def chat(self, messages: list[dict], stop: list[str] | None = None, **kwargs: Any) -> LLMResponse:
        payload = self.build_payload(messages, stop, kwargs.get("temperature", self.temperature))
        last_err: LLMError | None = None
        for attempt in range(self.max_retries):
            try:
                return self._request(payload)
            except LLMError as e:
                last_err = e
                if not e.retryable or attempt == self.max_retries - 1:
                    raise
                time.sleep(self.backoff_base * (2 ** attempt))
        raise last_err or LLMError("unknown")  # pragma: no cover

    def _request(self, payload: dict) -> LLMResponse:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.url, data=body, method="POST",
                                     headers={"Content-Type": "application/json", **self.headers()})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            retryable = e.code == 429 or e.code >= 500   # 含 Anthropic 的 529 overloaded
            raise LLMError(f"HTTP {e.code}: {detail}", retryable=retryable, status=e.code) from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            raise LLMError(f"网络错误: {e}", retryable=True) from e
        latency = int((time.perf_counter() - t0) * 1000)
        try:
            out = self.parse_response(json.loads(raw))
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            raise LLMError(f"响应格式异常: {raw[:200]}", retryable=True) from e
        if not out.content.strip() and not out.reasoning.strip():
            raise LLMError("LLM 返回空内容", retryable=True)
        out.latency_ms = latency
        return out


class OpenAICompatLLM(_HTTPLLM):
    """POST {base}/chat/completions。system 作为一条 role=system 的消息；返回 choices[0].message.content。"""

    @property
    def url(self) -> str:
        return self.base_url + "/chat/completions"

    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def build_payload(self, messages, stop, temperature) -> dict:
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature,
                                   "max_tokens": self.max_tokens}
        if stop:
            payload["stop"] = stop
        return payload

    def parse_response(self, data: dict) -> LLMResponse:
        msg = data["choices"][0]["message"]
        return LLMResponse(content=msg.get("content") or "", reasoning=msg.get("reasoning_content") or "",
                           usage=data.get("usage") or {}, model=data.get("model", self.model))


class AnthropicMessagesLLM(_HTTPLLM):
    """POST {base}/v1/messages。与 chat/completions 的差别：

    - system 不在 messages 里，而是顶层的 system 字段；
    - messages 必须 user/assistant 严格交替、且以 user 开头，相邻同角色要合并；
    - max_tokens 必填；stop 叫 stop_sequences；
    - 返回的 content 是块数组（text / thinking …），需要按类型拼接；
    - 鉴权用 x-api-key + anthropic-version 头，而不是 Bearer。
    """

    API_VERSION = "2023-06-01"

    @property
    def url(self) -> str:
        return self.base_url + "/v1/messages"

    def headers(self) -> dict:
        return {"x-api-key": self.api_key, "anthropic-version": self.API_VERSION}

    @staticmethod
    def convert_messages(messages: list[dict]) -> tuple[str, list[dict]]:
        system_parts: list[str] = []
        out: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                system_parts.append(m["content"])
                continue
            if out and out[-1]["role"] == m["role"]:
                out[-1]["content"] += "\n\n" + m["content"]
            else:
                out.append({"role": m["role"], "content": m["content"]})
        if out and out[0]["role"] != "user":
            out.insert(0, {"role": "user", "content": "（继续）"})
        return "\n\n".join(system_parts), out

    def build_payload(self, messages, stop, temperature) -> dict:
        system, msgs = self.convert_messages(messages)
        payload: dict[str, Any] = {"model": self.model, "max_tokens": self.max_tokens, "messages": msgs,
                                   "temperature": temperature}
        if system:
            payload["system"] = system
        if stop:
            payload["stop_sequences"] = stop
        return payload

    def parse_response(self, data: dict) -> LLMResponse:
        blocks = data["content"]
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        thinking = "\n".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking")
        raw_usage = data.get("usage") or {}
        usage = {"prompt_tokens": raw_usage.get("input_tokens"),
                 "completion_tokens": raw_usage.get("output_tokens"), **raw_usage}
        return LLMResponse(content=text, reasoning=thinking, usage=usage, model=data.get("model", self.model))


_PROVIDERS = {"anthropic": AnthropicMessagesLLM, "openai": OpenAICompatLLM}
_DEFAULT_BASE = {"anthropic": "https://api.deepseek.com/anthropic", "openai": "https://api.deepseek.com/v1"}


def llm_from_env(**kwargs: Any) -> _HTTPLLM:
    """按 LLM_PROVIDER（anthropic | openai，默认 anthropic）创建客户端。"""
    provider = os.environ.get("LLM_PROVIDER", "anthropic").strip().lower()
    if provider not in _PROVIDERS:
        raise LLMError(f"未知 LLM_PROVIDER={provider!r}，可选: {list(_PROVIDERS)}")
    return _PROVIDERS[provider](
        base_url=os.environ.get("LLM_BASE_URL", _DEFAULT_BASE[provider]),
        api_key=os.environ.get("LLM_API_KEY", ""),
        model=os.environ.get("LLM_MODEL", "deepseek-chat"),
        **kwargs,
    )


class ScriptedLLM(BaseLLM):
    """测试用的假 LLM：按顺序返回预设回复，并记录每次收到的 messages，方便断言 context 内容。

    script 中的元素可以是：
      - str：直接作为回复
      - Exception：抛出（模拟网络错误等）
      - callable(messages) -> str：根据上下文动态生成回复
    """

    model = "scripted"

    def __init__(self, script: list[str | Exception | Callable[[list[dict]], str]]):
        self.script = list(script)
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        self.calls.append([dict(m) for m in messages])
        if not self.script:
            raise LLMError("ScriptedLLM 脚本已用完")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        text = item(messages) if callable(item) else item
        return LLMResponse(content=text, model=self.model)
