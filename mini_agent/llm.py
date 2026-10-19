"""LLM 客户端。

只依赖 OpenAI-compatible 的 /chat/completions 文本接口，不使用厂商的 function calling，
工具调用协议由 protocol.py 自己定义和解析 —— 这样 DeepSeek / GLM / 通义 / Kimi / 豆包都能直接换。
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
    reasoning: str = ""             # 推理模型的 reasoning_content（如 deepseek-reasoner）
    usage: dict = field(default_factory=dict)
    latency_ms: int = 0
    model: str = ""


class BaseLLM:
    model: str = "base"

    def chat(self, messages: list[dict], **kwargs: Any) -> LLMResponse:  # pragma: no cover
        raise NotImplementedError


class OpenAICompatLLM(BaseLLM):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        max_retries: int = 3,
        temperature: float = 0.2,
        backoff_base: float = 1.0,
    ):
        if not api_key:
            raise LLMError("缺少 API Key：请在 .env 中设置 LLM_API_KEY")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.temperature = temperature
        self.backoff_base = backoff_base

    @classmethod
    def from_env(cls, **kwargs: Any) -> "OpenAICompatLLM":
        return cls(
            base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com/v1"),
            api_key=os.environ.get("LLM_API_KEY", ""),
            model=os.environ.get("LLM_MODEL", "deepseek-chat"),
            **kwargs,
        )

    def chat(self, messages: list[dict], stop: list[str] | None = None, **kwargs: Any) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.temperature),
        }
        if stop:
            payload["stop"] = stop
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
        req = urllib.request.Request(
            self.url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            retryable = e.code == 429 or e.code >= 500
            raise LLMError(f"HTTP {e.code}: {detail}", retryable=retryable, status=e.code) from e
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            raise LLMError(f"网络错误: {e}", retryable=True) from e
        latency = int((time.perf_counter() - t0) * 1000)
        try:
            data = json.loads(raw)
            msg = data["choices"][0]["message"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            raise LLMError(f"响应格式异常: {raw[:200]}", retryable=True) from e
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
        if not content.strip() and not reasoning.strip():
            raise LLMError("LLM 返回空内容", retryable=True)
        return LLMResponse(
            content=content,
            reasoning=reasoning,
            usage=data.get("usage") or {},
            latency_ms=latency,
            model=data.get("model", self.model),
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
