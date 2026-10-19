import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mini_agent import Agent, AgentConfig, LLMResponse, ScriptedLLM  # noqa: E402
from mini_agent.context import SUMMARY_PROMPT  # noqa: E402


class RoutingLLM(ScriptedLLM):
    """摘要请求单独应答（不消耗脚本），其余请求按脚本顺序应答。"""

    def __init__(self, script, summary="摘要", summary_error=None):
        super().__init__(script)
        self.summary = summary
        self.summary_error = summary_error
        self.summary_calls = []

    def chat(self, messages, **kwargs):
        if messages[0]["content"].startswith(SUMMARY_PROMPT[:8]):
            self.summary_calls.append(messages)
            if self.summary_error:
                raise self.summary_error
            return LLMResponse(content=self.summary)
        return super().chat(messages, **kwargs)


def tc(name: str, thought: str = "需要调用工具", **arguments) -> str:
    """拼一条标准的 tool_call 回复。"""
    import json
    return f"<thought>{thought}</thought>\n<tool_call>{json.dumps({'name': name, 'arguments': arguments}, ensure_ascii=False)}</tool_call>"


def fa(text: str, thought: str = "可以回答了") -> str:
    return f"<thought>{thought}</thought>\n<final_answer>{text}</final_answer>"


@pytest.fixture
def make_agent(tmp_path):
    def _make(script, llm=None, **cfg_overrides):
        cfg = AgentConfig(data_dir=tmp_path, **cfg_overrides)
        llm = llm or ScriptedLLM(script)
        return Agent(llm, config=cfg), llm
    return _make


def pytest_configure(config):
    config.addinivalue_line("markers", "live: 需要真实 LLM API（设置 LLM_API_KEY 后运行）")
