"""真实 LLM 端到端测试。需要 .env 中配置 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL。

运行：pytest -m live -v
断言只看“行为”（调了哪个工具、参数、状态），不对模型措辞做精确匹配。
"""
import os
import re

import pytest

from conftest import ROOT
from mini_agent import Agent, AgentConfig, llm_from_env, load_dotenv

load_dotenv(ROOT / ".env")
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("LLM_API_KEY"), reason="未配置 LLM_API_KEY"),
]


@pytest.fixture(scope="module")
def agent(tmp_path_factory):
    cfg = AgentConfig(data_dir=tmp_path_factory.mktemp("live"))
    return Agent(llm_from_env(timeout=90), config=cfg)


def names(r):
    return [c["name"] for c in r.tool_calls]


def test_window1_weather_and_todo(agent):
    r = agent.chat("A", "w1", "帮我查一下北京今天的天气，然后记一个待办：明天出门带伞")
    assert r.stop_reason == "final"
    assert "weather" in names(r) and "todo" in names(r)
    todos = agent.sessions.get("A", "w1").state["todos"]
    assert any("伞" in t["content"] for t in todos)


def test_window2_weekly_report_and_todo(agent):
    r = agent.chat("A", "w2", "帮我写一份简短的周报提纲（三点即可），然后记一个待办：周五前提交周报")
    assert r.stop_reason == "final" and "todo" in names(r)
    w2 = agent.sessions.get("A", "w2").state["todos"]
    w1 = agent.sessions.get("A", "w1").state["todos"]
    assert any("周报" in t["content"] for t in w2)
    assert not any("周报" in t["content"] for t in w1)  # 窗口隔离


def test_window1_tool_follow_up(agent):
    r = agent.chat("A", "w1", "那上海呢？")
    calls = [c for c in r.tool_calls if c["name"] == "weather"]
    assert calls and "上海" in calls[0]["arguments"]["city"]


def test_window1_state_follow_up(agent):
    r = agent.chat("A", "w1", "我在这个窗口都记了哪些待办？")
    assert "伞" in r.answer and "周报" not in r.answer


def test_window2_pure_dialog_follow_up(agent):
    r = agent.chat("A", "w2", "把提纲的第二点展开说两句")
    assert r.stop_reason == "final" and r.tool_calls == []


def test_calculator(agent):
    r = agent.chat("A", "w3", "(1234+5678)*3 再除以 4 等于多少？")
    assert "calculator" in names(r)
    assert "5184" in r.answer.replace(",", "")


def test_direct_reply_without_tools(agent):
    r = agent.chat("A", "w3", "用一句话介绍一下你自己")
    assert r.tool_calls == [] and r.answer


def test_compression_keeps_early_facts(tmp_path):
    cfg = AgentConfig(data_dir=tmp_path, max_context_tokens=2500, keep_recent_turns=2)
    agent = Agent(llm_from_env(timeout=90), config=cfg)
    agent.chat("A", "long", "记住：我叫小杨，在厦门工作，最喜欢的编程语言是 Rust。简单回复“好的”即可。")
    for topic in ["ReAct 是什么", "番茄工作法怎么用", "上下文压缩有哪些方法", "asyncio 的核心概念"]:
        agent.chat("A", "long", f"搜索一下：{topic}，并详细介绍")
    s = agent.sessions.get("A", "long")
    assert s.compress_count >= 1 and s.summary
    r = agent.chat("A", "long", "我叫什么、在哪个城市工作、最喜欢什么语言？")
    assert "小杨" in r.answer and "厦门" in r.answer and re.search("rust", r.answer, re.I)
