"""Agent 主循环：直接回复 / 工具调用 / 多步 / 最大步数 / 各类异常。"""
import json

from conftest import fa, tc
from mini_agent import LLMError


def test_direct_reply_without_tools(make_agent):
    agent, llm = make_agent([fa("你好！有什么可以帮你？")])
    r = agent.chat("A", "w1", "你好")
    assert r.answer == "你好！有什么可以帮你？" and r.stop_reason == "final"
    assert r.tool_calls == [] and len(llm.calls) == 1


def test_tool_call_then_final(make_agent):
    agent, llm = make_agent([tc("calculator", expression="(1234+5678)*3"), fa("结果是 20736")])
    r = agent.chat("A", "w1", "(1234+5678)*3 等于多少")
    assert r.answer == "结果是 20736"
    assert r.tool_calls == [{"name": "calculator", "arguments": {"expression": "(1234+5678)*3"}, "ok": True}]
    # 第二次调用 LLM 时，最后一条消息就是工具结果
    last = llm.calls[1][-1]
    assert last["role"] == "user" and "<observation" in last["content"] and "20736" in last["content"]


def test_multi_step_weather_then_todo(make_agent):
    agent, llm = make_agent([
        tc("weather", city="北京"),
        tc("todo", action="add", content="明天带伞"),
        fa("北京今天的天气已查好，并记下了待办：明天带伞"),
    ])
    r = agent.chat("A", "w1", "查下北京天气，再记个待办明天带伞")
    assert [c["name"] for c in r.tool_calls] == ["weather", "todo"]
    s = agent.sessions.get("A", "w1")
    assert s.state["todos"][0]["content"] == "明天带伞"


def test_parallel_tool_calls_in_one_step(make_agent):
    both = tc("weather", city="北京") + '\n<tool_call>{"name":"weather","arguments":{"city":"上海"}}</tool_call>'
    agent, llm = make_agent([both, fa("北京和上海的天气如上")])
    r = agent.chat("A", "w1", "北京和上海天气")
    assert r.steps == 2 and len(r.tool_calls) == 2
    assert llm.calls[1][-1]["content"].count("<observation") == 2


def test_max_steps_forces_final(make_agent):
    calls = [tc("search", query=f"q{i}") for i in range(3)]
    agent, llm = make_agent(calls + [fa("根据已查到的信息，答案是……")], max_steps=3)
    r = agent.chat("A", "w1", "研究一下")
    assert r.stop_reason == "max_steps" and r.answer.startswith("根据已查到")
    assert len(llm.calls) == 4
    assert "最大工具调用步数" in llm.calls[-1][-1]["content"]


def test_max_steps_fallback_when_model_still_calls_tools(make_agent):
    agent, _ = make_agent([tc("search", query=f"q{i}") for i in range(3)], max_steps=2)
    r = agent.chat("A", "w1", "研究一下")
    assert r.stop_reason == "max_steps" and "超出了单轮上限" in r.answer


def test_tool_error_is_fed_back_and_model_recovers(make_agent):
    agent, llm = make_agent([tc("weather", city="火星"), tc("weather", city="北京"), fa("北京天气如上")])
    r = agent.chat("A", "w1", "查天气")
    assert [c["ok"] for c in r.tool_calls] == [False, True]
    assert 'status="error"' in llm.calls[1][-1]["content"] and "暂不支持城市" in llm.calls[1][-1]["content"]


def test_unknown_tool_and_bad_args_are_fed_back(make_agent):
    agent, llm = make_agent([tc("translate", text="hi"), tc("todo", action="add"), fa("好的")])
    r = agent.chat("A", "w1", "x")
    assert r.answer == "好的"
    assert "不存在" in llm.calls[1][-1]["content"]
    assert "需要 content" in llm.calls[2][-1]["content"]


def test_format_error_then_recover(make_agent):
    agent, llm = make_agent(['<tool_call>{"name": weather}</tool_call>', fa("好了")])
    r = agent.chat("A", "w1", "x")
    assert r.answer == "好了" and r.stop_reason == "final"
    assert "不符合协议" in llm.calls[1][-1]["content"]


def test_format_error_twice_degrades(make_agent):
    agent, _ = make_agent(["<thought>嗯</thought>", "<thought>嗯嗯</thought>"])
    r = agent.chat("A", "w1", "x")
    assert r.stop_reason == "format_error" and "换个说法" in r.answer


def test_llm_error_is_handled_and_turn_excluded_from_context(make_agent):
    agent, llm = make_agent([LLMError("HTTP 500", retryable=True), fa("第二次正常")])
    r = agent.chat("A", "w1", "第一句")
    assert r.stop_reason == "llm_error" and "暂时不可用" in r.answer
    agent.chat("A", "w1", "第二句")
    contents = [m["content"] for m in llm.calls[1]]
    assert "第一句" not in contents  # 失败的轮次不进入 context
    assert len(agent.sessions.get("A", "w1").turns) == 2  # 但保留在 session 里便于审计


def test_duplicate_call_is_reused(make_agent):
    agent, llm = make_agent([tc("weather", city="北京"), tc("weather", city="北京"), fa("ok")])
    agent.chat("A", "w1", "x")
    assert "已复用结果" in llm.calls[2][-1]["content"]


def test_empty_input(make_agent):
    agent, llm = make_agent([])
    assert agent.chat("A", "w1", "   ").stop_reason == "empty_input" and not llm.calls


def test_trace_log_written(make_agent, tmp_path):
    agent, _ = make_agent([tc("calculator", expression="1+1"), fa("2")])
    r = agent.chat("A", "w1", "1+1")
    path = tmp_path / "traces" / "A" / "w1.jsonl"
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    names = [e["event"] for e in events]
    assert names == ["user_input", "llm_request", "llm_response", "tool_call", "tool_result",
                     "llm_request", "llm_response", "final"]
    assert all(e["run_id"] == r.run_id for e in events)
    tool_result = next(e for e in events if e["event"] == "tool_result")
    assert tool_result["ok"] and "duration_ms" in tool_result
    assert events[2]["thought"] == "需要调用工具"
