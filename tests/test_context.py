"""Context：多轮记忆、纯对话追问、带工具追问、压缩、召回。"""
from conftest import RoutingLLM, fa, tc
from mini_agent import LLMError


def _all(messages):
    return "\n".join(m["content"] for m in messages)


def test_pure_dialog_follow_up_sees_previous_answer(make_agent):
    agent, llm = make_agent([fa("周报三要点：1.完成事项 2.风险 3.下周计划"), fa("第二点“风险”是指……")])
    agent.chat("A", "w2", "周报怎么写")
    r = agent.chat("A", "w2", "第二点展开说说")
    assert r.tool_calls == []
    msgs = llm.calls[1]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert "2.风险" in msgs[2]["content"] and msgs[3]["content"] == "第二点展开说说"


def test_tool_follow_up_sees_previous_tool_call_and_result(make_agent):
    agent, llm = make_agent([
        tc("weather", city="北京"), fa("北京今天晴"),
        tc("weather", city="上海"), fa("上海今天多云"),
    ])
    agent.chat("A", "w1", "北京天气怎么样")
    r = agent.chat("A", "w1", "那上海呢")
    assert r.tool_calls[0]["arguments"] == {"city": "上海"}
    history = _all(llm.calls[2])
    # 上一轮的工具调用与结果都在 context 中，模型才能理解“那上海呢”要查的是天气
    assert '"name": "weather"' in history and '"city": "北京"' in history and "temp_high_c" in history


def test_previous_thoughts_not_in_history_but_current_ones_are(make_agent):
    agent, llm = make_agent([
        tc("weather", thought="旧思考XYZ", city="北京"), fa("晴", thought="旧结论思考"),
        tc("calculator", thought="当前思考ABC", expression="1+1"), fa("2"),
    ])
    agent.chat("A", "w1", "北京天气")
    agent.chat("A", "w1", "1+1")
    ctx = _all(llm.calls[3])
    assert "旧思考XYZ" not in ctx and "旧结论思考" not in ctx
    assert "当前思考ABC" in ctx


def test_todo_snapshot_injected_into_system_prompt(make_agent):
    agent, llm = make_agent([tc("todo", action="add", content="买牛奶"), fa("ok"), fa("你有一个待办：买牛奶")])
    agent.chat("A", "w1", "记一下买牛奶")
    agent.chat("A", "w1", "我还有啥没做")
    assert "- [ ] #1 买牛奶" in llm.calls[2][0]["content"]


def test_system_prompt_frozen_within_turn(make_agent):
    # 回归：真实 DeepSeek 在同一轮里加完待办再 list，看到实时快照里已有该条，误答“原本就有，没有重复添加”
    agent, llm = make_agent([tc("todo", action="add", content="提交周报"), tc("todo", action="list"), fa("ok"),
                             fa("你有一个待办")])
    agent.chat("A", "w2", "记个待办：提交周报")
    systems = [c[0]["content"] for c in llm.calls]
    assert systems[0] == systems[1] == systems[2] and "暂无待办" in systems[0]
    agent.chat("A", "w2", "我有哪些待办")
    assert "#1 提交周报" in llm.calls[3][0]["content"]  # 下一轮快照才刷新


def test_old_observations_truncated(make_agent):
    script = []
    for i in range(3):
        script += [tc("search", query="上下文压缩"), fa(f"答{i}")]
    script.append(fa("end"))
    agent, llm = make_agent(script, old_observation_max_chars=50)
    for i in range(3):
        agent.chat("A", "w1", f"问{i}")
    agent.chat("A", "w1", "最后")
    msgs = llm.calls[-1]
    observations = [m["content"] for m in msgs if m["content"].startswith("<observation")]
    assert "已截断" in observations[0]           # 最早一轮的工具结果被截短
    assert len(observations[-1]) > len(observations[0])  # 最近轮保留更多


def test_compression_triggered_by_token_budget(make_agent):
    long = "很长的回答" * 60
    llm = RoutingLLM([fa(f"{long}{i}") for i in range(6)] + [fa("我们之前聊了 6 个问题")],
                     summary="- 用户问过 0~5 号问题，助手都做了长回答")
    agent, _ = make_agent(None, llm=llm, max_context_tokens=1200, keep_recent_turns=2)
    for i in range(6):
        agent.chat("A", "w1", f"问题{i}")
    r = agent.chat("A", "w1", "我们之前聊了什么？")
    s = agent.sessions.get("A", "w1")
    assert s.compress_count >= 1 and llm.summary_calls
    assert len(s.turns) == 3 and len(s.archive) == 4  # 保留最近 2 轮 + 本轮，其余归档
    assert s.summary == "- 用户问过 0~5 号问题，助手都做了长回答"
    ev = [e for e in agent.tracer.base.joinpath("A", "w1.jsonl").read_text(encoding="utf-8").splitlines()
          if '"context_compressed"' in e]
    assert ev
    system = llm.calls[-1][0]["content"]
    assert "早前对话摘要" in system and "用户问过 0~5" in system
    assert "问题0" in llm.summary_calls[0][0]["content"]  # 摘要请求里包含被压缩的原文
    assert "问题0" not in _all(llm.calls[-1][1:])           # 原文已不在活跃历史中
    assert r.answer == "我们之前聊了 6 个问题"


def test_compression_reduces_tokens(make_agent):
    llm = RoutingLLM([fa("回答" * 300) for _ in range(4)] + [fa("ok")], summary="- 简短摘要")
    agent, _ = make_agent(None, llm=llm, max_context_tokens=1400, keep_recent_turns=1)
    results = [agent.chat("A", "w1", f"问题{i}") for i in range(5)]
    ev = next(e for r in results for e in r.events if e["event"] == "context_compressed")
    assert ev["method"] == "llm" and ev["tokens_after"] < ev["tokens_before"]


def test_compression_falls_back_to_rules_when_summary_llm_fails(make_agent):
    llm = RoutingLLM([fa("回答" * 200) for _ in range(4)] + [fa("ok")], summary_error=LLMError("timeout"))
    agent, _ = make_agent(None, llm=llm, max_context_tokens=900, keep_recent_turns=1)
    results = [agent.chat("A", "w1", f"问题{i}") for i in range(5)]
    s = agent.sessions.get("A", "w1")
    ev = next(e for r in results for e in r.events if e["event"] == "context_compressed")
    assert ev["method"].startswith("rule") and "用户问「问题0」" in s.summary
    assert results[-1].answer == "ok"  # 摘要失败不影响正常回答


def test_max_history_turns_triggers_compression(make_agent):
    llm = RoutingLLM([fa(f"答{i}") for i in range(5)])
    agent, _ = make_agent(None, llm=llm, max_history_turns=3, keep_recent_turns=2)
    for i in range(4):
        agent.chat("A", "w1", f"问{i}")
    r = agent.chat("A", "w1", "再问")
    assert r.compressed and len(agent.sessions.get("A", "w1").archive) == 2


def test_recall_from_archive(make_agent):
    script = [tc("weather", city="厦门"), fa("厦门今天晴，最高 25 度")]
    script += [fa(f"闲聊{i}" * 50) for i in range(4)] + [fa("之前查过，厦门晴")]
    llm = RoutingLLM(script, summary="- 用户聊过天气和一些闲聊")
    agent, _ = make_agent(None, llm=llm, max_context_tokens=1000, keep_recent_turns=1)
    agent.chat("A", "w1", "厦门天气怎么样")
    for i in range(4):
        agent.chat("A", "w1", f"随便聊聊{i}")
    r = agent.chat("A", "w1", "之前查的厦门天气是多少度？")
    recalled = next(e for e in r.events if e["event"] == "memory_recalled")
    assert "厦门今天晴" in recalled["snippets"][0]
    assert "可能相关的更早对话片段" in llm.calls[-1][0]["content"]


def test_shrink_scratch_within_long_turn(make_agent):
    script = [tc("search", query=f"上下文压缩{i}") for i in range(5)] + [fa("done")]
    agent, llm = make_agent(script, max_context_tokens=600, max_steps=8, old_observation_max_chars=80)
    agent.chat("A", "w1", "多查几次")
    first_obs = [m for m in llm.calls[-1] if m["content"].startswith("<observation")][0]
    assert len(first_obs["content"]) < 120
