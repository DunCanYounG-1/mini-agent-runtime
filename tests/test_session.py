"""Session：隔离、持久化恢复、并发与忙状态、非法 id。"""
import threading
import time

import pytest

from conftest import fa, tc
from mini_agent import Agent, AgentConfig, ScriptedLLM, SessionBusyError


def test_two_windows_are_isolated(make_agent):
    # 窗口 1：查天气记待办；窗口 2：写周报记待办；再回窗口 1 追问
    agent, llm = make_agent([
        tc("weather", city="北京"), tc("todo", action="add", content="明天带伞"), fa("已查天气并记下带伞"),
        tc("todo", action="add", content="周五交周报"), fa("周报提纲……已记待办"),
        fa("你在这个窗口记的待办是：明天带伞"),
    ])
    agent.chat("A", "w1", "查北京天气并记待办：明天带伞")
    agent.chat("A", "w2", "写周报并记待办：周五交周报")
    agent.chat("A", "w1", "我记了哪些待办？")

    w1, w2 = agent.sessions.get("A", "w1"), agent.sessions.get("A", "w2")
    assert [t["content"] for t in w1.state["todos"]] == ["明天带伞"]
    assert [t["content"] for t in w2.state["todos"]] == ["周五交周报"]
    assert len(w1.turns) == 2 and len(w2.turns) == 1

    # 只看 system 之外的历史（system 里的工具描述本身就含“周报”“北京”等词）
    w1_ctx = "\n".join(m["content"] for m in llm.calls[-1][1:])
    w1_sys = llm.calls[-1][0]["content"]
    assert "明天带伞" in w1_ctx and "周报" not in w1_ctx  # 窗口 2 的内容不会串到窗口 1
    assert "#1 明天带伞" in w1_sys and "周五交周报" not in w1_sys
    w2_ctx = "\n".join(m["content"] for m in llm.calls[3][1:])
    assert "北京" not in w2_ctx and "带伞" not in w2_ctx


def test_same_session_id_different_users_isolated(make_agent):
    agent, _ = make_agent([tc("todo", action="add", content="A 的事"), fa("ok"), fa("你还没有待办")])
    agent.chat("A", "w1", "记 A 的事")
    agent.chat("B", "w1", "我有什么待办")
    assert agent.sessions.get("B", "w1").state.get("todos", []) == []


def test_persist_and_resume_after_restart(tmp_path):
    cfg = AgentConfig(data_dir=tmp_path)
    a1 = Agent(ScriptedLLM([tc("todo", action="add", content="带伞"), fa("记好了")]), config=cfg)
    a1.chat("A", "w1", "记个待办带伞")

    llm2 = ScriptedLLM([fa("你的待办是带伞")])
    a2 = Agent(llm2, config=cfg)  # 模拟进程重启：全新的 Agent 与缓存
    a2.chat("A", "w1", "我有什么待办？")
    system = llm2.calls[0][0]["content"]
    assert "#1 带伞" in system  # 待办状态从磁盘恢复并注入
    assert any("记个待办带伞" in m["content"] for m in llm2.calls[0])  # 历史也恢复了
    assert [s["session_id"] for s in a2.sessions.list("A")] == ["w1"]


def test_busy_session_rejects_concurrent_message(make_agent):
    agent, _ = make_agent([])
    s = agent.sessions.get("A", "w1")
    with agent.sessions.acquire(s):
        with pytest.raises(SessionBusyError):
            agent.chat("A", "w1", "又来一条")


def test_different_sessions_run_concurrently(tmp_path):
    cfg = AgentConfig(data_dir=tmp_path)

    def slow_reply(msgs):
        time.sleep(0.3)
        return fa("ok")

    agent = Agent(ScriptedLLM([slow_reply, slow_reply]), config=cfg)
    t0 = time.perf_counter()
    threads = [threading.Thread(target=agent.chat, args=("A", sid, "hi")) for sid in ("w1", "w2")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert time.perf_counter() - t0 < 0.55  # 并行而不是串行（串行需要 ≥0.6s）
    assert len(agent.sessions.get("A", "w1").turns) == 1 and len(agent.sessions.get("A", "w2").turns) == 1


@pytest.mark.parametrize("bad", ["../etc", "a/b", "", "x" * 65])
def test_invalid_ids_rejected(make_agent, bad):
    agent, _ = make_agent([])
    with pytest.raises(ValueError):
        agent.sessions.get("A", bad)


def test_corrupt_session_file_is_backed_up(make_agent, tmp_path):
    agent, _ = make_agent([])
    p = agent.sessions.store.path("A", "w1")
    p.parent.mkdir(parents=True)
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError):
        agent.sessions.get("A", "w1")
    assert list(p.parent.glob("w1.corrupt.*"))
