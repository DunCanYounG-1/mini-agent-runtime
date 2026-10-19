"""命令行交互。

  python -m mini_agent --user A --session w1          # 进入窗口 1
  python -m mini_agent --user A --session w2 -v       # 另开终端进入窗口 2，-v 打印思考与工具调用
  python -m mini_agent --user A --session w1 -m "那明天呢"   # 单条消息模式

会话内命令：/sessions /switch <id> /new /todo /summary /trace /help /exit
"""
from __future__ import annotations

import argparse
import json
import sys

from .agent import Agent
from .config import AgentConfig, load_dotenv
from .llm import LLMError, OpenAICompatLLM
from .session import SessionBusyError
from .tools.todo import render_todos

HELP = """/sessions        列出当前用户的所有会话
/switch <id>     切换到某个会话（不存在则新建）
/new             新建一个会话
/todo            查看当前会话待办
/summary         查看当前会话的压缩摘要
/trace           查看上一轮的执行轨迹
/exit            退出"""


def _printer(verbose: bool):
    def on_event(e: dict) -> None:
        if not verbose:
            return
        ev = e["event"]
        if ev == "llm_response" and e.get("thought"):
            print(f"  \033[90m💭 {e['thought'][:300]}\033[0m")
        elif ev == "tool_call":
            print(f"  \033[36m🔧 {e['name']}({json.dumps(e['arguments'], ensure_ascii=False)})\033[0m")
        elif ev == "tool_result":
            flag = "✓" if e["ok"] else "✗"
            print(f"  \033[36m   {flag} {e['result'][:200]}\033[0m")
        elif ev == "context_compressed":
            print(f"  \033[33m🗜  context 压缩：{e['turns_compressed']} 轮 → 摘要，"
                  f"{e['tokens_before']}→{e['tokens_after']} tokens（{e['method']}）\033[0m")
        elif ev == "memory_recalled":
            print(f"  \033[33m🧠 召回 {len(e['snippets'])} 条历史片段\033[0m")
        elif ev in ("llm_error", "format_error", "max_steps_reached"):
            print(f"  \033[31m⚠ {ev}: {e.get('error') or e.get('errors') or ''}\033[0m")
    return on_event


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="mini_agent")
    p.add_argument("--user", default="A")
    p.add_argument("--session", default="default")
    p.add_argument("-m", "--message", help="单条消息模式：发送后打印回复并退出")
    p.add_argument("-v", "--verbose", action="store_true", help="打印思考过程与工具调用")
    args = p.parse_args(argv)

    load_dotenv()
    cfg = AgentConfig.from_env()
    try:
        llm = OpenAICompatLLM.from_env(timeout=cfg.llm_timeout, max_retries=cfg.llm_max_retries,
                                       temperature=cfg.temperature)
    except LLMError as e:
        print(f"[配置错误] {e}", file=sys.stderr)
        return 2
    agent = Agent(llm, config=cfg, on_event=_printer(args.verbose))
    user, sid = args.user, args.session
    last_events: list[dict] = []

    def send(text: str) -> None:
        nonlocal last_events
        try:
            r = agent.chat(user, sid, text)
        except SessionBusyError as e:
            print(f"[忙] {e}")
            return
        last_events = r.events
        print(f"Agent> {r.answer}")
        if r.stop_reason != "final":
            print(f"  （stop_reason={r.stop_reason}）")

    if args.message:
        send(args.message)
        return 0

    print(f"用户 {user} · 会话 {sid}（/help 查看命令）")
    while True:
        try:
            text = input(f"[{sid}] 你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text:
            continue
        if not text.startswith("/"):
            send(text)
            continue
        cmd, _, arg = text.partition(" ")
        session = agent.sessions.get(user, sid)
        if cmd == "/exit":
            return 0
        elif cmd == "/help":
            print(HELP)
        elif cmd == "/sessions":
            for s in agent.sessions.list(user):
                mark = "*" if s["session_id"] == sid else " "
                print(f" {mark} {s['session_id']:<16} {s['turns']:>3} 轮  {s['title']}")
        elif cmd in ("/switch", "/new"):
            sid = arg.strip() if cmd == "/switch" and arg.strip() else agent.sessions.new_session_id()
            try:
                agent.sessions.get(user, sid)
            except ValueError as e:
                print(e)
                continue
            print(f"已切换到会话 {sid}")
        elif cmd == "/todo":
            print(render_todos(session.state.get("todos", [])))
        elif cmd == "/summary":
            print(session.summary or "（还没有触发过压缩）")
        elif cmd == "/trace":
            for e in last_events:
                rest = {k: v for k, v in e.items() if k not in ("ts", "run_id", "user_id", "session_id", "event", "raw")}
                print(f"{e['elapsed_ms']:>6}ms {e['event']:<20} {json.dumps(rest, ensure_ascii=False)[:200]}")
        else:
            print("未知命令，/help 查看")


if __name__ == "__main__":
    sys.exit(main())
