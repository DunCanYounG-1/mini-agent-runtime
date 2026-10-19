"""用真实 LLM 跑一遍题目里的场景，并把完整过程写成 docs/demo_transcript.md。

  python scripts/demo.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mini_agent import Agent, AgentConfig, OpenAICompatLLM, load_dotenv  # noqa: E402

SCENARIO = [
    ("w1", "帮我查一下北京今天的天气，然后记一个待办：明天出门带伞"),
    ("w2", "帮我写一份简短的周报提纲（三点即可），然后记一个待办：周五前提交周报"),
    ("w1", "那上海呢？"),
    ("w1", "明天呢？"),
    ("w2", "把提纲的第二点展开说两句"),
    ("w1", "我在这个窗口都记了哪些待办？把带伞那条标记为完成"),
    ("w2", "我在这个窗口有哪些待办？"),
    ("w1", "(1234+5678)*3 再除以 4 等于多少？"),
]


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env")
    data = ROOT / "data" / "demo"
    shutil.rmtree(data, ignore_errors=True)
    cfg = AgentConfig.from_env()
    cfg.data_dir = data
    llm = OpenAICompatLLM.from_env(timeout=90)
    agent = Agent(llm, config=cfg)

    lines = [f"# Demo transcript\n\n模型：`{llm.model}`　用户：A　窗口：w1 / w2\n"]
    for sid, text in SCENARIO:
        print(f"\n[{sid}] 用户> {text}")
        r = agent.chat("A", sid, text)
        print(f"[{sid}] Agent> {r.answer}")
        lines.append(f"\n## [{sid}] 用户：{text}\n")
        for e in r.events:
            if e["event"] == "llm_response" and e.get("thought"):
                lines.append(f"- 💭 思考（step {e['step']}，{e['latency_ms']}ms）：{e['thought'][:300]}")
            elif e["event"] == "tool_call":
                lines.append(f"- 🔧 `{e['name']}({json.dumps(e['arguments'], ensure_ascii=False)})`")
            elif e["event"] == "tool_result":
                lines.append(f"  - {'✓' if e['ok'] else '✗'} `{e['result'][:160]}`")
            elif e["event"] == "context_compressed":
                lines.append(f"- 🗜 压缩 {e['turns_compressed']} 轮：{e['tokens_before']}→{e['tokens_after']} tokens")
        lines.append(f"\n**Agent**（stop_reason=`{r.stop_reason}`）：\n\n{r.answer}\n")

    for sid in ("w1", "w2"):
        todos = agent.sessions.get("A", sid).state.get("todos", [])
        lines.append(f"\n### 窗口 {sid} 最终待办\n\n```json\n{json.dumps(todos, ensure_ascii=False, indent=1)}\n```")
    out = ROOT / "docs" / "demo_transcript.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
