"""Context 组装、压缩与 memory 召回。

每次调用 LLM 时的 messages 布局：

  [system]  角色 + 输出协议 + 工具 Schema
            + 当前时间
            + 【实时状态】待办快照（从 session.state 现读，是事实来源）
            + 【早前对话摘要】session.summary（压缩产物）
            + 【相关历史片段】从 archive 召回的原文（仅在命中时出现）
  [history] 活跃轮次：user → (assistant tool_call → user observation)* → assistant final_answer
            · 思考过程不进历史（它只对当轮推理有用，留着浪费 token 还会诱导模型复读旧思路）
            · 工具调用 + 结果保留（带工具的追问要靠它补全参数），但旧轮次的结果截短
            · 失败的轮次（LLM 报错）不进 context
  [current] 本轮 user 输入 + 本轮已产生的 thought/tool_call/observation（思考保留）

压缩时机：组装前估算 token，超过 max_context_tokens * compress_threshold，
或活跃轮数超过 max_history_turns，就把最早的轮次（保留最近 keep_recent_turns 轮）
用 LLM 合并进滚动摘要，原文移入 archive。LLM 摘要失败时退化为规则摘要，保证不阻塞对话。
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import TYPE_CHECKING

from .protocol import ToolCall, render_assistant, render_observation
from .session import Session, Turn
from .tools.todo import render_todos

if TYPE_CHECKING:
    from .config import AgentConfig
    from .llm import BaseLLM
    from .tools import ToolRegistry
    from .trace import RunTrace

SYSTEM_TEMPLATE = """你是一个可以调用工具的中文智能助手。

## 输出协议（必须严格遵守）
每次回复先用 <thought> 写简短思考，然后二选一：

A. 需要工具：
<thought>为什么需要这个工具、参数怎么定</thought>
<tool_call>{{"name": "工具名", "arguments": {{参数 JSON}}}}</tool_call>
可以连续写多个 <tool_call>。写完立即停止，等待系统返回 <observation>，绝不能自己编造工具结果。

B. 可以回答：
<thought>简短思考</thought>
<final_answer>给用户的最终回答</final_answer>

## 决策规则
- 闲聊、常识、对之前回答的追问（如“展开说说第二点”），直接 final_answer，不调用工具。
- 天气、精确计算、待办的增删查、需要查资料时，必须调用对应工具，不要凭记忆编造。
- 用户省略了信息的追问（如查完北京天气后问“那上海呢”“明天呢”），结合上文补全参数后再调用工具。
- 用户一句话里有多个需求（如“查天气并记个待办”），逐个完成后再给 final_answer。
- 工具返回 error 时，根据错误信息修正参数重试一次；仍失败就如实告诉用户。
- arguments 必须严格符合参数 Schema。

## 可用工具
{tools}

## 运行环境
当前时间：{now}（{weekday}）
用户：{user_id}　会话：{session_id}

## 当前会话状态（实时，以此为准）
待办事项：
{todos}
{summary_block}{recall_block}"""

SUMMARY_PROMPT = """你负责压缩一段对话历史。请把【已有摘要】和【新增对话】合并成一份新的摘要。
要求：
- 保留：用户的身份/偏好/约束、已确认的事实与结论、工具查询得到的关键数据（城市、日期、数值）、用户提过但尚未完成的事项。
- 丢弃：寒暄、重复内容、推理过程。
- 用要点列表，不超过 {limit} 字。只输出摘要本身。

【已有摘要】
{old}

【新增对话】
{dialog}"""

_WEEKDAYS = "一二三四五六日"


def estimate_tokens(text: str) -> int:
    """粗略估算：CJK 字符约 1 token/字，其余约 4 字符/token。够用来做阈值判断。"""
    cjk = len(re.findall(r"[一-鿿　-〿＀-￯]", text))
    return cjk + (len(text) - cjk) // 4 + 1


def messages_tokens(messages: list[dict]) -> int:
    return sum(estimate_tokens(m["content"]) + 4 for m in messages)


def _bigrams(text: str) -> set[str]:
    t = re.sub(r"\s+", "", text.lower())
    return {t[i : i + 2] for i in range(len(t) - 1)}


def _clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[:n] + "…"


class ContextManager:
    def __init__(self, config: "AgentConfig", registry: "ToolRegistry", llm: "BaseLLM"):
        self.cfg = config
        self.registry = registry
        self.llm = llm

    # ---------- 组装 ----------
    def render_system(self, session: Session, recalled: list[str] | None = None, now: datetime | None = None) -> str:
        now = now or datetime.now()
        summary_block = f"\n## 早前对话摘要（更早的对话已压缩）\n{session.summary}\n" if session.summary else ""
        recall_block = ""
        if recalled:
            body = "\n".join(f"- {r}" for r in recalled)
            recall_block = f"\n## 可能相关的更早对话片段（召回自归档，仅供参考）\n{body}\n"
        return SYSTEM_TEMPLATE.format(
            tools=self.registry.render_for_prompt(),
            now=now.strftime("%Y-%m-%d %H:%M"),
            weekday="星期" + _WEEKDAYS[now.weekday()],
            user_id=session.user_id,
            session_id=session.session_id,
            todos=render_todos(session.state.get("todos", [])),
            summary_block=summary_block,
            recall_block=recall_block,
        )

    def render_turn(self, turn: Turn, obs_max_chars: int) -> list[dict]:
        msgs = [{"role": "user", "content": turn.user}]
        for step in turn.steps:
            if not step.tool_calls:
                continue
            calls = [ToolCall(tc.name, tc.arguments) for tc in step.tool_calls]
            msgs.append({"role": "assistant", "content": render_assistant("", calls)})
            obs = "\n".join(render_observation(tc.name, tc.ok, tc.result, obs_max_chars) for tc in step.tool_calls)
            msgs.append({"role": "user", "content": obs})
        msgs.append({"role": "assistant", "content": render_assistant("", final=turn.final)})
        return msgs

    def render_history(self, session: Session) -> list[dict]:
        turns = [t for t in session.turns if t.ok]
        msgs: list[dict] = []
        for i, t in enumerate(turns):
            recent = i >= len(turns) - 2  # 最近两轮保留较完整的工具结果，方便追问
            limit = self.cfg.tool_result_max_chars if recent else self.cfg.old_observation_max_chars
            msgs.extend(self.render_turn(t, limit))
        return msgs

    def build(self, session: Session, user_input: str, scratch: list[dict], recalled: list[str] | None = None,
              system: str | None = None) -> list[dict]:
        # system 由调用方在每轮开始时生成一次并冻结：本轮内的状态变化只通过 observation 体现，
        # 否则模型会在快照里看到自己刚加的待办，误以为“原本就有”（真实模型踩过的坑）。
        return [
            {"role": "system", "content": system if system is not None else self.render_system(session, recalled)},
            *self.render_history(session),
            {"role": "user", "content": user_input},
            *scratch,
        ]

    # ---------- 压缩 ----------
    def needs_compress(self, session: Session, user_input: str) -> bool:
        if len(session.turns) > self.cfg.max_history_turns:
            return True
        tokens = messages_tokens(self.build(session, user_input, []))
        return tokens > self.cfg.max_context_tokens * self.cfg.compress_threshold

    def maybe_compress(self, session: Session, user_input: str, trace: "RunTrace | None" = None) -> bool:
        if not self.needs_compress(session, user_input) or len(session.turns) <= self.cfg.keep_recent_turns:
            return False
        before = messages_tokens(self.build(session, user_input, []))
        cut = len(session.turns) - self.cfg.keep_recent_turns
        old, session.turns = session.turns[:cut], session.turns[cut:]
        method = "llm"
        try:
            session.summary = self._llm_summary(session.summary, old)
        except Exception as e:  # 摘要失败不能影响对话
            method = f"rule ({type(e).__name__})"
            session.summary = self._rule_summary(session.summary, old)
        session.archive.extend(old)
        session.compress_count += 1
        after = messages_tokens(self.build(session, user_input, []))
        if trace:
            trace.log("context_compressed", method=method, turns_compressed=len(old),
                      tokens_before=before, tokens_after=after, summary=session.summary)
        return True

    def _dialog_text(self, turns: list[Turn]) -> str:
        lines = []
        for t in turns:
            lines.append(f"用户：{t.user}")
            for s in t.steps:
                for tc in s.tool_calls:
                    lines.append(f"  [工具 {tc.name}({tc.arguments}) → {_clip(tc.result, 200)}]")
            lines.append(f"助手：{_clip(t.final, 400)}")
        return "\n".join(lines)

    def _llm_summary(self, old_summary: str, turns: list[Turn]) -> str:
        prompt = SUMMARY_PROMPT.format(limit=400, old=old_summary or "（无）", dialog=self._dialog_text(turns))
        resp = self.llm.chat([{"role": "user", "content": prompt}], temperature=0)
        text = resp.content.strip()
        if not text:
            raise ValueError("空摘要")
        return _clip(text, 1200)

    def _rule_summary(self, old_summary: str, turns: list[Turn]) -> str:
        lines = [old_summary] if old_summary else []
        for t in turns:
            tools = ",".join(tc.name for s in t.steps for tc in s.tool_calls)
            tool_note = f"（用了 {tools}）" if tools else ""
            lines.append(f"- 用户问「{_clip(t.user, 60)}」{tool_note}→ {_clip(t.final, 80)}")
        return _clip("\n".join(lines), 1500)

    # ---------- 召回 ----------
    def recall(self, session: Session, query: str) -> list[str]:
        """从 archive（已被压缩出 context 的原文）里按 bigram 重叠召回相关片段。

        时机：每次用户输入、在调用 LLM 之前；只在有 archive 时才会触发。
        放置：system prompt 末尾的“相关片段”区块，而不是伪造成历史消息，避免打乱对话顺序。
        """
        if not session.archive or self.cfg.recall_top_k <= 0:
            return []
        q = _bigrams(query)
        if not q:
            return []
        scored = []
        for t in session.archive:
            doc = t.user + " " + t.final + " " + " ".join(tc.result for s in t.steps for tc in s.tool_calls)
            score = len(q & _bigrams(doc)) / len(q)
            if score >= 0.2:
                scored.append((score, t))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f"[{t.ts}] 用户：{_clip(t.user, 80)} ｜ 助手：{_clip(t.final, 160)}"
                for _, t in scored[: self.cfg.recall_top_k]]

    # ---------- 本轮内的 scratch 控制 ----------
    def shrink_scratch(self, scratch: list[dict]) -> None:
        """单轮内工具调用很多时，把较早的 observation 截短，防止当轮就爆 context。"""
        if messages_tokens(scratch) < self.cfg.max_context_tokens * 0.4:
            return
        for m in scratch[:-2]:
            if m["role"] == "user" and len(m["content"]) > self.cfg.old_observation_max_chars:
                m["content"] = _clip(m["content"], self.cfg.old_observation_max_chars) + "\n</observation>"
