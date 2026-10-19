"""Agent Runtime 主循环。

  接收用户输入
    → 召回 / 必要时压缩 context
    → loop（最多 max_steps 次）：
         调 LLM → 解析出 thought / tool_calls / final_answer
         · tool_calls：执行工具，把结果作为 observation 追加，继续 loop
         · final_answer：结束，返回用户
         · 格式错误：把错误回传给 LLM 让它改，连续错太多就降级
    → 达到 max_steps：再给 LLM 一次“禁止调用工具、基于已有信息作答”的机会
    → 落盘 session，返回结果
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Callable

from .config import AgentConfig
from .context import ContextManager, messages_tokens
from .llm import BaseLLM, LLMError
from .protocol import FORMAT_ERROR_HINT, parse_output, render_assistant, render_observation
from .session import SessionManager, SessionStore, Step, ToolRecord, Turn
from .tools import ToolContext, ToolRegistry, ToolResult, default_registry
from .trace import Tracer

FORCE_FINAL_PROMPT = (
    '<observation tool="system" status="error">\n已达到本轮最大工具调用步数。'
    "请不要再调用任何工具，直接基于上面已有的信息给出 <final_answer>；信息不足的部分如实说明。\n</observation>"
)


@dataclass
class AgentResult:
    answer: str
    stop_reason: str
    run_id: str
    steps: int
    tool_calls: list[dict] = field(default_factory=list)
    compressed: bool = False
    events: list[dict] = field(default_factory=list)


class Agent:
    def __init__(
        self,
        llm: BaseLLM,
        registry: ToolRegistry | None = None,
        config: AgentConfig | None = None,
        sessions: SessionManager | None = None,
        on_event: Callable[[dict], None] | None = None,
    ):
        self.cfg = config or AgentConfig()
        self.llm = llm
        self.registry = registry or default_registry()
        self.sessions = sessions or SessionManager(SessionStore(self.cfg.data_dir))
        self.ctx = ContextManager(self.cfg, self.registry, llm)
        self.tracer = Tracer(self.cfg.data_dir, on_event)

    def chat(self, user_id: str, session_id: str, user_input: str, busy_timeout: float = 0.0) -> AgentResult:
        session = self.sessions.get(user_id, session_id)
        with self.sessions.acquire(session, timeout=busy_timeout):
            return self._run(session, user_input)

    # ------------------------------------------------------------------
    def _run(self, session, user_input: str) -> AgentResult:
        user_input = (user_input or "").strip()
        run_id = uuid.uuid4().hex[:10]
        trace = self.tracer.run(session.user_id, session.session_id, run_id)
        trace.log("user_input", text=user_input, turn_index=session.total_turns)
        if not user_input:
            return AgentResult("请输入内容。", "empty_input", run_id, 0, events=trace.events)
        if not session.title:
            session.title = user_input[:20]

        compressed = self.ctx.maybe_compress(session, user_input, trace)
        recalled = self.ctx.recall(session, user_input)
        if recalled:
            trace.log("memory_recalled", snippets=recalled)
        system = self.ctx.render_system(session, recalled)  # 整轮冻结，见 ContextManager.build

        tool_ctx = ToolContext(session=session, user_id=session.user_id, session_id=session.session_id)
        turn = Turn(user=user_input, run_id=run_id)
        scratch: list[dict] = []
        seen_calls: dict[str, str] = {}   # 同一轮内完全相同的调用直接复用结果，防止模型原地打转
        format_errors = 0
        final: str | None = None
        stop_reason = ""

        for step_no in range(1, self.cfg.max_steps + 1):
            self.ctx.shrink_scratch(scratch)
            messages = self.ctx.build(session, user_input, scratch, system=system)
            trace.log("llm_request", step=step_no, n_messages=len(messages), est_tokens=messages_tokens(messages))
            try:
                resp = self.llm.chat(messages, stop=["<observation"])
            except LLMError as e:
                trace.log("llm_error", step=step_no, error=str(e), status=e.status)
                final, stop_reason = f"抱歉，模型服务暂时不可用（{_short(str(e))}），请稍后重试。", "llm_error"
                break
            parsed = parse_output(resp.content, resp.reasoning)
            trace.log("llm_response", step=step_no, latency_ms=resp.latency_ms, usage=resp.usage,
                      kind=parsed.kind, thought=parsed.thought, raw=resp.content,
                      warnings=parsed.warnings, errors=parsed.errors)

            if parsed.kind == "tool_call":
                format_errors = 0
                step = Step(thought=parsed.thought)
                observations = []
                for call in parsed.tool_calls:
                    key = call.name + json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)
                    trace.log("tool_call", step=step_no, name=call.name, arguments=call.arguments)
                    if key in seen_calls:
                        result = ToolResult(call.name, call.arguments, True,
                                            seen_calls[key] + "\n（与本轮之前的调用完全相同，已复用结果，请不要重复调用）",
                                            error_type="duplicate")
                    else:
                        result = self.registry.execute(call.name, call.arguments, tool_ctx)
                        if result.ok:
                            seen_calls[key] = result.content
                    trace.log("tool_result", step=step_no, name=call.name, ok=result.ok,
                              duration_ms=result.duration_ms, error_type=result.error_type,
                              result=result.content[:2000])
                    step.tool_calls.append(ToolRecord(call.name, call.arguments, result.ok,
                                                      result.content[:4000], result.duration_ms))
                    observations.append(render_observation(call.name, result.ok, result.content,
                                                           self.cfg.tool_result_max_chars))
                # 部分 tool_call 解析失败（已执行好的那部分），也告诉模型
                for w in parsed.warnings:
                    if "JSON" in w or "缺少" in w:
                        observations.append(FORMAT_ERROR_HINT.format(errors=w))
                turn.steps.append(step)
                scratch.append({"role": "assistant", "content": render_assistant(parsed.thought, parsed.tool_calls)})
                scratch.append({"role": "user", "content": "\n".join(observations)})
                continue

            if parsed.kind == "final":
                turn.steps.append(Step(thought=parsed.thought))
                final, stop_reason = parsed.final_answer, "final"
                break

            # 格式错误
            format_errors += 1
            trace.log("format_error", step=step_no, errors=parsed.errors, count=format_errors)
            if format_errors >= self.cfg.max_format_errors:
                final = "抱歉，我这次没能组织好回答，请换个说法再试一次。"
                stop_reason = "format_error"
                break
            scratch.append({"role": "assistant", "content": resp.content[:1000] or "(空)"})
            scratch.append({"role": "user", "content": FORMAT_ERROR_HINT.format(errors="; ".join(parsed.errors))})
        else:
            final, stop_reason = self._force_final(session, user_input, scratch, system, trace), "max_steps"

        turn.final, turn.stop_reason = final or "", stop_reason
        session.turns.append(turn)
        try:
            self.sessions.save(session)
        except OSError as e:
            trace.log("save_error", error=str(e))
        tool_calls = [{"name": tc.name, "arguments": tc.arguments, "ok": tc.ok}
                      for s in turn.steps for tc in s.tool_calls]
        trace.log("final", stop_reason=stop_reason, answer=turn.final, steps=len(turn.steps), tool_calls=tool_calls)
        return AgentResult(turn.final, stop_reason, run_id, len(turn.steps), tool_calls, compressed, trace.events)

    def _force_final(self, session, user_input, scratch, system, trace) -> str:
        trace.log("max_steps_reached", max_steps=self.cfg.max_steps)
        messages = self.ctx.build(session, user_input, scratch + [{"role": "user", "content": FORCE_FINAL_PROMPT}],
                                  system=system)
        try:
            resp = self.llm.chat(messages, stop=["<observation"])
            parsed = parse_output(resp.content, resp.reasoning)
            if parsed.final_answer:
                return parsed.final_answer
        except LLMError as e:
            trace.log("llm_error", step="force_final", error=str(e))
        return "抱歉，这个问题需要的步骤超出了单轮上限，我先停在这里。你可以把问题拆小一点再问我。"


def _short(s: str, n: int = 60) -> str:
    return s if len(s) <= n else s[:n] + "…"
