"""LLM 输出协议与解析。

协议（写在 system prompt 里）：
    <thought>思考过程</thought>
    <tool_call>{"name": "...", "arguments": {...}}</tool_call>   # 可多个
  或
    <thought>思考过程</thought>
    <final_answer>给用户的回答</final_answer>

为什么用 XML 标签 + JSON 而不是厂商 function calling：
  - 与供应商解耦，任何能输出文本的模型都能跑；
  - 思考过程 / 工具调用 / 最终答案三者边界清晰，可以显式解析、显式容错；
  - 题目要求自己实现解析逻辑。

解析器的容错点（真实模型都踩过）：
  - 标签缺失闭合、JSON 外面包了 ```json 代码块、arguments 被写成字符串；
  - 字段名写成 tool/args/parameters；
  - 推理模型把思考放在 <think> 或 reasoning_content 里；
  - tool_call 之后又自己编了 <observation>（丢弃）；
  - 同时给了 tool_call 和 final_answer（以 tool_call 为准，final 属于“没看结果就下结论”）；
  - 完全没有标签（视为直接回复）。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

_THOUGHT_RE = re.compile(r"<(thought|think)>(.*?)(?:</\1>|(?=<tool_call>|<final_answer>)|$)", re.S)
_TOOL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|(?=<tool_call>|<final_answer>|<observation)|$)", re.S)
_FINAL_RE = re.compile(r"<final_answer>(.*?)(?:</final_answer>|$)", re.S)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")
_ANY_TAG_RE = re.compile(r"</?(thought|think|tool_call|final_answer|observation)[^>]*>")


@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str = ""


@dataclass
class ParsedOutput:
    raw: str
    thought: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    final_answer: str | None = None
    errors: list[str] = field(default_factory=list)   # 需要回传给 LLM 让它修正的格式错误
    warnings: list[str] = field(default_factory=list)  # 已自动容错的问题，只记日志

    @property
    def kind(self) -> str:
        if self.tool_calls:
            return "tool_call"
        if self.final_answer is not None:
            return "final"
        return "format_error"


def _extract_json_object(text: str) -> str | None:
    """从文本中取出第一个括号配平的 {...}（考虑字符串内的括号与转义）。"""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_tool_call(body: str) -> ToolCall:
    body = _FENCE_RE.sub("", body.strip())
    blob = _extract_json_object(body)
    if blob is None:
        raise ValueError(f"tool_call 内没有 JSON 对象: {body[:80]!r}")
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        # 常见小毛病：尾逗号
        try:
            obj = json.loads(re.sub(r",\s*([}\]])", r"\1", blob))
        except json.JSONDecodeError as e:
            raise ValueError(f"tool_call JSON 解析失败: {e.msg}") from e
    name = obj.get("name") or obj.get("tool") or obj.get("tool_name")
    if not name or not isinstance(name, str):
        raise ValueError("tool_call 缺少 name 字段")
    args = obj.get("arguments", obj.get("args", obj.get("parameters", {})))
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError as e:
            raise ValueError("arguments 是字符串但不是合法 JSON") from e
    if args is None:
        args = {}
    return ToolCall(name=name.strip(), arguments=args)


def parse_output(text: str, reasoning: str = "") -> ParsedOutput:
    text = text or ""
    out = ParsedOutput(raw=text)

    thoughts = [m.group(2).strip() for m in _THOUGHT_RE.finditer(text)]
    if reasoning.strip():
        thoughts.insert(0, reasoning.strip())
    out.thought = "\n".join(t for t in thoughts if t)

    # 模型自己编的 observation 之后的内容全部丢弃
    cut = text.find("<observation")
    body = text[:cut] if cut >= 0 else text
    if cut >= 0:
        out.warnings.append("模型自行生成了 <observation>，已丢弃其后内容")

    for i, m in enumerate(_TOOL_RE.finditer(body), start=1):
        try:
            call = parse_tool_call(m.group(1))
            call.id = f"call_{i}"
            out.tool_calls.append(call)
        except ValueError as e:
            out.errors.append(str(e))

    fm = _FINAL_RE.search(body)
    if fm:
        final = fm.group(1).strip()
        if out.tool_calls:
            out.warnings.append("同时输出了 tool_call 和 final_answer，以 tool_call 为准")
        else:
            out.final_answer = final

    if out.errors and out.tool_calls:
        # 部分 tool_call 坏了：先执行好的，坏的作为错误在 observation 里提示
        out.warnings.extend(out.errors)
        out.errors = []

    if not out.tool_calls and out.final_answer is None and not out.errors:
        # 没有任何协议标签：把去掉思考后的正文当作直接回复（兼容不听话的模型）
        plain = _THOUGHT_RE.sub("", text)
        plain = _ANY_TAG_RE.sub("", plain).strip()
        if plain:
            out.final_answer = plain
            out.warnings.append("未使用协议标签，按直接回复处理")
        else:
            out.errors.append("输出为空或只有思考，没有 tool_call 或 final_answer")
    return out


def render_assistant(thought: str, tool_calls: list[ToolCall] | None = None, final: str | None = None) -> str:
    """把解析后的结果规范化地写回历史，丢掉模型输出里的噪音。"""
    parts = []
    if thought:
        parts.append(f"<thought>{thought}</thought>")
    for c in tool_calls or []:
        payload = json.dumps({"name": c.name, "arguments": c.arguments}, ensure_ascii=False)
        parts.append(f"<tool_call>{payload}</tool_call>")
    if final is not None:
        parts.append(f"<final_answer>{final}</final_answer>")
    return "\n".join(parts)


def render_observation(name: str, ok: bool, content: str, max_chars: int) -> str:
    if len(content) > max_chars:
        content = content[:max_chars] + f"…（已截断，原长 {len(content)} 字符）"
    status = "ok" if ok else "error"
    return f'<observation tool="{name}" status="{status}">\n{content}\n</observation>'


FORMAT_ERROR_HINT = (
    "<observation tool=\"system\" status=\"error\">\n你的上一条输出不符合协议：{errors}\n"
    "请严格使用 <thought>…</thought> 加 <tool_call>{{JSON}}</tool_call> 或 <final_answer>…</final_answer>。\n"
    "</observation>"
)
