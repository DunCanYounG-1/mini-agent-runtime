"""工具注册机制：每个工具 = 名称 + 描述 + 参数 JSON Schema + 执行函数。

Registry 负责：
  1. 把 Schema 渲染进 system prompt，让 LLM 自主决定调用哪个工具；
  2. 执行前按 Schema 校验参数（错误信息回传给 LLM，让它自己修正）；
  3. 捕获工具异常与超时，统一包装成 ToolResult，绝不让异常打断 Agent 主循环。
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ..session import Session


@dataclass
class ToolContext:
    """工具执行时能拿到的运行时信息。有状态的工具（todo）通过它读写所属 session。"""

    session: "Session"
    user_id: str
    session_id: str


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[[dict, ToolContext], Any]
    timeout_s: float = 10.0

    def prompt_spec(self) -> str:
        schema = json.dumps(self.parameters, ensure_ascii=False)
        return f"### {self.name}\n{self.description}\n参数 Schema: {schema}"


@dataclass
class ToolResult:
    name: str
    arguments: dict
    ok: bool
    content: str
    duration_ms: int = 0
    error_type: str = ""            # validation / not_found / timeout / exception
    extra: dict = field(default_factory=dict)


class ToolError(Exception):
    """工具内部可预期的业务错误（如 todo id 不存在），信息会原样返回给 LLM。"""


_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}


def validate_schema(schema: dict, value: Any, path: str = "arguments") -> list[str]:
    """JSON Schema 的一个够用子集：type / required / properties / enum / items / additionalProperties。"""
    errors: list[str] = []
    expected = schema.get("type")
    if expected:
        py_type = _TYPE_MAP.get(expected)
        # bool 是 int 的子类，需要单独排除
        bad_bool = expected in ("integer", "number") and isinstance(value, bool)
        if py_type and (not isinstance(value, py_type) or bad_bool):
            return [f"{path} 应为 {expected}，实际为 {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path} 必须是 {schema['enum']} 之一，实际为 {value!r}")
    if expected == "object":
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in value:
                errors.append(f"缺少必填参数 {path}.{req}")
        for key, sub in value.items():
            if key in props:
                errors.extend(validate_schema(props[key], sub, f"{path}.{key}"))
            elif schema.get("additionalProperties") is False:
                errors.append(f"未知参数 {path}.{key}")
    if expected == "array" and "items" in schema:
        for i, item in enumerate(value):
            errors.extend(validate_schema(schema["items"], item, f"{path}[{i}]"))
    return errors


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool")

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"工具重名: {tool.name}")
        if tool.parameters.get("type") != "object":
            raise ValueError(f"{tool.name} 的参数 Schema 顶层必须是 object")
        self._tools[tool.name] = tool
        return tool

    def tool(self, name: str, description: str, parameters: dict, timeout_s: float = 10.0):
        """装饰器写法：@registry.tool(name=..., description=..., parameters=...)"""

        def deco(func: Callable[[dict, ToolContext], Any]) -> Callable:
            self.register(Tool(name, description, parameters, func, timeout_s))
            return func

        return deco

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def render_for_prompt(self) -> str:
        return "\n\n".join(t.prompt_spec() for t in self._tools.values())

    def execute(self, name: str, arguments: Any, ctx: ToolContext) -> ToolResult:
        t0 = time.perf_counter()

        def done(ok: bool, content: str, error_type: str = "") -> ToolResult:
            ms = int((time.perf_counter() - t0) * 1000)
            args = arguments if isinstance(arguments, dict) else {"_raw": arguments}
            return ToolResult(name, args, ok, content, ms, error_type)

        tool = self._tools.get(name)
        if tool is None:
            return done(False, f"工具 {name!r} 不存在，可用工具: {self.names}", "not_found")
        if not isinstance(arguments, dict):
            return done(False, "arguments 必须是 JSON 对象", "validation")
        errors = validate_schema(tool.parameters, arguments)
        if errors:
            return done(False, "参数校验失败: " + "; ".join(errors), "validation")
        future = self._pool.submit(tool.func, arguments, ctx)
        try:
            out = future.result(timeout=tool.timeout_s)
        except FutureTimeout:
            return done(False, f"工具执行超时（>{tool.timeout_s}s）", "timeout")
        except ToolError as e:
            return done(False, str(e), "tool_error")
        except Exception as e:  # 工具里的任何 bug 都不应该打断 Agent
            return done(False, f"工具内部异常: {type(e).__name__}: {e}", "exception")
        content = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
        return done(True, content)
