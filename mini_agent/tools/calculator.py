"""安全计算器：用 AST 白名单求值，不用 eval。"""
from __future__ import annotations

import ast
import math
import operator

from .base import Tool, ToolContext, ToolError

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "log": math.log, "log10": math.log10,
    "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "floor": math.floor, "ceil": math.ceil, "max": max, "min": min,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ToolError("指数过大")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS and not node.keywords:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ToolError(f"不支持的表达式成分: {ast.dump(node)[:60]}")


def calculate(expression: str) -> str:
    expr = expression.replace("×", "*").replace("÷", "/").replace("^", "**").replace("，", ",")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ToolError(f"表达式语法错误: {expression}") from e
    try:
        value = _eval(tree)
    except ZeroDivisionError as e:
        raise ToolError("除数不能为 0") from e
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            value = int(value)
        else:
            value = round(value, 10)
    return f"{expression} = {value}"


def _run(args: dict, ctx: ToolContext) -> str:
    return calculate(args["expression"])


calculator_tool = Tool(
    name="calculator",
    description="精确计算数学表达式。支持 + - * / // % **、括号，以及 sqrt/abs/round/log/exp/sin/cos/max/min、常量 pi/e。"
    "任何需要精确数值的计算都应使用本工具。",
    parameters={
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "数学表达式，如 (3+5)*2 或 sqrt(16)"}},
        "required": ["expression"],
        "additionalProperties": False,
    },
    func=_run,
)
