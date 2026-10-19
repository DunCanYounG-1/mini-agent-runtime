import time
from datetime import date

import pytest

from mini_agent.session import Session
from mini_agent.tools import Tool, ToolContext, ToolRegistry, default_registry, validate_schema
from mini_agent.tools.calculator import calculate
from mini_agent.tools.weather import get_weather
from mini_agent.tools.base import ToolError


@pytest.fixture
def ctx():
    s = Session("A", "w1")
    return ToolContext(session=s, user_id="A", session_id="w1")


def test_registry_renders_name_description_schema():
    text = default_registry().render_for_prompt()
    for name in ("calculator", "search", "weather", "todo"):
        assert f"### {name}" in text
    assert '"required": ["action"]' in text and '"enum": ["add", "list", "complete", "delete"]' in text


def test_registry_rejects_duplicate_and_bad_schema():
    reg = ToolRegistry()
    t = Tool("x", "d", {"type": "object", "properties": {}}, lambda a, c: "ok")
    reg.register(t)
    with pytest.raises(ValueError):
        reg.register(t)
    with pytest.raises(ValueError):
        reg.register(Tool("y", "d", {"type": "string"}, lambda a, c: "ok"))


def test_decorator_registration(ctx):
    reg = ToolRegistry()

    @reg.tool(name="echo", description="回显", parameters={"type": "object", "properties": {"s": {"type": "string"}}})
    def echo(args, c):
        return args["s"]

    assert reg.execute("echo", {"s": "hi"}, ctx).content == "hi"


def test_schema_validation():
    schema = default_registry().get("todo").parameters
    assert validate_schema(schema, {"action": "add", "content": "x"}) == []
    assert "缺少必填参数" in validate_schema(schema, {})[0]
    assert "必须是" in validate_schema(schema, {"action": "fly"})[0]
    assert "未知参数" in validate_schema(schema, {"action": "list", "foo": 1})[0]
    assert "应为 integer" in validate_schema(schema, {"action": "complete", "id": "1"})[0]
    assert "应为 integer" in validate_schema(schema, {"action": "complete", "id": True})[0]


def test_execute_errors_never_raise(ctx):
    reg = default_registry()
    r = reg.execute("nope", {}, ctx)
    assert not r.ok and r.error_type == "not_found" and "calculator" in r.content
    r = reg.execute("weather", {"town": "北京"}, ctx)
    assert not r.ok and r.error_type == "validation"
    r = reg.execute("weather", "北京", ctx)
    assert not r.ok and r.error_type == "validation"

    def boom(args, c):
        raise RuntimeError("炸了")

    reg.register(Tool("boom", "d", {"type": "object", "properties": {}}, boom))
    r = reg.execute("boom", {}, ctx)
    assert not r.ok and r.error_type == "exception" and "炸了" in r.content


def test_tool_timeout(ctx):
    reg = ToolRegistry()
    reg.register(Tool("slow", "d", {"type": "object", "properties": {}}, lambda a, c: time.sleep(1), timeout_s=0.1))
    r = reg.execute("slow", {}, ctx)
    assert not r.ok and r.error_type == "timeout"


@pytest.mark.parametrize("expr,expected", [
    ("1+2*3", "7"), ("(1234+5678)*3", "20736"), ("sqrt(16)", "4"), ("10/4", "2.5"),
    ("2^10", "1024"), ("3×4", "12"), ("round(pi, 2)", "3.14"), ("-5 % 3", "1"),
])
def test_calculator(expr, expected):
    assert calculate(expr).endswith("= " + expected)


@pytest.mark.parametrize("expr", ["__import__('os').system('ls')", "open('x')", "1/0", "9**99999", "1+"])
def test_calculator_rejects_unsafe_or_invalid(expr):
    with pytest.raises(ToolError):
        calculate(expr)


def test_weather_deterministic_and_relative_dates():
    today = date(2026, 10, 19)
    a = get_weather("北京", "today", today)
    assert a == get_weather("北京市", "2026-10-19", today)
    assert get_weather("北京", "明天", today)["date"] == "2026-10-20"
    assert a["temp_low_c"] < a["temp_high_c"]
    with pytest.raises(ToolError):
        get_weather("火星", "today", today)
    with pytest.raises(ToolError):
        get_weather("北京", "下个月", today)


def test_todo_crud(ctx):
    reg = default_registry()
    r = reg.execute("todo", {"action": "add", "content": "带伞", "due": "明天"}, ctx)
    assert r.ok and '"id": 1' in r.content
    reg.execute("todo", {"action": "add", "content": "写周报"}, ctx)
    assert reg.execute("todo", {"action": "complete", "id": 1}, ctx).ok
    open_items = reg.execute("todo", {"action": "list", "status": "open"}, ctx).content
    assert "写周报" in open_items and "带伞" not in open_items
    assert reg.execute("todo", {"action": "delete", "id": 2}, ctx).ok
    r = reg.execute("todo", {"action": "complete", "id": 99}, ctx)
    assert not r.ok and "不存在" in r.content
    r = reg.execute("todo", {"action": "add"}, ctx)
    assert not r.ok and "content" in r.content
    assert ctx.session.state["next_todo_id"] == 3


def test_search_mock():
    r = default_registry().execute("search", {"query": "周报怎么写"}, ToolContext(Session("A", "x"), "A", "x"))
    assert r.ok and "周报写作指南" in r.content
