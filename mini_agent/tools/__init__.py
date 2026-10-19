from .base import Tool, ToolContext, ToolError, ToolRegistry, ToolResult, validate_schema
from .calculator import calculator_tool
from .search import search_tool
from .todo import todo_tool
from .weather import weather_tool


def default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for t in (calculator_tool, search_tool, weather_tool, todo_tool):
        reg.register(t)
    return reg


__all__ = [
    "Tool", "ToolContext", "ToolError", "ToolRegistry", "ToolResult", "validate_schema", "default_registry",
]
