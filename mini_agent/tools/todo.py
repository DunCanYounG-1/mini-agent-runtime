"""待办工具。数据存在 session.state 里，所以天然按 session 隔离、随 session 一起持久化。"""
from __future__ import annotations

from datetime import datetime

from .base import Tool, ToolContext, ToolError


def _todos(ctx: ToolContext) -> list[dict]:
    return ctx.session.state.setdefault("todos", [])


def _find(ctx: ToolContext, todo_id: int | None) -> dict:
    if todo_id is None:
        raise ToolError("该操作需要提供 id")
    for item in _todos(ctx):
        if item["id"] == todo_id:
            return item
    ids = [t["id"] for t in _todos(ctx)]
    raise ToolError(f"待办 id={todo_id} 不存在，现有 id: {ids}")


def _run(args: dict, ctx: ToolContext) -> dict:
    action = args["action"]
    todos = _todos(ctx)
    if action == "add":
        content = (args.get("content") or "").strip()
        if not content:
            raise ToolError("add 操作需要 content")
        next_id = ctx.session.state.get("next_todo_id", 1)
        item = {
            "id": next_id,
            "content": content,
            "due": args.get("due", ""),
            "done": False,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        todos.append(item)
        ctx.session.state["next_todo_id"] = next_id + 1
        return {"ok": True, "added": item, "total": len(todos)}
    if action == "list":
        status = args.get("status", "all")
        items = [t for t in todos if status == "all" or t["done"] == (status == "done")]
        return {"ok": True, "status": status, "items": items}
    if action == "complete":
        item = _find(ctx, args.get("id"))
        item["done"] = True
        return {"ok": True, "completed": item}
    if action == "delete":
        item = _find(ctx, args.get("id"))
        todos.remove(item)
        return {"ok": True, "deleted": item}
    raise ToolError(f"未知 action {action}")  # Schema 的 enum 已拦截，这里兜底


todo_tool = Tool(
    name="todo",
    description="管理当前会话的待办事项：add 新增、list 查看、complete 标记完成、delete 删除。"
    "用户说“记一下/提醒我/加个待办”时用 add；问“有哪些待办”时用 list。",
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["add", "list", "complete", "delete"]},
            "content": {"type": "string", "description": "待办内容（add 必填）"},
            "due": {"type": "string", "description": "截止时间，可选，如 2026-10-20 或 明天下午"},
            "id": {"type": "integer", "description": "待办 id（complete/delete 必填）"},
            "status": {"type": "string", "enum": ["all", "open", "done"], "description": "list 过滤，默认 all"},
        },
        "required": ["action"],
        "additionalProperties": False,
    },
    func=_run,
)


def render_todos(todos: list[dict]) -> str:
    """供 system prompt 使用的待办快照。"""
    if not todos:
        return "（暂无待办）"
    lines = []
    for t in todos:
        mark = "x" if t["done"] else " "
        due = f"（截止 {t['due']}）" if t.get("due") else ""
        lines.append(f"- [{mark}] #{t['id']} {t['content']}{due}")
    return "\n".join(lines)
