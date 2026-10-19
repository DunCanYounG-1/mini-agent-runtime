"""Mock 搜索：在内置语料上做字符二元组（bigram）重叠打分，返回 top-k。

接口形状与真实搜索一致（title / snippet / url），替换成真实搜索 API 只需改 _search 函数。
"""
from __future__ import annotations

from .base import Tool, ToolContext

CORPUS = [
    {
        "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
        "url": "https://example.com/react",
        "snippet": "ReAct 让大模型交替输出思考（Thought）与动作（Action），根据工具返回的观察（Observation）继续推理，"
        "是 Agent 循环的经典范式，2022 年由 Yao 等人提出。",
    },
    {
        "title": "周报写作指南",
        "url": "https://example.com/weekly-report",
        "snippet": "一份好的周报包含：本周完成事项（附量化结果）、进行中事项与风险、下周计划、需要的支持。"
        "建议用要点列表，每条以动词开头，控制在一页以内。",
    },
    {
        "title": "大模型上下文窗口与压缩",
        "url": "https://example.com/context-compression",
        "snippet": "常见的上下文压缩方法有：滑动窗口保留最近 N 轮、对早期对话做摘要、截断过长的工具输出、"
        "把关键状态（待办、用户偏好）结构化存储并每轮重新注入。",
    },
    {
        "title": "Python asyncio 入门",
        "url": "https://example.com/asyncio",
        "snippet": "asyncio 是 Python 的异步 IO 库，核心概念包括事件循环、协程（async def）、await、Task 与 gather。",
    },
    {
        "title": "北京气候概况",
        "url": "https://example.com/beijing-climate",
        "snippet": "北京属温带季风气候，秋季（9-11 月）天高气爽，昼夜温差大，10 月平均气温约 13°C，建议早晚添衣。",
    },
    {
        "title": "番茄工作法",
        "url": "https://example.com/pomodoro",
        "snippet": "番茄工作法：专注 25 分钟、休息 5 分钟为一个番茄，每 4 个番茄后长休息 15-30 分钟，适合拆解待办任务。",
    },
    {
        "title": "OpenAI 兼容接口 function calling 说明",
        "url": "https://example.com/function-calling",
        "snippet": "在请求中传 tools 字段（JSON Schema 描述函数），模型返回 tool_calls，"
        "客户端执行函数后以 role=tool 的消息回传结果。",
    },
]


def _bigrams(text: str) -> set[str]:
    t = "".join(text.lower().split())
    return {t[i : i + 2] for i in range(len(t) - 1)} or {t}


def _search(query: str, top_k: int) -> list[dict]:
    q = _bigrams(query)
    scored = []
    for doc in CORPUS:
        d = _bigrams(doc["title"] + doc["snippet"])
        score = len(q & d) / (len(q) or 1)
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [dict(doc, score=round(s, 3)) for s, doc in scored[:top_k]]


def _run(args: dict, ctx: ToolContext) -> dict:
    results = _search(args["query"], args.get("top_k", 3))
    return {"query": args["query"], "results": results, "note": "mock 搜索，数据来自内置语料"}


search_tool = Tool(
    name="search",
    description="搜索互联网资料（mock，内置语料覆盖：Agent/ReAct、周报写法、上下文压缩、Python、北京气候、番茄工作法等）。"
    "需要查找资料、方法、背景知识时使用。",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
            "top_k": {"type": "integer", "description": "返回条数，默认 3"},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    func=_run,
)
