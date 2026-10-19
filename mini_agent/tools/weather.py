"""Mock 天气：按 (城市, 日期) 做哈希生成确定性的天气，保证同一问题多次查询结果一致，便于测试。"""
from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta

from .base import Tool, ToolContext, ToolError

CITY_BASE_TEMP = {
    "北京": 14, "上海": 19, "广州": 25, "深圳": 26, "厦门": 23, "杭州": 18,
    "武汉": 18, "成都": 17, "西安": 14, "哈尔滨": 5, "南京": 17, "重庆": 19,
}
CONDITIONS = ["晴", "多云", "阴", "小雨", "中雨", "晴转多云", "多云转小雨"]
_RELATIVE = {"today": 0, "今天": 0, "tomorrow": 1, "明天": 1, "后天": 2, "day_after_tomorrow": 2}


def _resolve_date(raw: str, today: date) -> date:
    raw = raw.strip()
    if raw in _RELATIVE:
        return today + timedelta(days=_RELATIVE[raw])
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError as e:
        raise ToolError(f"无法识别的日期 {raw!r}，请用 today/tomorrow/后天 或 YYYY-MM-DD") from e


def get_weather(city: str, date_str: str = "today", today: date | None = None) -> dict:
    city = city.strip().removesuffix("市")
    if city not in CITY_BASE_TEMP:
        raise ToolError(f"暂不支持城市 {city!r}，支持: {'、'.join(CITY_BASE_TEMP)}")
    today = today or date.today()
    d = _resolve_date(date_str, today)
    if not -1 <= (d - today).days <= 6:
        raise ToolError("只能查询昨天到未来 6 天的天气")
    h = int(hashlib.md5(f"{city}{d.isoformat()}".encode()).hexdigest(), 16)
    base = CITY_BASE_TEMP[city]
    high = base + h % 7
    low = high - 5 - (h >> 4) % 5
    return {
        "city": city,
        "date": d.isoformat(),
        "condition": CONDITIONS[(h >> 8) % len(CONDITIONS)],
        "temp_high_c": high,
        "temp_low_c": low,
        "humidity_pct": 40 + (h >> 12) % 50,
        "wind": f"{['北', '东北', '东', '东南', '南', '西南', '西', '西北'][(h >> 16) % 8]}风{1 + (h >> 20) % 4}级",
        "source": "mock",
    }


def _run(args: dict, ctx: ToolContext) -> dict:
    return get_weather(args["city"], args.get("date", "today"))


weather_tool = Tool(
    name="weather",
    description="查询中国主要城市的天气（mock 数据）。可查昨天到未来 6 天。",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "城市名，如 北京、上海"},
            "date": {"type": "string", "description": "today / tomorrow / 后天 或 YYYY-MM-DD，默认 today"},
        },
        "required": ["city"],
        "additionalProperties": False,
    },
    func=_run,
)
