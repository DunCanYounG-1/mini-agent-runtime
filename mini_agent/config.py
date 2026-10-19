"""运行时配置。所有阈值集中在这里，便于测试时调小来触发边界行为。"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def load_dotenv(path: str | os.PathLike = ".env") -> None:
    """极简 .env 解析：KEY=VALUE，已存在的环境变量不覆盖。"""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class AgentConfig:
    # —— Loop ——
    max_steps: int = 6                  # 单次用户输入内最多调用 LLM 的次数（防死循环）
    max_format_errors: int = 2          # 连续输出格式错误的容忍次数
    # —— Context ——
    max_context_tokens: int = 8000      # 估算 token 上限
    compress_threshold: float = 0.75    # 超过上限的这个比例就触发压缩
    keep_recent_turns: int = 4          # 压缩时保留的最近完整轮数
    max_history_turns: int = 20         # 活跃历史最大轮数（超出即使 token 未满也压缩）
    tool_result_max_chars: int = 1500   # 本轮工具结果截断长度
    old_observation_max_chars: int = 300  # 历史轮次里工具结果的截断长度
    recall_top_k: int = 2               # 从归档里召回的历史片段数
    # —— LLM ——
    temperature: float = 0.2
    llm_timeout: float = 60.0
    llm_max_retries: int = 3
    # —— Storage ——
    data_dir: Path = field(default_factory=lambda: Path("data"))

    @classmethod
    def from_env(cls) -> "AgentConfig":
        cfg = cls()
        cfg.max_steps = _env_int("AGENT_MAX_STEPS", cfg.max_steps)
        cfg.max_context_tokens = _env_int("AGENT_MAX_CONTEXT_TOKENS", cfg.max_context_tokens)
        cfg.data_dir = Path(os.environ.get("AGENT_DATA_DIR", cfg.data_dir))
        return cfg
