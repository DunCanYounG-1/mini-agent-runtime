"""Session 管理。

一个 session = (user_id, session_id) 唯一确定，拥有自己独立的：
  - turns    活跃历史（会进入 context）
  - archive  被压缩掉的旧轮次原文（不进 context，只用于召回）
  - summary  旧轮次的滚动摘要（每轮注入 system prompt）
  - state    结构化状态，如 todos（每轮从这里实时渲染，作为“事实来源”）

每个 session 一个 JSON 文件，原子写入；进程重启后可以接着聊。
同一 session 同时只允许一个 run（加锁），不同 session 可以并发。
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

_ID_RE = re.compile(r"^[A-Za-z0-9_\-一-鿿]{1,64}$")


class SessionBusyError(Exception):
    """同一 session 已有请求在执行。"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def check_id(value: str, kind: str) -> str:
    if not _ID_RE.match(value or ""):
        raise ValueError(f"非法 {kind}: {value!r}（只允许字母数字下划线横线中文，≤64）")
    return value


@dataclass
class ToolRecord:
    name: str
    arguments: dict
    ok: bool
    result: str
    duration_ms: int = 0


@dataclass
class Step:
    thought: str = ""
    tool_calls: list[ToolRecord] = field(default_factory=list)


@dataclass
class Turn:
    user: str
    final: str = ""
    steps: list[Step] = field(default_factory=list)
    stop_reason: str = ""            # final / max_steps / llm_error / format_error
    run_id: str = ""
    ts: str = field(default_factory=_now)

    @property
    def ok(self) -> bool:
        return self.stop_reason in ("final", "max_steps", "format_error")

    @staticmethod
    def from_dict(d: dict) -> "Turn":
        steps = [
            Step(s.get("thought", ""), [ToolRecord(**tc) for tc in s.get("tool_calls", [])])
            for s in d.get("steps", [])
        ]
        return Turn(d["user"], d.get("final", ""), steps, d.get("stop_reason", ""), d.get("run_id", ""),
                    d.get("ts", _now()))


@dataclass
class Session:
    user_id: str
    session_id: str
    title: str = ""
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    turns: list[Turn] = field(default_factory=list)
    archive: list[Turn] = field(default_factory=list)
    summary: str = ""
    state: dict = field(default_factory=dict)
    compress_count: int = 0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    @property
    def total_turns(self) -> int:
        return len(self.archive) + len(self.turns)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Session":
        s = Session(
            user_id=d["user_id"],
            session_id=d["session_id"],
            title=d.get("title", ""),
            created_at=d.get("created_at", _now()),
            updated_at=d.get("updated_at", _now()),
            summary=d.get("summary", ""),
            state=d.get("state", {}),
            compress_count=d.get("compress_count", 0),
        )
        s.turns = [Turn.from_dict(t) for t in d.get("turns", [])]
        s.archive = [Turn.from_dict(t) for t in d.get("archive", [])]
        return s


class SessionStore:
    def __init__(self, base_dir: str | os.PathLike):
        self.base = Path(base_dir) / "sessions"

    def path(self, user_id: str, session_id: str) -> Path:
        return self.base / check_id(user_id, "user_id") / f"{check_id(session_id, 'session_id')}.json"

    def load(self, user_id: str, session_id: str) -> Session | None:
        p = self.path(user_id, session_id)
        if not p.exists():
            return None
        try:
            return Session.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # 文件损坏时保留现场，不静默覆盖
            backup = p.with_suffix(f".corrupt.{int(time.time())}")
            p.rename(backup)
            raise RuntimeError(f"session 文件损坏，已备份到 {backup}: {e}") from e

    def save(self, session: Session) -> None:
        p = self.path(session.user_id, session.session_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        session.updated_at = _now()
        tmp = p.with_suffix(f".tmp.{uuid.uuid4().hex[:6]}")
        tmp.write_text(json.dumps(session.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, p)  # 原子替换，进程崩溃不会留下半个文件

    def list(self, user_id: str) -> list[dict]:
        d = self.base / check_id(user_id, "user_id")
        if not d.exists():
            return []
        out = []
        for f in sorted(d.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            out.append({
                "session_id": data["session_id"],
                "title": data.get("title", ""),
                "turns": len(data.get("turns", [])) + len(data.get("archive", [])),
                "updated_at": data.get("updated_at", ""),
            })
        return out


class SessionManager:
    def __init__(self, store: SessionStore):
        self.store = store
        self._cache: dict[tuple[str, str], Session] = {}
        self._guard = threading.Lock()

    def get(self, user_id: str, session_id: str, create: bool = True) -> Session:
        key = (check_id(user_id, "user_id"), check_id(session_id, "session_id"))
        with self._guard:
            if key not in self._cache:
                s = self.store.load(*key)
                if s is None:
                    if not create:
                        raise KeyError(f"session 不存在: {key}")
                    s = Session(user_id=key[0], session_id=key[1])
                self._cache[key] = s
            return self._cache[key]

    def new_session_id(self) -> str:
        return datetime.now().strftime("s%m%d%H%M%S") + uuid.uuid4().hex[:4]

    def list(self, user_id: str) -> list[dict]:
        return self.store.list(user_id)

    def save(self, session: Session) -> None:
        self.store.save(session)

    @contextmanager
    def acquire(self, session: Session, timeout: float = 0.0) -> Iterator[Session]:
        """session 级互斥。timeout=0 表示忙就立即报错（交给上层决定排队还是提示用户）。"""
        got = session._lock.acquire(timeout=timeout) if timeout > 0 else session._lock.acquire(blocking=False)
        if not got:
            raise SessionBusyError(f"会话 {session.session_id} 正在处理上一条消息")
        try:
            yield session
        finally:
            session._lock.release()
