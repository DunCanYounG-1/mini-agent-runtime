"""执行日志 / trace。

每个事件一行 JSON，写到 data/traces/{user}/{session}.jsonl，同一次 run 共享 run_id。
同时可挂一个回调（CLI 的 --verbose 用它实时打印思考和工具调用）。
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

EventCallback = Callable[[dict], None]


class Tracer:
    def __init__(self, base_dir: str | Path | None, on_event: EventCallback | None = None):
        self.base = Path(base_dir) / "traces" if base_dir is not None else None
        self.on_event = on_event
        self._lock = threading.Lock()

    def run(self, user_id: str, session_id: str, run_id: str) -> "RunTrace":
        path = self.base / user_id / f"{session_id}.jsonl" if self.base else None
        return RunTrace(self, path, user_id, session_id, run_id)

    def _write(self, path: Path | None, event: dict) -> None:
        if path is not None:
            with self._lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
        if self.on_event:
            try:
                self.on_event(event)
            except Exception:
                pass  # 打印回调出错不影响主流程


class RunTrace:
    def __init__(self, tracer: Tracer, path: Path | None, user_id: str, session_id: str, run_id: str):
        self.tracer = tracer
        self.path = path
        self.user_id = user_id
        self.session_id = session_id
        self.run_id = run_id
        self.events: list[dict] = []
        self._t0 = time.perf_counter()

    def log(self, event: str, **data: Any) -> dict:
        e = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "elapsed_ms": int((time.perf_counter() - self._t0) * 1000),
            "run_id": self.run_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "event": event,
            **data,
        }
        self.events.append(e)
        self.tracer._write(self.path, e)
        return e

    def of(self, event: str) -> list[dict]:
        return [e for e in self.events if e["event"] == event]
