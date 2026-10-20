# -*- coding: utf-8 -*-
"""机械臂运行状态中心。

ArmController 在每个动作步骤前后更新一次状态，Orchestrator / WebSocket 端点
订阅后即可把状态实时推给前端，用于「机械臂状态」可视化卡片。

状态字段:
    state        运行状态: idle(空闲) / busy(执行中) / estop(已急停)
    phase        当前阶段描述，如「下降到抓取点」
    driver       驱动类型，如 MockArmDriver / MyCobotDriver
    target       当前操作的目标物体，空闲时为空串
    updated_at   最近一次更新时间 HH:MM:SS

订阅者在子线程里被回调（工具在 asyncio.to_thread 中执行），
因此回调只做轻量转发，不阻塞动作流程。
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any, Callable

Subscriber = Callable[[dict[str, Any]], None]

STATE_LABELS: dict[str, str] = {
    "idle": "空闲",
    "busy": "执行中",
    "estop": "已急停",
}


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


class RobotStatusHub:
    """线程安全的状态中心，支持发布/订阅。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[int, Subscriber] = {}
        self._next_token = 0
        self._status: dict[str, Any] = {
            "state": "idle",
            "phase": "待机中",
            "driver": "-",
            "target": "",
            "updated_at": _now(),
        }

    # ---------- 读写状态 ----------

    def snapshot(self) -> dict[str, Any]:
        """返回当前状态副本。"""
        with self._lock:
            return dict(self._status)

    def update(self, **fields: Any) -> dict[str, Any]:
        """合并字段、刷新时间戳并广播，返回更新后的状态副本。"""
        with self._lock:
            self._status.update(fields)
            self._status["updated_at"] = _now()
            snapshot = dict(self._status)
            subscribers = list(self._subscribers.values())

        for callback in subscribers:
            try:
                callback(dict(snapshot))
            except Exception:  # noqa: BLE001 - 单个订阅者异常不影响状态更新
                pass
        return snapshot

    def configure(self, driver: str) -> dict[str, Any]:
        """设置驱动类型（连接建立或驱动切换时调用）。"""
        return self.update(driver=driver)

    def reset(self) -> dict[str, Any]:
        """回到空闲状态（系统复位 / 会话重置时调用）。"""
        return self.update(state="idle", phase="待机中", target="")

    # ---------- 订阅 ----------

    def subscribe(self, callback: Subscriber) -> int:
        """注册订阅者，返回用于注销的 token。"""
        with self._lock:
            self._next_token += 1
            token = self._next_token
            self._subscribers[token] = callback
            return token

    def unsubscribe(self, token: int) -> None:
        with self._lock:
            self._subscribers.pop(token, None)


status_hub = RobotStatusHub()
