# -*- coding: utf-8 -*-
"""控制网关：接收外部系统下发的坐标指令，校验后稳定驱动机械臂或仿真器。

设计目标（联调四）:
    - 协议入口统一：一条 JSON 指令 = 一个动作，可来自 WebSocket / Function Calling；
    - 先校验后驱动：维度、类型、有限性、工作空间越界都直接拒绝并返回原因，
      不做静默裁剪，避免上层误判指令已完整执行；
    - 串行驱动：同一网关内的指令用锁串行执行，避免两条指令同时抢驱动；
    - 幂等：带 command_id 的重复下发不会重复动作（网络重试常见场景）；
    - 急停保护：急停状态下拒绝新指令，复位后恢复；
    - 失败可观测：异常被捕获并复位状态机，返回 ok=False + error + 耗时。

指令协议:
    {
      "command_id": "cmd-001",            # 可选，用于幂等
      "action": "move" | "grasp",         # move 移动到坐标；grasp 坐标抓取并递送
      "pick_pose": [x, y, z, rx, ry, rz], # 必填，单位 mm / 度
      "place_pose": [...],                # grasp 可选，缺省用网关默认递送点
      "object_name": "水杯"                # grasp 可选
    }
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from src.qiming.arm_control.status import status_hub
from src.qiming.arm_control.workspace import PoseValidationError, validate_pose
from src.qiming.models import ArmAction

# 幂等缓存保留的最近指令条数
_IDEMPOTENCY_CACHE_SIZE = 100


@dataclass
class GatewayStats:
    """网关累计统计，供联调脚本 / 状态查询使用。"""

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    rejected: int = 0
    duplicate: int = 0
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "rejected": self.rejected,
            "duplicate": self.duplicate,
            "last_error": self.last_error,
        }


@dataclass
class CoordinateCommand:
    """校验通过的坐标指令。"""

    action: str
    pick_pose: list[float]
    place_pose: list[float] | None = None
    object_name: str = "物体"
    command_id: str | None = None


class CoordinateGateway:
    """坐标指令网关：校验 + 串行 + 幂等 + 急停保护。"""

    def __init__(
        self,
        controller,
        default_place_pose: Sequence[float] | None = None,
    ) -> None:
        self.controller = controller
        self.default_place_pose = list(default_place_pose) if default_place_pose else None
        self._lock = threading.RLock()
        self._results: dict[str, dict[str, Any]] = {}
        self.stats = GatewayStats()

    # ------------------------------------------------------------------
    # 对外入口
    # ------------------------------------------------------------------
    def submit(self, command: dict[str, Any]) -> dict[str, Any]:
        """接收一条坐标指令并执行，返回结构化结果（不抛异常）。"""
        command_id = None
        if isinstance(command, dict):
            raw_id = command.get("command_id")
            command_id = str(raw_id) if raw_id else None

        try:
            cmd = self._normalize(command)
        except PoseValidationError as exc:
            return self._record_rejected(command_id, str(exc))

        with self._lock:  # 串行驱动：同一时刻只允许一条指令占用机械臂
            cached = self._results.get(cmd.command_id) if cmd.command_id else None
            if cached is not None:
                self.stats.duplicate += 1
                return {**cached, "duplicate": True}

            if status_hub.snapshot()["state"] == "estop":
                return self._record_rejected(
                    cmd.command_id, "机械臂处于急停状态，请先复位后再下发指令")

            started = time.perf_counter()
            try:
                message = self._dispatch(cmd)
                result = {
                    "ok": True,
                    "action": cmd.action,
                    "command_id": cmd.command_id,
                    "pose": list(cmd.pick_pose),
                    "object": cmd.object_name if cmd.action == "grasp" else None,
                    "driver": status_hub.snapshot()["driver"],
                    "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
                    "message": message,
                }
                self.stats.total += 1
                self.stats.succeeded += 1
            except Exception as exc:  # noqa: BLE001 - 驱动异常统一回收，保持网关可用
                status_hub.update(state="idle", phase="执行中断", target="")
                result = {
                    "ok": False,
                    "action": cmd.action,
                    "command_id": cmd.command_id,
                    "pose": list(cmd.pick_pose),
                    "driver": status_hub.snapshot()["driver"],
                    "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 2),
                    "error": f"驱动执行失败: {exc}",
                }
                self.stats.total += 1
                self.stats.failed += 1
                self.stats.last_error = result["error"]

            if cmd.command_id:
                self._remember(cmd.command_id, result)
            return result

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _normalize(self, command: dict[str, Any]) -> CoordinateCommand:
        """校验并规范化指令；不合法时抛 PoseValidationError。"""
        if not isinstance(command, dict):
            raise PoseValidationError("坐标指令必须是 JSON 对象")

        action = str(command.get("action") or "move").strip().lower()
        if action not in ("move", "grasp"):
            raise PoseValidationError(f"action 只能是 move 或 grasp，实际: {action}")

        if "pick_pose" not in command:
            raise PoseValidationError("缺少 pick_pose")
        pick_pose = validate_pose(command["pick_pose"], "pick_pose")

        place_pose = None
        if action == "grasp":
            raw_place = command.get("place_pose")
            if raw_place is not None:
                place_pose = validate_pose(raw_place, "place_pose")
            elif self.default_place_pose is not None:
                place_pose = validate_pose(self.default_place_pose, "default_place_pose")
            else:
                raise PoseValidationError("grasp 指令缺少 place_pose，且网关未配置默认递送点")

        object_name = str(command.get("object_name") or "物体")
        raw_id = command.get("command_id")
        command_id = str(raw_id) if raw_id else None

        return CoordinateCommand(
            action=action,
            pick_pose=pick_pose,
            place_pose=place_pose,
            object_name=object_name,
            command_id=command_id,
        )

    def _dispatch(self, cmd: CoordinateCommand) -> str:
        """把已校验的指令交给执行层，返回给调用方的消息。"""
        if cmd.action == "move":
            self.controller.move_to_pose(
                cmd.pick_pose, label="移动到目标坐标", target=cmd.object_name)
            return f"已移动到坐标 {[round(v, 2) for v in cmd.pick_pose[:3]]}"

        action = ArmAction(
            target_name=cmd.object_name,
            pick_pose=tuple(cmd.pick_pose),
            place_pose=tuple(cmd.place_pose or []),
            reply=f"好的，正在为您拿取{cmd.object_name}",
        )
        return self.controller.execute(action)

    def _record_rejected(self, command_id: str | None, reason: str) -> dict[str, Any]:
        with self._lock:
            self.stats.total += 1
            self.stats.rejected += 1
            self.stats.last_error = reason
        return {
            "ok": False,
            "rejected": True,
            "command_id": command_id,
            "error": reason,
        }

    def _remember(self, command_id: str, result: dict[str, Any]) -> None:
        """缓存指令结果用于幂等；超过上限时丢弃最早的一条。"""
        self._results[command_id] = result
        while len(self._results) > _IDEMPOTENCY_CACHE_SIZE:
            self._results.pop(next(iter(self._results)))

    def reset(self) -> None:
        """清空幂等缓存与统计（系统复位时调用）。"""
        with self._lock:
            self._results.clear()
            self.stats = GatewayStats()
