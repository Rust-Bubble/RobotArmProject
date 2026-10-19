# -*- coding: utf-8 -*-
"""内置的外部系统工具（Function Calling 基础 API）。

覆盖三类外部系统：
    - 机械臂：查询状态、急停、抓取执行（通过 ArmController，mock/真实由配置决定）
    - 语音反馈：TTS 播报
    - 系统：软件复位（暂以日志代替）

抓取动作的末端姿态与接近高度统一由末端姿态预设脚本给出
（见 src/qiming/arm_control/pose_presets.py），工具层只负责给出「抓哪里 / 放哪里」。

所有工具同步实现，统一返回 dict，方便大模型读结果。
"""

from __future__ import annotations

from src.qiming.arm_control.controller import ArmController
from src.qiming.arm_control.pose_presets import (
    DEFAULT_PLACE_PRESET_NAME,
    DEFAULT_PRESET_NAME,
    apply_preset,
)
from src.qiming.arm_control.status import status_hub
from src.qiming.feedback.tts import TtsFeedback
from src.qiming.models import ArmAction
from src.qiming.tools.registry import ToolRegistry

# 抓取点默认高度（桌面作业面，单位 mm）
DEFAULT_GRASP_HEIGHT = 20.0
# 未配置 handover_coords 时的兜底递送点
FALLBACK_HANDOVER = [80.0, -220.0, 180.0]

# 全局共享的反馈/控制对象（无硬件阶段用 mock 驱动）
_tts = TtsFeedback()
_arm_controller: ArmController | None = None


def _load_robot_cfg() -> dict:
    """读取 config/robot.yaml 的 robot 段，配置缺失时返回空字典。"""
    from src.qiming.utils.config import load_yaml

    try:
        cfg = load_yaml("config/robot.yaml") or {}
    except Exception:  # noqa: BLE001 - 配置缺失退回默认值
        return {}
    return cfg.get("robot", cfg)


def _get_arm_controller() -> ArmController:
    """懒加载机械臂控制器，避免 import 阶段就触碰硬件配置。"""
    global _arm_controller
    if _arm_controller is None:
        _arm_controller = ArmController.from_config(_load_robot_cfg())
    return _arm_controller


def create_default_registry() -> ToolRegistry:
    """创建注册好全部内置工具的注册表。"""
    registry = ToolRegistry()

    # ---------- 机械臂 ----------
    @registry.tool(
        name="query_robot_status",
        description="查询机械臂当前的工作状态，返回是否空闲、当前阶段、驱动模式等信息。",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    def query_robot_status() -> dict:
        snapshot = status_hub.snapshot()
        return {
            "status": snapshot["state"],  # 兼容旧字段名
            "state": snapshot["state"],
            "phase": snapshot["phase"],
            "driver": snapshot["driver"],
            "target": snapshot["target"],
            "updated_at": snapshot["updated_at"],
        }

    @registry.tool(
        name="emergency_stop",
        description="紧急停止机械臂的一切动作，遇到危险或用户要求停止时调用。",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    def emergency_stop() -> dict:
        # 急停动作先打开夹爪释放物体，避免夹持物坠落
        _get_arm_controller().stop()
        return {"stopped": True, "message": "机械臂已紧急停止"}

    @registry.tool(
        name="execute_grasp",
        description="让机械臂抓取一个物体并递到用户手边。",
        parameters={
            "type": "object",
            "properties": {
                "object_name": {"type": "string", "description": "要抓取的物体名称，如 水杯"},
                "x": {"type": "number", "description": "抓取点 x 坐标（可选）"},
                "y": {"type": "number", "description": "抓取点 y 坐标（可选）"},
            },
            "required": ["object_name"],
        },
    )
    def execute_grasp(object_name: str, x: float = 0.0, y: float = 0.0) -> dict:
        cfg = _load_robot_cfg()
        # 夹取用 grasp_preset，放置用 place_preset，姿态与接近高度都由预设脚本给出
        pick_pose = apply_preset(
            [x, y, DEFAULT_GRASP_HEIGHT], cfg.get("grasp_preset", DEFAULT_PRESET_NAME), phase="grasp")
        handover = list(cfg.get("handover_coords") or FALLBACK_HANDOVER)
        place_pose = apply_preset(
            handover[:3], cfg.get("place_preset", DEFAULT_PLACE_PRESET_NAME), phase="grasp")

        action = ArmAction(
            target_name=object_name,
            pick_pose=pick_pose,
            place_pose=place_pose,
            reply=f"好的，正在为您拿取{object_name}",
        )
        message = _get_arm_controller().execute(action)
        return {
            "executed": True,
            "object": object_name,
            "pick_pose": list(pick_pose),
            "place_pose": list(place_pose),
            "message": message,
        }

    # ---------- 语音反馈 ----------
    @registry.tool(
        name="speak",
        description="把一段文字通过语音播报给用户。",
        parameters={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要播报的文字"},
            },
            "required": ["text"],
        },
    )
    def speak(text: str) -> dict:
        _tts.say(text)
        return {"spoken": True, "text": text}

    # ---------- 系统 ----------
    @registry.tool(
        name="reset_system",
        description="软件层面的系统复位，恢复各子系统到初始状态。",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    def reset_system() -> dict:
        global _arm_controller
        _arm_controller = None  # 下次访问时按配置重建
        status_hub.reset()
        return {"reset": True, "message": "系统已复位（软件层）"}

    return registry
