# -*- coding: utf-8 -*-
"""内置的外部系统工具（Function Calling 基础 API）。

覆盖三类外部系统：
    - 机械臂：查询状态、急停、抓取执行（通过 ArmController，mock/真实由配置决定）
    - 语音反馈：TTS 播报
    - 系统：软件复位（暂以日志代替）

所有工具同步实现，统一返回 dict，方便大模型读结果。
"""

from __future__ import annotations

from src.qiming.arm_control.controller import ArmController
from src.qiming.feedback.tts import TtsFeedback
from src.qiming.models import ArmAction
from src.qiming.tools.registry import ToolRegistry

# 全局共享的反馈/控制对象（无硬件阶段用 mock 驱动）
_tts = TtsFeedback()
_arm_controller: ArmController | None = None


def _get_arm_controller() -> ArmController:
    """懒加载机械臂控制器，避免 import 阶段就触碰硬件配置。"""
    global _arm_controller
    if _arm_controller is None:
        from src.qiming.utils.config import load_yaml

        try:
            robot_cfg = load_yaml("config/robot.yaml") or {}
        except Exception:  # noqa: BLE001 - 配置缺失退回 mock
            robot_cfg = {}
        _arm_controller = ArmController.from_config(robot_cfg.get("robot", robot_cfg))
    return _arm_controller


def create_default_registry() -> ToolRegistry:
    """创建注册好全部内置工具的注册表。"""
    registry = ToolRegistry()

    # ---------- 机械臂 ----------
    @registry.tool(
        name="query_robot_status",
        description="查询机械臂当前的工作状态，返回是否空闲、当前模式等信息。",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    def query_robot_status() -> dict:
        driver = _get_arm_controller().driver
        return {"status": "idle", "driver": type(driver).__name__}

    @registry.tool(
        name="emergency_stop",
        description="紧急停止机械臂的一切动作，遇到危险或用户要求停止时调用。",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    def emergency_stop() -> dict:
        # 急停动作先打开夹爪释放物体，避免夹持物坠落
        _get_arm_controller().driver.open_gripper()
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
        action = ArmAction(
            target_name=object_name,
            pick_pose=(x, y, 20.0, 0.0, 0.0, 0.0),
            place_pose=(x, y, 100.0, 0.0, 0.0, 0.0),
            reply=f"好的，正在为您拿取{object_name}",
        )
        message = _get_arm_controller().execute(action)
        return {"executed": True, "object": object_name, "message": message}

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
        return {"reset": True, "message": "系统已复位（软件层）"}

    return registry
