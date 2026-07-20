""" 功能：模拟机械臂驱动。"""

from __future__ import annotations

from src.qiming.models import Pose


class MockArmDriver:
    """无硬件调试阶段使用的机械臂模拟驱动。"""

    def __init__(self, robot_cfg: dict) -> None:
        self.robot_cfg = robot_cfg

    def move_to(self, pose: Pose) -> None:
        print(f"[机械臂] 移动到位姿：{pose}")

    def open_gripper(self) -> None:
        print("[机械臂] 打开夹爪")

    def close_gripper(self) -> None:
        print("[机械臂] 闭合夹爪")
