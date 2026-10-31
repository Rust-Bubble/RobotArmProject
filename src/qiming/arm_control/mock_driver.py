""" 功能：模拟机械臂驱动。"""

from __future__ import annotations

from src.qiming.models import Pose


class MockArmDriver:
    """无硬件调试阶段使用的机械臂模拟驱动。"""

    def __init__(self, robot_cfg: dict) -> None:
        self.robot_cfg = robot_cfg

    def move_to(self, pose: Pose) -> None:
        print(f"[机械臂] 移动到位姿：{pose}")

    def move_joint(self, joint_id: int, angle: float) -> None:
        print(f"[机械臂] 第 {joint_id} 关节转到 {angle} 度")

    def open_gripper(self) -> None:
        print("[机械臂] 打开夹爪")

    def close_gripper(self) -> None:
        print("[机械臂] 闭合夹爪")

    def set_led_color(self, r: int, g: int, b: int) -> None:
        print(f"[机械臂] 灯板颜色设置为 RGB({r}, {g}, {b})")
