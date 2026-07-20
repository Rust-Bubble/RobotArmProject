
"""真实 MyCobot 驱动预留。

后续接入真实硬件时，可以在这里封装 pymycobot。
"""

from __future__ import annotations

from src.qiming.models import Pose


class MyCobotDriver:
    def __init__(self, robot_cfg: dict) -> None:
        self.robot_cfg = robot_cfg

    def move_to(self, pose: Pose) -> None:
        raise NotImplementedError("这里接入 pymycobot 的 send_coords 或 send_angles。")

    def open_gripper(self) -> None:
        raise NotImplementedError("这里接入 pymycobot 的夹爪打开指令。")

    def close_gripper(self) -> None:
        raise NotImplementedError("这里接入 pymycobot 的夹爪闭合指令。")
