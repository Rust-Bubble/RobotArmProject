"""定义机械臂驱动接口。"""

from __future__ import annotations

from typing import Protocol

from src.qiming.models import Pose


class ArmDriver(Protocol):
    def move_to(self, pose: Pose) -> None:
        ...

    def move_joint(self, joint_id: int, angle: float) -> None:
        ...

    def open_gripper(self) -> None:
        ...

    def close_gripper(self) -> None:
        ...
