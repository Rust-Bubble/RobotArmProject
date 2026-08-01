"""机械臂动作执行入口，负责把抓取动作转换为夹爪和移动指令。"""

from __future__ import annotations

from src.qiming.arm_control.driver import ArmDriver
from src.qiming.arm_control.mock_driver import MockArmDriver
from src.qiming.arm_control.mycobot_driver import MyCobotDriver
from src.qiming.models import ArmAction


class ArmController:
    """机械臂执行层。"""

    def __init__(self, driver: ArmDriver) -> None:
        self.driver = driver

    @classmethod
    def from_config(cls, robot_cfg: dict) -> "ArmController":
        mode = robot_cfg.get("mode", "mock")
        if mode == "real":
            return cls(MyCobotDriver(robot_cfg))
        return cls(MockArmDriver(robot_cfg))

    def execute(self, action: ArmAction) -> str:
        self.driver.open_gripper()
        self.driver.move_to(action.pick_pose)
        self.driver.close_gripper()
        self.driver.move_to(action.place_pose)
        self.driver.open_gripper()
        return action.reply
