"""机械臂动作执行入口：按末端姿态预设展开抓取流程，并逐阶段上报运行状态。"""

from __future__ import annotations

from src.qiming.arm_control.driver import ArmDriver
from src.qiming.arm_control.grasp import GraspFlow
from src.qiming.arm_control.mock_driver import MockArmDriver
from src.qiming.arm_control.mycobot_driver import MyCobotDriver
from src.qiming.arm_control.pose_presets import (
    DEFAULT_PLACE_PRESET_NAME,
    DEFAULT_PRESET_NAME,
)
from src.qiming.arm_control.status import status_hub
from src.qiming.models import ArmAction


class ArmController:
    """机械臂执行层：把抓取动作展开成 GraspFlow 步骤并逐步上报状态。"""

    def __init__(self, driver: ArmDriver, robot_cfg: dict | None = None) -> None:
        cfg = robot_cfg or {}
        self.driver = driver
        # 「怎么抓 / 怎么放」由末端姿态预设决定，见 pose_presets.py
        self.flow = GraspFlow(
            driver,
            grasp_preset=cfg.get("grasp_preset", DEFAULT_PRESET_NAME),
            place_preset=cfg.get("place_preset", DEFAULT_PLACE_PRESET_NAME),
        )
        status_hub.configure(type(driver).__name__)

    @classmethod
    def from_config(cls, robot_cfg: dict) -> "ArmController":
        mode = robot_cfg.get("mode", "mock")
        driver = MyCobotDriver(robot_cfg) if mode == "real" else MockArmDriver(robot_cfg)
        return cls(driver, robot_cfg)

    def execute(self, action: ArmAction) -> str:
        """执行一次「抓取 -> 递送」，每个步骤开始前上报当前阶段。"""
        status_hub.update(state="busy", target=action.target_name, phase="开始执行")
        try:
            for step in self.flow.build_steps(action):
                status_hub.update(phase=step.label)
                step.run()
        except Exception:  # noqa: BLE001 - 异常时也要复位状态，避免卡片一直停在“执行中”
            status_hub.update(state="idle", phase="执行中断", target="")
            raise
        status_hub.update(state="idle", phase="任务完成", target="")
        return action.reply

    def stop(self) -> None:
        """急停：先打开夹爪释放物体，再把状态切到已急停。"""
        self.driver.open_gripper()
        status_hub.update(state="estop", phase="已急停", target="")
