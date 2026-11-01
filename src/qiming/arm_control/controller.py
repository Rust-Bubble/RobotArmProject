"""机械臂动作执行入口：按末端姿态预设展开抓取流程，并逐阶段上报运行状态。"""

from __future__ import annotations

import time

from src.qiming.arm_control.driver import ArmDriver
from src.qiming.arm_control.grasp import GraspFlow
from src.qiming.arm_control.mock_driver import MockArmDriver
from src.qiming.arm_control.motion_scripts import get_script
from src.qiming.arm_control.mycobot_driver import MyCobotDriver
from src.qiming.arm_control.pose_presets import (
    DEFAULT_PLACE_PRESET_NAME,
    DEFAULT_PRESET_NAME,
)
from src.qiming.arm_control.status import status_hub
from src.qiming.models import ArmAction

# 灯板变色脚本中每种颜色的停留时间（秒）
LED_HOLD_SECONDS = 3.0
# 相邻机械臂动作之间的间隔（秒），避免连续动作过快
MOTION_INTERVAL_SECONDS = 1.0


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

    def move_to_pose(self, pose, label: str = "移动到目标坐标", target: str = "") -> None:
        """按坐标指令直接移动到位姿（控制网关的 move 指令走这里）。

        与 execute 共用同一套状态上报与异常复位逻辑，保证网关下发的单点移动
        也能在前端状态卡片上看到执行过程。
        """
        status_hub.update(state="busy", target=target, phase=label)
        try:
            self.driver.move_to(tuple(float(v) for v in pose))
        except Exception:  # noqa: BLE001
            status_hub.update(state="idle", phase="执行中断", target="")
            raise
        status_hub.update(state="idle", phase="任务完成", target="")

    def stop(self) -> None:
        """急停：先打开夹爪释放物体，再把状态切到已急停。"""
        self.driver.open_gripper()
        status_hub.update(state="estop", phase="已急停", target="")

    def set_gripper(self, action: str) -> dict:
        """按 Agent 指定的动作开合夹爪（不依赖预设脚本）。

        参数:
            action: "open" 打开夹爪 / "close" 闭合夹爪
        """
        if action not in ("open", "close"):
            raise ValueError(f"不支持的夹爪动作：{action}，仅支持 open / close")
        label = "打开夹爪" if action == "open" else "闭合夹爪"
        status_hub.update(state="busy", target="夹爪", phase=label)
        try:
            if action == "open":
                self.driver.open_gripper()
            else:
                self.driver.close_gripper()
        except Exception:  # noqa: BLE001
            status_hub.update(state="idle", phase="执行中断", target="")
            raise
        status_hub.update(state="idle", phase="任务完成", target="")
        return {"action": action, "message": "夹爪已打开" if action == "open" else "夹爪已闭合"}

    def set_led_color(self, r: int, g: int, b: int) -> dict:
        """按 Agent 传入的 RGB 设置灯板颜色。"""
        for value in (r, g, b):
            if not 0 <= value <= 255:
                raise ValueError(f"RGB 取值必须在 0~255 之间，收到 {r},{g},{b}")
        status_hub.update(state="busy", target="灯板", phase=f"灯板变色 RGB({r},{g},{b})")
        try:
            self.driver.set_led_color(r, g, b)
        except Exception:  # noqa: BLE001
            status_hub.update(state="idle", phase="执行中断", target="")
            raise
        status_hub.update(state="idle", phase="任务完成", target="")
        return {"r": r, "g": g, "b": b, "message": f"灯板颜色已设置为 RGB({r}, {g}, {b})"}

    def run_motion_script(self, name: str) -> dict:
        """执行预设运动脚本（纯运动与夹爪开合，不依赖视觉）。

        参数:
            name: 脚本名称，见 motion_scripts.py

        返回:
            {"script": 名称, "label": 中文名, "steps": 步数}
        """
        script = get_script(name)  # KeyError 由调用方转换为用户可读错误
        status_hub.update(state="busy", target=script.label, phase="开始执行脚本")
        try:
            for step in script.steps:
                status_hub.update(phase=step.label)
                if step.kind == "move" and step.pose is not None:
                    self.driver.move_to(step.pose)
                elif step.kind == "joint" and step.joint is not None:
                    self.driver.move_joint(*step.joint)
                elif step.kind == "open":
                    self.driver.open_gripper()
                elif step.kind == "close":
                    self.driver.close_gripper()
                elif step.kind == "led" and step.color is not None:
                    self.driver.set_led_color(*step.color)
                    time.sleep(LED_HOLD_SECONDS)
                if step.kind != "led":
                    time.sleep(MOTION_INTERVAL_SECONDS)
        except Exception:  # noqa: BLE001
            status_hub.update(state="idle", phase="执行中断", target="")
            raise
        status_hub.update(state="idle", phase="脚本完成", target="")
        return {"script": script.name, "label": script.label, "steps": len(script.steps)}
