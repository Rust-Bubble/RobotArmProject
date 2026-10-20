# -*- coding: utf-8 -*-
"""抓取流程：把一次「抓取 -> 递送」拆成按末端姿态预设执行的动作步骤。

分工:
    pose_presets  只回答「怎么抓 / 怎么放」——末端姿态与接近抬升高度；
    GraspFlow     回答「按什么顺序动」——靠近 -> 下降 -> 夹取 -> 抬起 -> 递送 -> 下降 -> 放置 -> 撤离。

由于 ArmAction 里的 pick_pose / place_pose 已经带了姿态（由坐标转换或工具层用预设生成），
这里只需按对应预设的 approach_offset 把目标点沿 z 轴抬高，即可得到接近/撤离位姿。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence

from src.qiming.arm_control.pose_presets import (
    DEFAULT_PLACE_PRESET_NAME,
    DEFAULT_PRESET_NAME,
    get_preset,
)
from src.qiming.models import ArmAction, Pose


@dataclass(frozen=True)
class GraspStep:
    """一个可执行的动作步骤：标签用于状态展示，run 负责真正下发指令。"""

    label: str
    run: Callable[[], None]


def _approach_pose(pose: Sequence[float], preset_name: str) -> Pose:
    """按预设的接近抬升高度把目标位姿沿 z 轴抬高，得到接近/撤离位姿。"""
    preset = get_preset(preset_name)
    x, y, z, rx, ry, rz = (float(v) for v in pose[:6])
    return (x, y, z + preset.approach_offset, rx, ry, rz)


class GraspFlow:
    """抓取流程：按预设生成并执行动作步骤序列。"""

    def __init__(
        self,
        driver,
        grasp_preset: str = DEFAULT_PRESET_NAME,
        place_preset: str = DEFAULT_PLACE_PRESET_NAME,
    ) -> None:
        self.driver = driver
        self.grasp_preset = grasp_preset
        self.place_preset = place_preset

    def build_steps(self, action: ArmAction) -> List[GraspStep]:
        """生成「抓取 -> 递送」的完整步骤序列。"""
        driver = self.driver
        pick_approach = _approach_pose(action.pick_pose, self.grasp_preset)
        place_approach = _approach_pose(action.place_pose, self.place_preset)
        pick_pose: Pose = tuple(float(v) for v in action.pick_pose[:6])
        place_pose: Pose = tuple(float(v) for v in action.place_pose[:6])

        return [
            GraspStep("打开夹爪", driver.open_gripper),
            GraspStep("移动到抓取点上方", lambda: driver.move_to(pick_approach)),
            GraspStep("下降到抓取点", lambda: driver.move_to(pick_pose)),
            GraspStep("闭合夹爪", driver.close_gripper),
            GraspStep("抬起物体", lambda: driver.move_to(pick_approach)),
            GraspStep("移动到递送点上方", lambda: driver.move_to(place_approach)),
            GraspStep("下降到递送点", lambda: driver.move_to(place_pose)),
            GraspStep("松开夹爪", driver.open_gripper),
            GraspStep("撤离递送点", lambda: driver.move_to(place_approach)),
        ]

    def run(self, action: ArmAction) -> None:
        """依次执行全部步骤（不负责状态上报，由 ArmController 统一处理）。"""
        for step in self.build_steps(action):
            step.run()
