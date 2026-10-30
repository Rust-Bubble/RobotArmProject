# -*- coding: utf-8 -*-
"""预设机械臂运动脚本（不依赖视觉模块）。

每个脚本是一串固定的「移动到位姿 / 开合夹爪」步骤，位姿全部写死，
不接收外部计算的坐标，也不调用相机、手眼标定或坐标转换。

用途：在接入视觉之前，先单独验证「用户指令 -> 模型选脚本 -> 机械臂执行」链路。

用法:
    from src.qiming.arm_control.motion_scripts import get_script, list_scripts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

Pose = Tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class MotionStep:
    """脚本中的一步：移动到位姿，或开/合夹爪。"""

    kind: str  # "move" | "open" | "close"
    label: str
    pose: Pose | None = None


@dataclass(frozen=True)
class MotionScript:
    """一个预设运动脚本。"""

    name: str
    label: str
    description: str  # 给模型看的一句话说明
    steps: Tuple[MotionStep, ...]


def _move(label: str, pose: Pose) -> MotionStep:
    return MotionStep(kind="move", label=label, pose=pose)


def _open(label: str = "打开夹爪") -> MotionStep:
    return MotionStep(kind="open", label=label)


def _close(label: str = "闭合夹爪") -> MotionStep:
    return MotionStep(kind="close", label=label)


# 俯视姿态 (rx, ry, rz) = (0, 180, 90)，位置均为基坐标 mm
_TOP_DOWN = (0.0, 180.0, 90.0)


def _pose(x: float, y: float, z: float) -> Pose:
    rx, ry, rz = _TOP_DOWN
    return (x, y, z, rx, ry, rz)


_SCRIPTS: List[MotionScript] = [
    MotionScript(
        name="gripper_test",
        label="夹爪开合",
        description="仅测试夹爪：打开 -> 闭合 -> 打开，机械臂不移动。",
        steps=(_open("打开夹爪"), _close(), _open("再次打开")),
    ),
    MotionScript(
        name="home_demo",
        label="回到工作位",
        description="从当前位置移动到预设的工作位置，姿态垂直俯视。",
        steps=(_move("移动到工作位", _pose(120.0, -120.0, 150.0)),),
    ),
    MotionScript(
        name="wave_demo",
        label="左右挥手",
        description="在工作位附近左右移动三次，模拟挥手示意。",
        steps=(
            _move("移动到工作位", _pose(120.0, -120.0, 180.0)),
            _move("向左", _pose(40.0, -120.0, 180.0)),
            _move("向右", _pose(200.0, -120.0, 180.0)),
            _move("向左", _pose(40.0, -120.0, 180.0)),
            _move("回到工作位", _pose(120.0, -120.0, 180.0)),
        ),
    ),
    MotionScript(
        name="pick_place_demo",
        label="抓取放置演示",
        description="在固定坐标演示完整的抓取与放置：靠近、下降、夹取、抬起、移动、放下、撤离。",
        steps=(
            _open(),
            _move("移动到抓取点上方", _pose(120.0, -120.0, 130.0)),
            _move("下降到抓取点", _pose(120.0, -120.0, 40.0)),
            _close(),
            _move("抬起物体", _pose(120.0, -120.0, 130.0)),
            _move("移动到放置点上方", _pose(80.0, -220.0, 240.0)),
            _move("下降到放置点", _pose(80.0, -220.0, 190.0)),
            _open("松开夹爪"),
            _move("撤离放置点", _pose(80.0, -220.0, 240.0)),
        ),
    ),
    MotionScript(
        name="square_demo",
        label="方形轨迹",
        description="沿一个固定矩形轨迹移动一圈，用于验证多段位姿连续运动。",
        steps=(
            _move("移动到起点", _pose(80.0, -200.0, 200.0)),
            _move("第一段", _pose(200.0, -200.0, 200.0)),
            _move("第二段", _pose(200.0, -60.0, 200.0)),
            _move("第三段", _pose(80.0, -60.0, 200.0)),
            _move("回到起点", _pose(80.0, -200.0, 200.0)),
        ),
    ),
]

SCRIPTS: Dict[str, MotionScript] = {script.name: script for script in _SCRIPTS}


def list_scripts() -> List[MotionScript]:
    """返回全部脚本。"""
    return list(_SCRIPTS)


def get_script(name: str) -> MotionScript:
    """按名称获取脚本，未找到时抛出 KeyError。

    为方便用户/模型，除英文 name 外也接受中文 label（忽略大小写）。
    """
    key = (name or "").strip()
    if key in SCRIPTS:
        return SCRIPTS[key]

    lowered = key.lower()
    for script in _SCRIPTS:
        if key == script.label or lowered == script.name.lower():
            return script

    raise KeyError(
        f"未知的运动脚本: {name}，可选: "
        + ", ".join(f"{s.name}（{s.label}）" for s in _SCRIPTS))


if __name__ == "__main__":
    for script in list_scripts():
        print(f"- {script.name}（{script.label}）: {script.description}  [{len(script.steps)} 步]")
