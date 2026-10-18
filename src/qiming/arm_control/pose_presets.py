"""末端姿态预设脚本：为夹取、放置等典型抓取动作提供可复用的末端姿态。

设计思路是把「怎么抓」和「抓哪里」解耦：
    - 预设只固定末端的姿态角 (rx, ry, rz) 以及接近/抓取的高度偏移；
    - 目标点 (x, y, z) 由视觉坐标转换给出；
    - 两者组合即可得到机械臂可直接执行的完整位姿。

姿态沿用机械臂位姿约定 [x, y, z, rx, ry, rz]（欧拉角 Z-Y-X，单位 度）。

用法:
    from src.qiming.arm_control.pose_presets import get_preset, apply_preset

    grasp = apply_preset([120.0, -120.0, 60.0], "top_down", phase="grasp")
    above = apply_preset([120.0, -120.0, 60.0], "top_down", phase="approach")

直接运行可查看所有预设:
    python -m src.qiming.arm_control.pose_presets
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

Pose = Tuple[float, float, float, float, float, float]
Orientation = Tuple[float, float, float]


@dataclass(frozen=True)
class PosePreset:
    """一个末端姿态预设。"""

    name: str
    label: str
    description: str
    orientation: Orientation  # 末端欧拉角 (rx, ry, rz)，单位 度
    approach_offset: float = 70.0  # 接近阶段相对目标点在 z 方向的抬升量 (mm)
    grasp_offset: float = 0.0  # 抓取阶段相对目标点的 z 偏移 (mm)
    grip_value: int = 30  # 夹爪闭合值 (越小夹得越紧)


# ============ 预设定义 ============
_PRESETS: List[PosePreset] = [
    PosePreset(
        name="top_down",
        label="垂直俯视夹取",
        description="夹爪竖直向下，适用于桌面平放的常规物体，最稳定。",
        orientation=(0.0, 180.0, 90.0),
        approach_offset=70.0,
        grasp_offset=0.0,
        grip_value=30,
    ),
    PosePreset(
        name="top_down_rotate",
        label="垂直俯视旋转夹取",
        description="俯视下压但夹爪相对 top_down 旋转 90 度，适配细长物体的横向夹持。",
        orientation=(0.0, 180.0, 0.0),
        approach_offset=70.0,
        grasp_offset=0.0,
        grip_value=30,
    ),
    PosePreset(
        name="side",
        label="水平侧向夹取",
        description="夹爪水平伸入，适用于较高或背面不便从上方接触的物体。",
        orientation=(0.0, 90.0, 0.0),
        approach_offset=50.0,
        grasp_offset=0.0,
        grip_value=35,
    ),
    PosePreset(
        name="top_down_place",
        label="垂直俯视放置",
        description="放置动作用姿态，垂直下放后松开夹爪，接近高度更小以平稳落位。",
        orientation=(0.0, 180.0, 90.0),
        approach_offset=50.0,
        grasp_offset=10.0,
        grip_value=80,
    ),
]

PRESETS: Dict[str, PosePreset] = {preset.name: preset for preset in _PRESETS}
DEFAULT_PRESET_NAME = "top_down"


def get_preset(name: str) -> PosePreset:
    """按名称获取预设，未找到时抛出 KeyError。"""
    if name not in PRESETS:
        raise KeyError(f"未知的末端姿态预设: {name}，可选: {', '.join(PRESETS)}")
    return PRESETS[name]


def list_presets() -> List[PosePreset]:
    """返回全部预设。"""
    return list(_PRESETS)


def apply_preset(
    position: Sequence[float],
    preset_name: str = DEFAULT_PRESET_NAME,
    phase: str = "grasp",
) -> Pose:
    """把预设姿态作用到目标位置，得到完整末端位姿。

    参数:
        position: 目标点 [x, y, z]，单位 mm；多余分量会被忽略
        preset_name: 预设名称，见 PRESETS
        phase: "grasp" 抓取位姿 / "approach" 接近（抬升）位姿

    返回:
        [x, y, z, rx, ry, rz] 完整位姿
    """
    preset = get_preset(preset_name)
    if phase not in ("grasp", "approach"):
        raise ValueError('phase 只能是 "grasp" 或 "approach"')

    x, y, z = (float(v) for v in position[:3])
    offset = preset.approach_offset if phase == "approach" else preset.grasp_offset
    rx, ry, rz = preset.orientation
    return (x, y, z + offset, rx, ry, rz)


if __name__ == "__main__":
    print("末端姿态预设列表：\n")
    for item in list_presets():
        print(f"- {item.name} ({item.label})")
        print(f"    姿态(rx, ry, rz): {item.orientation}   接近抬升: {item.approach_offset}mm"
              f"   抓取偏移: {item.grasp_offset}mm   夹爪: {item.grip_value}")
        print(f"    说明: {item.description}")
    print("\n示例（目标点 [120, -120, 60]，top_down）：")
    print("    抓取位姿:", apply_preset([120.0, -120.0, 60.0], "top_down", "grasp"))
    print("    接近位姿:", apply_preset([120.0, -120.0, 60.0], "top_down", "approach"))
