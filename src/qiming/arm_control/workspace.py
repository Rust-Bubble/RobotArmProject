# -*- coding: utf-8 -*-
"""机械臂工作空间定义与位姿校验。

坐标转换层（clamp 到安全范围）与控制网关（越界直接拒绝）共用同一份边界，
避免两处硬编码出现偏差。

位姿约定 [x, y, z, rx, ry, rz]，位置单位 mm，姿态单位 度。
"""

from __future__ import annotations

import math
from typing import Sequence

# myCobot 320 安全工作空间（单位 mm / 度），与机械臂产品手册的坐标范围一致
X_LIMITS = (-350.0, 350.0)
Y_LIMITS = (-350.0, 350.0)
Z_LIMITS = (-41.0, 523.9)
ANGLE_LIMITS = (-180.0, 180.0)

POSE_LENGTH = 6


class PoseValidationError(ValueError):
    """位姿不合法（维度、类型、数值或越界）。"""


def clamp_pose(pose: Sequence[float]) -> list[float]:
    """把位姿限制到安全工作空间内（用于视觉链路，宁可裁边也不中断任务）。"""
    x, y, z, rx, ry, rz = (float(v) for v in pose[:POSE_LENGTH])
    return [
        min(max(x, X_LIMITS[0]), X_LIMITS[1]),
        min(max(y, Y_LIMITS[0]), Y_LIMITS[1]),
        min(max(z, Z_LIMITS[0]), Z_LIMITS[1]),
        min(max(rx, ANGLE_LIMITS[0]), ANGLE_LIMITS[1]),
        min(max(ry, ANGLE_LIMITS[0]), ANGLE_LIMITS[1]),
        min(max(rz, ANGLE_LIMITS[0]), ANGLE_LIMITS[1]),
    ]


def validate_pose(pose: Sequence[float], field: str = "pose") -> list[float]:
    """校验位姿并返回 float 列表；不合法时抛 PoseValidationError。

    与 clamp_pose 不同，这里对越界坐标直接拒绝——控制网关收到的是外部系统
    下发的指令，静默裁剪会让上层误以为指令被完整执行，因此宁可显式报错。
    """
    if isinstance(pose, (str, bytes)) or not isinstance(pose, Sequence):
        raise PoseValidationError(f"{field} 必须是长度为 6 的数组")
    if len(pose) != POSE_LENGTH:
        raise PoseValidationError(
            f"{field} 需要 {POSE_LENGTH} 个分量，实际 {len(pose)} 个")

    values: list[float] = []
    for index, raw in enumerate(pose):
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise PoseValidationError(f"{field}[{index}] 不是数值: {raw!r}") from None
        if not math.isfinite(value):
            raise PoseValidationError(f"{field}[{index}] 不是有限数值: {raw!r}")
        values.append(value)

    x, y, z, rx, ry, rz = values
    for label, value, limits in (
        ("x", x, X_LIMITS),
        ("y", y, Y_LIMITS),
        ("z", z, Z_LIMITS),
        ("rx", rx, ANGLE_LIMITS),
        ("ry", ry, ANGLE_LIMITS),
        ("rz", rz, ANGLE_LIMITS),
    ):
        if not limits[0] <= value <= limits[1]:
            raise PoseValidationError(
                f"{field}.{label}={value} 超出工作空间 [{limits[0]}, {limits[1]}]")

    return values
