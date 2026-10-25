# -*- coding: utf-8 -*-
"""本地仿真环境（任务：搭建本地仿真测试环境，验证算法到指令的输出准确率）。

无需相机与实机，用几何仿真复现「基坐标真值点 -> 相机成像 -> 深度图 ->
坐标转换 -> 机械臂指令」全链路，输出误差分布与准确率。
"""

from src.qiming.simulation.env import (
    AccuracyReport,
    SimulatedCamera,
    SimulatedDepthCamera,
    evaluate_conversion,
    make_extrinsic,
    sample_points,
)

__all__ = [
    "AccuracyReport",
    "SimulatedCamera",
    "SimulatedDepthCamera",
    "evaluate_conversion",
    "make_extrinsic",
    "sample_points",
]
