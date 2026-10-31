# -*- coding: utf-8 -*-
"""本地仿真环境：无需相机与实机即可验证「视觉坐标 -> 机械臂指令」链路的准确率。

仿真链路刻意与真实流程保持一致：
    已知基坐标三维点（真值）
        -> 用相机内参 K 与手眼外参 T_BC 投影成像素（模拟相机成像）
        -> 生成该像素处的深度图（模拟深度相机）
        -> CoordinateConverter.pixel_to_robot_coords（真实坐标转换算法）
        -> 目标位姿指令
    最后把指令位置与真值比较，得到误差分布与准确率。

关键点：深度图由 T_BC 反向渲染，坐标转换又把像素按 T_BC 正向映射回基坐标，
因此这条链路是自洽的——误差只来自像素取整与深度量化，可用来把「算法写错」
（坐标系方向、像素行列、单位混用）与「硬件噪声」区分开。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from src.qiming.arm_control.pose_presets import DEFAULT_PRESET_NAME, get_preset
from src.qiming.vision.coordinate import CoordinateConverter
from src.qiming.vision.depth_camera import DepthCamera

# 640x480 深度相机的典型内参（fx=fy=600, 主点在图像中心）
DEFAULT_INTRINSIC = np.array([
    [600.0, 0.0, 320.0],
    [0.0, 600.0, 240.0],
    [0.0, 0.0, 1.0],
])

# 桌面作业面的典型相机安装位置：位于工作区斜上方、俯视基坐标原点前侧
DEFAULT_CAMERA_POSITION = (200.0, 0.0, 500.0)


def look_at(camera_position: Sequence[float], target: Sequence[float]) -> np.ndarray:
    """构造位于 camera_position、光轴指向 target 的相机姿态（相机 -> 基坐标）。"""
    z_axis = np.asarray(target, dtype=float) - np.asarray(camera_position, dtype=float)
    z_axis /= np.linalg.norm(z_axis)
    up = np.array([1.0, 0.0, 0.0])  # 机械臂坐标系 z 轴朝上，用 x 轴作参考方向避免退化
    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.cross(np.array([0.0, 1.0, 0.0]), z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def make_extrinsic(
    position: Sequence[float] = DEFAULT_CAMERA_POSITION,
    target: Sequence[float] = (0.0, -150.0, 0.0),
) -> np.ndarray:
    """生成 Eye-to-Hand 外参 T_BC（相机在基坐标系下的位姿，平移单位 mm）。"""
    T_BC = np.eye(4)
    T_BC[:3, :3] = look_at(position, target)
    T_BC[:3, 3] = np.asarray(position, dtype=float)
    return T_BC


class SimulatedCamera:
    """仿真深度相机：把基坐标点投影成像素，并渲染深度图。"""

    def __init__(
        self,
        intrinsic: np.ndarray | None = None,
        T_BC: np.ndarray | None = None,
        width: int = 640,
        height: int = 480,
        min_depth: float = 50.0,
    ) -> None:
        self.intrinsic = np.asarray(intrinsic if intrinsic is not None else DEFAULT_INTRINSIC, dtype=float)
        self.T_BC = np.asarray(T_BC if T_BC is not None else make_extrinsic(), dtype=float)
        self.width = width
        self.height = height
        self.min_depth = min_depth
        self._T_BC_inv = np.linalg.inv(self.T_BC)

    def to_camera(self, point_base: Sequence[float]) -> np.ndarray:
        """基坐标点 (mm) -> 相机坐标 (mm)。"""
        point = np.array([point_base[0], point_base[1], point_base[2], 1.0], dtype=float)
        return self._T_BC_inv @ point

    def project_point(self, point_base: Sequence[float]) -> tuple[float, float, float] | None:
        """基坐标点投影为像素，返回 (u, v, depth_mm)；不可见时返回 None。"""
        p_cam = self.to_camera(point_base)
        if p_cam[2] <= self.min_depth:
            return None

        fx, fy = self.intrinsic[0, 0], self.intrinsic[1, 1]
        cx, cy = self.intrinsic[0, 2], self.intrinsic[1, 2]
        u = fx * p_cam[0] / p_cam[2] + cx
        v = fy * p_cam[1] / p_cam[2] + cy
        if not (0 <= u < self.width and 0 <= v < self.height):
            return None
        return float(u), float(v), float(p_cam[2])

    def render_depth(self, points: Iterable[Sequence[float]]) -> np.ndarray:
        """渲染深度图：在每个可见点的像素位置写入其相机系深度（单位 mm）。"""
        depth = np.zeros((self.height, self.width), dtype=np.uint16)
        for point in points:
            projected = self.project_point(point)
            if projected is None:
                continue
            u, v, z = projected
            ui, vi = int(round(u)), int(round(v))
            if 0 <= ui < self.width and 0 <= vi < self.height:
                depth[vi, ui] = int(round(z))
        return depth


class SimulatedDepthCamera(DepthCamera):
    """把仿真深度图注入真实 DepthCamera 接口，坐标转换无需感知仿真。"""

    def __init__(self, intrinsic: np.ndarray, depth_image: np.ndarray) -> None:
        super().__init__({})
        self.intrinsic_matrix = np.asarray(intrinsic, dtype=float)
        self.dist_coeffs = np.zeros((5, 1))
        self._depth_image = depth_image
        self._color_image = np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.uint8)

    def set_depth_image(self, depth_image: np.ndarray) -> None:
        self._depth_image = depth_image

    def get_frames(self):
        return self._color_image, self._depth_image


def sample_points(count: int, seed: int = 0) -> list[tuple[float, float, float]]:
    """在机械臂工作区随机采样若干个位于桌面高度的目标点（单位 mm）。"""
    rng = np.random.default_rng(seed)
    points: list[tuple[float, float, float]] = []
    for _ in range(count):
        x = float(rng.uniform(-150.0, 250.0))
        y = float(rng.uniform(-300.0, 50.0))
        z = float(rng.uniform(0.0, 80.0))
        points.append((x, y, z))
    return points


@dataclass
class AccuracyReport:
    """一次「算法到指令」评估的结果。"""

    total: int
    converted: int
    errors_mm: list[float]
    orientation_mismatches: int
    tolerance_mm: float

    @property
    def conversion_rate(self) -> float:
        """能成功转换的比例（目标是否在视野内）。"""
        return self.converted / self.total if self.total else 0.0

    @property
    def within_tolerance(self) -> int:
        return sum(1 for e in self.errors_mm if e <= self.tolerance_mm)

    @property
    def accuracy(self) -> float:
        """在容差内的比例，即「算法到指令」的输出准确率。"""
        return self.within_tolerance / self.converted if self.converted else 0.0

    @property
    def mean_error(self) -> float:
        return float(np.mean(self.errors_mm)) if self.errors_mm else 0.0

    @property
    def p95_error(self) -> float:
        return float(np.percentile(self.errors_mm, 95)) if self.errors_mm else 0.0

    @property
    def max_error(self) -> float:
        return max(self.errors_mm) if self.errors_mm else 0.0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "converted": self.converted,
            "conversion_rate": round(self.conversion_rate, 4),
            "tolerance_mm": self.tolerance_mm,
            "within_tolerance": self.within_tolerance,
            "accuracy": round(self.accuracy, 4),
            "mean_error_mm": round(self.mean_error, 4),
            "p95_error_mm": round(self.p95_error, 4),
            "max_error_mm": round(self.max_error, 4),
            "orientation_mismatches": self.orientation_mismatches,
        }


def evaluate_conversion(
    points: Sequence[Sequence[float]],
    preset_name: str = DEFAULT_PRESET_NAME,
    intrinsic: np.ndarray | None = None,
    T_BC: np.ndarray | None = None,
    tolerance_mm: float = 1.0,
) -> AccuracyReport:
    """跑完整链路：真值点 -> 仿真成像 -> 坐标转换 -> 指令，统计误差与准确率。

    参数:
        points: 基坐标真值点列表，单位 mm
        preset_name: 用于生成指令姿态的末端姿态预设
        tolerance_mm: 判定「准确」的位置误差阈值
    """
    camera = SimulatedCamera(intrinsic=intrinsic, T_BC=T_BC)
    depth_camera = SimulatedDepthCamera(camera.intrinsic, np.zeros((camera.height, camera.width), np.uint16))
    converter = CoordinateConverter(depth_camera, arm_driver=None)
    converter.set_T_BC(camera.T_BC)
    converter.set_grasp_preset(preset_name)
    expected_orientation = tuple(get_preset(preset_name).orientation)

    errors: list[float] = []
    converted = 0
    orientation_mismatches = 0

    for point in points:
        projected = camera.project_point(point)
        if projected is None:
            continue
        u, v, _depth = projected
        depth_camera.set_depth_image(camera.render_depth([point]))
        try:
            pose = converter.pixel_to_robot_coords(u, v)
        except RuntimeError:
            continue
        converted += 1
        errors.append(math.dist(pose[:3], list(point)))
        if tuple(pose[3:]) != expected_orientation:
            orientation_mismatches += 1

    return AccuracyReport(
        total=len(points),
        converted=converted,
        errors_mm=errors,
        orientation_mismatches=orientation_mismatches,
        tolerance_mm=tolerance_mm,
    )
