# -*- coding: utf-8 -*-
"""本地仿真环境（算法 -> 指令准确率）的测试。"""

import math

import numpy as np

from src.qiming.arm_control.pose_presets import get_preset
from src.qiming.simulation.env import (
    SimulatedCamera,
    evaluate_conversion,
    make_extrinsic,
    sample_points,
)


def test_project_and_convert_round_trip():
    """仿真投影 + 真实坐标转换应能还原真值（误差仅来自像素/深度量化）。"""
    camera = SimulatedCamera()
    point = (120.0, -150.0, 40.0)
    u, v, depth = camera.project_point(point)

    report = evaluate_conversion([point], tolerance_mm=1.0)
    assert report.converted == 1
    assert report.accuracy == 1.0
    assert report.max_error < 1.0
    assert report.orientation_mismatches == 0


def test_points_behind_camera_are_not_projected():
    """相机后方的点不应被投影（仿真数据有效性检查）。"""
    camera = SimulatedCamera()
    # 相机位于 (200, 0, 500)，其后方是 z > 500 的方向
    assert camera.project_point((200.0, 0.0, 900.0)) is None


def test_out_of_view_points_skipped():
    """视野外的目标不参与准确率统计。"""
    camera = SimulatedCamera()
    far_point = (-900.0, -900.0, 0.0)
    assert camera.project_point(far_point) is None

    report = evaluate_conversion([far_point])
    assert report.total == 1
    assert report.converted == 0
    assert report.accuracy == 0.0


def test_accuracy_over_sample_points():
    """批量采样点的输出准确率应达标，且姿态与预设一致。"""
    points = sample_points(120, seed=7)
    report = evaluate_conversion(points, preset_name="top_down", tolerance_mm=1.0)

    assert report.converted > 0
    assert report.accuracy >= 0.98, f"准确率过低: {report.to_dict()}"
    assert report.mean_error < 0.5
    assert report.orientation_mismatches == 0


def test_depth_image_rendering():
    """深度图只在目标像素处写入相机系深度。"""
    camera = SimulatedCamera()
    point = (100.0, -100.0, 50.0)
    u, v, depth = camera.project_point(point)
    image = camera.render_depth([point])

    assert image.shape == (camera.height, camera.width)
    assert int(image[int(round(v)), int(round(u))]) == int(round(depth))
    assert int(image.sum()) == int(round(depth))  # 只有该像素非零


def test_extrinsic_is_rigid():
    """生成的外参旋转部分必须是正交阵且行列式为 1。"""
    T_BC = make_extrinsic()
    rotation = T_BC[:3, :3]
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
    assert math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-9)
