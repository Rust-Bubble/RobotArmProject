from __future__ import annotations

import numpy as np
from typing import List

from src.qiming.models import ArmAction, MllmDecision, Pose
from src.qiming.vision.depth_camera import DepthCamera
from src.qiming.arm_control.mycobot_driver import MyCobotDriver


class CoordinateConverter:
    def __init__(self, depth_camera: DepthCamera, arm_driver: MyCobotDriver):
        self.depth_camera = depth_camera
        self.arm_driver = arm_driver

    def pixel_to_robot_coords(self, cx, cy):
        color_img, depth_img = self.depth_camera.get_frames()
        if depth_img is None:
            raise RuntimeError("深度图未获取到，请先打开相机")

        depth = depth_img[int(cy), int(cx)]

        K = self.depth_camera.intrinsic_matrix
        if K is None:
            raise RuntimeError("未找到相机内参，请先完成标定！")

        fx, fy = K[0, 0], K[1, 1]
        cx_k, cy_k = K[0, 2], K[1, 2]

        Xc = (cx - cx_k) * depth / fx
        Yc = (cy - cy_k) * depth / fy
        Zc = depth
        P_cam = np.array([Xc, Yc, Zc, 1]).reshape(4, 1)

        T_cam2end = self.depth_camera.T_cam2end
        if T_cam2end is None or len(T_cam2end) == 0:
            raise RuntimeError("未找到手眼标定矩阵，请先进行手眼标定！")

        P_end = T_cam2end @ P_cam
        Xe, Ye, Ze = P_end[:3, 0]

        R_end2base, t_end2base = self.arm_driver.get_end2base_matrix()
        T_end2base = np.eye(4)
        T_end2base[:3, :3] = R_end2base
        T_end2base[:3, 3] = t_end2base

        P_base = T_end2base @ np.array([Xe, Ye, Ze, 1]).reshape(4, 1)
        Xb, Yb, Zb = P_base[:3, 0]

        Rx, Ry, Rz = 0, 180, 90
        coords = [float(Xb), float(Yb), float(Zb), Rx, Ry, Rz]

        target_coords = self.clamp_coords(coords)

        print(f"【像素→机械臂坐标系】目标点: {target_coords}")
        return target_coords

    def clamp_coords(self, coords):
        x, y, z, rx, ry, rz = coords
        x = float(np.clip(x, -350.0, 350.0))
        y = float(np.clip(y, -350.0, 350.0))
        z = float(np.clip(z, -41.0, 523.9))
        rx = float(np.clip(rx, -180.0, 180.0))
        ry = float(np.clip(ry, -180.0, 180.0))
        rz = float(np.clip(rz, -180.0, 180.0))
        return [x, y, z, rx, ry, rz]

    def to_arm_action(self, decision: MllmDecision, handover_coords: List[float]) -> ArmAction:
        if decision.image_position is not None:
            cx, cy = decision.image_position
            pick_pose = tuple(self.pixel_to_robot_coords(cx, cy))  # type: ignore[assignment]
        else:
            pick_pose = (120.0, -120.0, 100.0, 0.0, 180.0, 90.0)

        place_pose = tuple(handover_coords)  # type: ignore[assignment]
        return ArmAction(
            target_name=decision.target_name,
            pick_pose=pick_pose,
            place_pose=place_pose,
            reply=decision.reply,
        )