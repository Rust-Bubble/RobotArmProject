from __future__ import annotations

import numpy as np
from typing import List, Optional

from src.qiming.models import ArmAction, MllmDecision, Pose
from src.qiming.vision.depth_camera import DepthCamera
from src.qiming.arm_control.mycobot_driver import MyCobotDriver


class CoordinateConverter:
    """
    坐标转换器
    
    支持两种手眼标定模式:
    - Eye-to-Hand (眼在手外): T_BC 相机在基坐标系下的外参
    - Eye-in-Hand (眼在手上): T_CE 相机在末端坐标系下的外参
    
    默认使用 Eye-to-Hand 模式
    """

    def __init__(self, depth_camera: DepthCamera, arm_driver: MyCobotDriver):
        self.depth_camera = depth_camera
        self.arm_driver = arm_driver
        self.calibration_mode = "eye_to_hand"  # "eye_to_hand" 或 "eye_in_hand"
        
        # Eye-to-Hand 模式: 相机在基坐标系下的外参
        self.T_BC: Optional[np.ndarray] = None
        
        # Eye-in-Hand 模式: 相机在末端坐标系下的外参 (保留兼容)
        self.T_CE: Optional[np.ndarray] = None

    def set_calibration_mode(self, mode: str):
        """
        设置手眼标定模式
        
        参数:
            mode: "eye_to_hand" 或 "eye_in_hand"
        """
        if mode in ["eye_to_hand", "eye_in_hand"]:
            self.calibration_mode = mode
            print(f"[配置] 手眼标定模式设置为: {mode}")
        else:
            print(f"[错误] 未知模式: {mode}")

    def set_T_BC(self, T_BC: np.ndarray):
        """
        设置 Eye-to-Hand 标定矩阵 (相机在基坐标系下的外参)
        
        参数:
            T_BC: 4x4 齐次变换矩阵
        """
        self.T_BC = T_BC
        print("[配置] T_BC 矩阵已设置")

    def set_T_CE(self, T_CE: np.ndarray):
        """
        设置 Eye-in-Hand 标定矩阵 (相机在末端坐标系下的外参)
        
        参数:
            T_CE: 4x4 齐次变换矩阵
        """
        self.T_CE = T_CE
        print("[配置] T_CE 矩阵已设置")

    def pixel_to_camera_coords(self, cx: float, cy: float, depth_img: np.ndarray, K: np.ndarray) -> np.ndarray:
        """
        像素坐标转换为相机坐标系坐标
        
        参数:
            cx: 像素x坐标
            cy: 像素y坐标
            depth_img: 深度图像
            K: 相机内参矩阵
            
        返回:
            4x1 齐次坐标 [X, Y, Z, 1]
        """
        if depth_img is None:
            raise RuntimeError("深度图未获取到")

        height, width = depth_img.shape
        cx_int, cy_int = int(round(cx)), int(round(cy))
        
        if cx_int < 0 or cx_int >= width or cy_int < 0 or cy_int >= height:
            raise RuntimeError(f"像素坐标 ({cx}, {cy}) 超出图像范围 ({width}x{height})")

        depth = float(depth_img[cy_int, cx_int])
        
        if depth == 0:
            raise RuntimeError(f"像素 ({cx}, {cy}) 处的深度值为0，无法转换")

        fx, fy = K[0, 0], K[1, 1]
        cx_k, cy_k = K[0, 2], K[1, 2]

        Xc = (cx - cx_k) * depth / fx
        Yc = (cy - cy_k) * depth / fy
        Zc = depth

        P_cam = np.array([Xc, Yc, Zc, 1.0])
        return P_cam

    def camera_to_base_eye_to_hand(self, P_cam: np.ndarray) -> np.ndarray:
        """
        相机坐标 -> 基坐标 (Eye-to-Hand模式)
        
        P_base = T_BC @ P_cam
        
        参数:
            P_cam: 4x1 相机坐标系齐次坐标
        返回:
            4x1 基坐标系齐次坐标
        """
        if self.T_BC is None:
            raise RuntimeError("T_BC矩阵未设置，请先完成Eye-to-Hand手眼标定")
        
        P_base = self.T_BC @ P_cam
        return P_base

    def camera_to_base_eye_in_hand(self, P_cam: np.ndarray) -> np.ndarray:
        """
        相机坐标 -> 基坐标 (Eye-in-Hand模式)
        
        P_base = T_EB @ T_CE @ P_cam
        
        参数:
            P_cam: 4x1 相机坐标系齐次坐标
        返回:
            4x1 基坐标系齐次坐标
        """
        if self.T_CE is None:
            raise RuntimeError("T_CE矩阵未设置，请先完成Eye-in-Hand手眼标定")
        
        R_end2base, t_end2base = self.arm_driver.get_end2base_matrix()
        T_EB = np.eye(4)
        T_EB[:3, :3] = R_end2base
        T_EB[:3, 3] = t_end2base.flatten()
        
        P_base = T_EB @ self.T_CE @ P_cam
        return P_base

    def pixel_to_robot_coords(self, cx: float, cy: float) -> List[float]:
        """
        像素坐标 -> 机械臂基坐标 (根据当前标定模式)
        
        参数:
            cx: 像素x坐标
            cy: 像素y坐标
        返回:
            [X, Y, Z, Rx, Ry, Rz] 机械臂位姿
        """
        color_img, depth_img = self.depth_camera.get_frames()
        if depth_img is None:
            raise RuntimeError("深度图未获取到，请先打开相机")

        K = self.depth_camera.intrinsic_matrix
        if K is None:
            raise RuntimeError("未找到相机内参，请先完成标定！")

        # Step 1: 像素坐标 -> 相机坐标
        P_cam = self.pixel_to_camera_coords(cx, cy, depth_img, K)
        
        print(f"[坐标转换] 相机坐标系: X={P_cam[0]:.2f}, Y={P_cam[1]:.2f}, Z={P_cam[2]:.2f}")

        # Step 2: 相机坐标 -> 基坐标
        if self.calibration_mode == "eye_to_hand":
            P_base = self.camera_to_base_eye_to_hand(P_cam)
        else:
            P_base = self.camera_to_base_eye_in_hand(P_cam)
        
        Xb, Yb, Zb = P_base[:3]
        print(f"[坐标转换] 基坐标系: X={Xb:.2f}, Y={Yb:.2f}, Z={Zb:.2f}")

        # Step 3: 附加固定俯视姿态
        Rx, Ry, Rz = 0.0, 180.0, 90.0
        coords = [float(Xb), float(Yb), float(Zb), Rx, Ry, Rz]

        target_coords = self.clamp_coords(coords)
        print(f"[坐标转换] 机械臂目标位姿: {target_coords}")
        return target_coords

    def clamp_coords(self, coords: List[float]) -> List[float]:
        """
        限制机械臂坐标在安全工作空间内
        
        参数:
            coords: [X, Y, Z, Rx, Ry, Rz]
        返回:
            限制后的坐标
        """
        x, y, z, rx, ry, rz = coords
        x = float(np.clip(x, -350.0, 350.0))
        y = float(np.clip(y, -350.0, 350.0))
        z = float(np.clip(z, -41.0, 523.9))
        rx = float(np.clip(rx, -180.0, 180.0))
        ry = float(np.clip(ry, -180.0, 180.0))
        rz = float(np.clip(rz, -180.0, 180.0))
        return [x, y, z, rx, ry, rz]

    def to_arm_action(self, decision: MllmDecision, handover_coords: List[float]) -> ArmAction:
        """
        将MLLM决策转换为机械臂动作
        
        参数:
            decision: MLLM决策结果
            handover_coords: 递送点位姿
        返回:
            ArmAction 对象
        """
        if decision.image_position is not None:
            cx, cy = decision.image_position
            pick_pose = tuple(self.pixel_to_robot_coords(cx, cy))
        else:
            pick_pose = (120.0, -120.0, 100.0, 0.0, 180.0, 90.0)

        place_pose = tuple(handover_coords)
        return ArmAction(
            target_name=decision.target_name,
            pick_pose=pick_pose,
            place_pose=place_pose,
            reply=decision.reply,
        )