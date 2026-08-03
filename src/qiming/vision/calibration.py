from __future__ import annotations

import cv2
import numpy as np
from typing import List, Tuple, Optional

from src.qiming.vision.depth_camera import DepthCamera
from src.qiming.arm_control.mycobot_driver import MyCobotDriver


class EyeToHandCalibrator:
    """
    Eye-to-Hand (眼在手外) 手眼标定框架
    
    相机固定在外部，标定板固定在机械臂末端。
    核心方程: AX = XB
    
    A = T_E2B * inv(T_E1B)  -- 两次机械臂位姿的相对变换
    B = T_K2C * inv(T_K1C)  -- 两次视觉观测的相对变换
    X = T_BC                -- 相机在基坐标系下的外参（待求解）
    """

    def __init__(self, depth_camera: DepthCamera, arm_driver: MyCobotDriver):
        self.depth_camera = depth_camera
        self.arm_driver = arm_driver

        # ============ 标定参数 ============
        self.calibration_times = 20  # 标定点数量
        self.CHESSBOARD_SIZE = (9, 6)  # 棋盘格内角点
        self.SQUARE_SIZE = 0.025  # 方格尺寸(米)

        # ============ 存储数据 ============
        # 机械臂位姿: 末端到基坐标系的变换矩阵 T_EB
        # 每个元素: (rotation_matrix, translation_vector)
        self.robot_poses: List[Tuple[np.ndarray, np.ndarray]] = []

        # 视觉观测: 标定板到相机坐标系的变换矩阵 T_KC
        # 每个元素: (rotation_matrix, translation_vector)
        self.camera_observations: List[Tuple[np.ndarray, np.ndarray]] = []

        # ============ 标定结果 ============
        self.T_BC: Optional[np.ndarray] = None  # 相机在基坐标系下的外参 (4x4)
        
        # ============ 内参矩阵占位 ============
        # TODO: 接入真实相机后通过标定获取
        # 目前置空，需要手动填入
        self.camera_intrinsic_matrix: Optional[np.ndarray] = None  # 相机内参矩阵 K
        self.camera_dist_coeffs: Optional[np.ndarray] = None  # 畸变系数
        
        # ============ 配置标记 ============
        self._intrinsic_calibrated = False  # 内参是否已标定
        self._data_collected = False  # 数据是否已采集
        self._calibrated = False  # 手眼标定是否完成

    def set_intrinsic_matrix(self, intrinsic_matrix: np.ndarray, dist_coeffs: np.ndarray):
        """
        设置相机内参矩阵（手动填入或从文件加载）
        
        参数:
            intrinsic_matrix: 3x3 相机内参矩阵 K
            dist_coeffs: 畸变系数向量
        """
        self.camera_intrinsic_matrix = intrinsic_matrix
        self.camera_dist_coeffs = dist_coeffs
        self._intrinsic_calibrated = True
        print("[配置] 相机内参矩阵已设置")
        print(f"  K = \n{intrinsic_matrix}")

    def collect_robot_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        采集当前机械臂位姿 (T_EB: 末端到基坐标系)
        
        返回:
            (rotation_matrix: 3x3, translation_vector: 3x1)
        """
        rotation_matrix, translation_vector = self.arm_driver.get_end2base_matrix()
        self.robot_poses.append((rotation_matrix, translation_vector))
        print(f"[采集] 机械臂位姿 #{len(self.robot_poses)}:")
        print(f"  R = \n{rotation_matrix}")
        print(f"  t = {translation_vector}")
        return rotation_matrix, translation_vector

    def collect_camera_observation(self, rvec: np.ndarray, tvec: np.ndarray):
        """
        采集当前视觉观测 (T_KC: 标定板到相机坐标系)
        
        参数:
            rvec: Rodrigues旋转向量 (3x1)
            tvec: 平移向量 (3x1)
        """
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation_vector = tvec.flatten()
        self.camera_observations.append((rotation_matrix, translation_vector))
        print(f"[采集] 视觉观测 #{len(self.camera_observations)}:")
        print(f"  R = \n{rotation_matrix}")
        print(f"  t = {translation_vector}")

    def collect_camera_observation_from_frame(self, frame) -> bool:
        """
        从图像帧自动检测标定板并采集视觉观测
        
        参数:
            frame: BGR图像帧
            
        返回:
            是否成功采集
        """
        if self.camera_intrinsic_matrix is None:
            print("[错误] 内参矩阵未设置，无法进行solvePnP")
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 准备棋盘格3D点
        objp = np.zeros((self.CHESSBOARD_SIZE[0] * self.CHESSBOARD_SIZE[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.CHESSBOARD_SIZE[0], 0:self.CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
        objp *= self.SQUARE_SIZE

        # 检测棋盘格角点
        ret, corners = cv2.findChessboardCorners(gray, self.CHESSBOARD_SIZE, None)
        
        if not ret:
            print("[警告] 未检测到棋盘格")
            return False

        # 亚像素优化
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        # solvePnP求解标定板位姿
        retval, rvec, tvec = cv2.solvePnP(
            objp, corners2,
            self.camera_intrinsic_matrix,
            self.camera_dist_coeffs
        )

        if retval:
            self.collect_camera_observation(rvec, tvec)
            return True
        else:
            print("[错误] solvePnP求解失败")
            return False

    def get_transform_matrix(self, rotation_matrix: np.ndarray, translation_vector: np.ndarray) -> np.ndarray:
        """
        构造4x4齐次变换矩阵
        
        参数:
            rotation_matrix: 3x3 旋转矩阵
            translation_vector: 3x1 平移向量
            
        返回:
            4x4 齐次变换矩阵
        """
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = translation_vector.flatten()
        return T

    def compute_A_matrix(self, idx_1: int, idx_2: int) -> np.ndarray:
        """
        计算A矩阵: A = T_E2B * inv(T_E1B)
        
        参数:
            idx_1: 第一个位姿的索引
            idx_2: 第二个位姿的索引
            
        返回:
            4x4 A矩阵
        """
        R1, t1 = self.robot_poses[idx_1]
        R2, t2 = self.robot_poses[idx_2]

        T1 = self.get_transform_matrix(R1, t1)
        T2 = self.get_transform_matrix(R2, t2)

        # A = T2 * inv(T1)
        T1_inv = np.linalg.inv(T1)
        A = T2 @ T1_inv

        print(f"[计算] A矩阵 (位姿#{idx_2} * 位姿#{idx_1}^-1):")
        print(f"  A = \n{A}")
        return A

    def compute_B_matrix(self, idx_1: int, idx_2: int) -> np.ndarray:
        """
        计算B矩阵: B = T_K2C * inv(T_K1C)
        
        参数:
            idx_1: 第一个观测的索引
            idx_2: 第二个观测的索引
            
        返回:
            4x4 B矩阵
        """
        R1, t1 = self.camera_observations[idx_1]
        R2, t2 = self.camera_observations[idx_2]

        T1 = self.get_transform_matrix(R1, t1)
        T2 = self.get_transform_matrix(R2, t2)

        # B = T2 * inv(T1)
        T1_inv = np.linalg.inv(T1)
        B = T2 * T1_inv

        print(f"[计算] B矩阵 (观测#{idx_2} * 观测#{idx_1}^-1):")
        print(f"  B = \n{B}")
        return B

    def calibrate_hand_eye(self) -> Optional[np.ndarray]:
        """
        求解手眼标定: AX = XB
        
        使用OpenCV的calibrateHandEye函数求解X
        
        返回:
            T_BC: 相机在基坐标系下的外参 (4x4)
        """
        if len(self.robot_poses) < 2:
            print("[错误] 需要至少2组机械臂位姿数据")
            return None
        if len(self.camera_observations) < 2:
            print("[错误] 需要至少2组视觉观测数据")
            return None

        if len(self.robot_poses) != len(self.camera_observations):
            print("[错误] 机械臂位姿和视觉观测数据数量不匹配")
            return None

        # 准备输入数据
        r_end2base = [pose[0] for pose in self.robot_poses]
        t_end2base = [pose[1].reshape(3, 1) for pose in self.robot_poses]
        
        r_cam2target = [obs[0] for obs in self.camera_observations]
        t_cam2target = [obs[1].reshape(3, 1) for obs in self.camera_observations]

        # 调用OpenCV标定函数
        # method=0: CALIB_HAND_EYE_TSAI
        # method=1: CALIB_HAND_EYE_PARK
        # method=2: CALIB_HAND_EYE_HORAUD
        # method=3: CALIB_HAND_EYE_ANDREFF
        # method=4: CALIB_HAND_EYE_DANIILIDIS
        r_cam2base, t_cam2base = cv2.calibrateHandEye(
            r_end2base,
            t_end2base,
            r_cam2target,
            t_cam2target,
            self.camera_intrinsic_matrix,
            self.camera_dist_coeffs,
            method=cv2.CALIB_HAND_EYE_TSAI
        )

        # 构造4x4变换矩阵
        self.T_BC = self.get_transform_matrix(r_cam2base, t_cam2base)
        
        self._calibrated = True
        self.depth_camera.T_cam2end = self.T_BC  # 复用现有属性名
        
        print("\n" + "="*50)
        print("[结果] 手眼标定完成!")
        print("="*50)
        print(f"T_BC (相机在基坐标系下的外参):")
        print(self.T_BC)
        print("="*50)
        
        return self.T_BC

    def run_calibration(self):
        """
        执行完整的手眼标定流程
        
        注意: 需要先设置内参矩阵，且机械臂和相机已连接
        """
        if not self._intrinsic_calibrated:
            print("[警告] 内参矩阵未设置，请先调用 set_intrinsic_matrix()")
            print("  示例:")
            print("    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])")
            print("    dist = np.zeros((5, 1))")
            print("    calibrator.set_intrinsic_matrix(K, dist)")
            return None

        print(f"\n开始Eye-to-Hand手眼标定流程")
        print(f"计划采集 {self.calibration_times} 组数据\n")

        for i in range(self.calibration_times):
            print(f"\n--- 第 {i+1}/{self.calibration_times} 轮 ---")
            
            # 1. 采集机械臂位姿
            input("移动机械臂到标定位置后按 Enter 采集位姿...")
            self.collect_robot_pose()
            
            # 2. 采集视觉观测
            input("确保标定板在视野中后按 Enter 采集图像...")
            
            # 这里需要相机实际连接后获取帧
            # frame = self.depth_camera.get_frames()[0]
            # self.collect_camera_observation_from_frame(frame)
            
            # 临时: 使用占位数据 (需要手动提供rvec, tvec)
            print("  [占位] 请提供solvePnP结果 (rvec, tvec)")
            print("  [占位] 或等待相机接入后自动采集")

        self._data_collected = True
        
        # 3. 计算手眼标定
        return self.calibrate_hand_eye()

    def get_camera_to_base_transform(self) -> Optional[np.ndarray]:
        """
        获取相机到基坐标系的变换矩阵
        
        返回:
            T_BC 矩阵，未标定时返回None
        """
        return self.T_BC

    def reset(self):
        """重置标定数据"""
        self.robot_poses = []
        self.camera_observations = []
        self.T_BC = None
        self._data_collected = False
        self._calibrated = False
        print("[重置] 标定数据已清空")

    def load_intrinsic_from_file(self, filepath: str):
        """
        从文件加载内参矩阵
        
        参数:
            filepath: .npz文件路径 (包含 'K' 和 'dist' 键)
        """
        try:
            data = np.load(filepath)
            K = data['K']
            dist = data['dist']
            self.set_intrinsic_matrix(K, dist)
            print(f"[加载] 内参矩阵已从 {filepath} 加载")
        except Exception as e:
            print(f"[错误] 加载内参矩阵失败: {e}")

    def save_calibration_result(self, filepath: str):
        """
        保存标定结果到文件
        
        参数:
            filepath: .npz文件路径
        """
        if self.T_BC is None:
            print("[错误] 标定结果为空，无法保存")
            return
        
        np.savez(filepath, T_BC=self.T_BC)
        print(f"[保存] 标定结果已保存到 {filepath}")

    def load_calibration_result(self, filepath: str):
        """
        从文件加载标定结果
        
        参数:
            filepath: .npz文件路径
        """
        try:
            data = np.load(filepath)
            self.T_BC = data['T_BC']
            self._calibrated = True
            self.depth_camera.T_cam2end = self.T_BC
            print(f"[加载] 标定结果已从 {filepath} 加载")
            print(f"  T_BC = \n{self.T_BC}")
        except Exception as e:
            print(f"[错误] 加载标定结果失败: {e}")