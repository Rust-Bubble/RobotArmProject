from __future__ import annotations

import math
import time
import numpy as np

from pymycobot import MyCobot320, PI_PORT

from src.qiming.arm_control.kinematics import ArmKinematics
from src.qiming.models import Pose


class MyCobotDriver:
    def __init__(self, robot_cfg: dict) -> None:
        self.robot_cfg = robot_cfg
        self.port = robot_cfg.get("port", PI_PORT)
        self.baudrate = robot_cfg.get("baudrate", 115200)
        self.default_speed = robot_cfg.get("default_speed", 35)
        self.gripper_config = robot_cfg.get("gripper", {})
        self.mc = None

        # 逆运动学求解器：优先使用固件自带解算，无硬件时回退到本地算法
        kinematics_cfg = robot_cfg.get("kinematics", {})
        self.kinematics = ArmKinematics(
            dh_params=kinematics_cfg.get("dh_params"),
            joint_limits=kinematics_cfg.get("joint_limits"),
        )

        self._connect()

    def _connect(self) -> None:
        try:
            self.mc = MyCobot320(self.port, self.baudrate)
            self.mc.set_fresh_mode(0)
            print(f"机械臂连接成功: {self.port} @ {self.baudrate}")
        except Exception as e:
            print(f"机械臂连接失败: {str(e)}")
            self.mc = None

    def move_to(self, pose: Pose) -> None:
        if self.mc is None:
            print("[机械臂] 未连接，跳过移动")
            return
        x, y, z, rx, ry, rz = pose
        self.mc.send_coords([x, y, z, rx, ry, rz], self.default_speed)
        time.sleep(2)

    # ==================== 运动学: 位姿 <-> 关节角度 ====================
    def get_angles(self) -> list:
        """读取当前关节角度，未连接时返回零位。"""
        if self.mc is not None:
            try:
                angles = self.mc.get_angles()
                if angles and len(angles) == 6:
                    return [float(v) for v in angles]
            except Exception as e:
                print(f"[机械臂] 读取关节角度失败: {e}")
        return [0.0] * 6

    def pose_to_joints(self, pose: Pose, current_angles: list = None) -> list:
        """基坐标位姿 -> 关节角度。

        已连接实机时优先使用固件自带逆解（与真实 DH 完全一致），
        否则使用本地 ArmKinematics 数值求解；无解时返回 None。
        """
        if self.mc is not None:
            try:
                seed = current_angles or self.get_angles()
                angles = self.mc.solve_inv_kinematics(list(pose), list(seed))
                if angles and len(angles) == 6:
                    return [float(v) for v in angles]
            except Exception as e:
                print(f"[机械臂] 固件逆解失败({e})，回退到本地算法")
        joints = self.kinematics.pose_to_joints(list(pose), seed=current_angles)
        if joints is None:
            print(f"[机械臂] 位姿不可达: {pose}")
        return joints

    def joints_to_pose(self, joints: list = None) -> list:
        """关节角度 -> 基坐标位姿；未指定时读取当前角度。"""
        joints = joints if joints is not None else self.get_angles()
        if self.mc is not None:
            try:
                coords = self.mc.angles_to_coords(list(joints))
                if coords and len(coords) == 6:
                    return [float(v) for v in coords]
            except Exception as e:
                print(f"[机械臂] 固件正解失败({e})，回退到本地算法")
        return self.kinematics.joints_to_pose(joints)

    def move_to_joints(self, joints: list, speed: int = None) -> None:
        """按关节角度移动机械臂。"""
        if self.mc is None:
            print(f"[机械臂] 未连接，跳过关节移动: {[round(v, 2) for v in joints]}")
            return
        self.mc.send_angles(list(joints), speed or self.default_speed)
        time.sleep(2)

    def move_to_by_ik(self, pose: Pose, speed: int = None) -> bool:
        """先做逆运动学转换，再以关节角度方式移动。"""
        joints = self.pose_to_joints(pose)
        if joints is None:
            return False
        print(f"[机械臂] IK 求解 {[round(v, 1) for v in pose]} -> "
              f"关节角 {[round(v, 2) for v in joints]}")
        self.move_to_joints(joints, speed)
        return True

    def open_gripper(self) -> None:
        if self.mc is None:
            print("[机械臂] 未连接，跳过打开夹爪")
            return
        try:
            self.mc.set_gripper_mode(0)
            self.mc.init_electric_gripper()
            self.mc.set_gripper_state(0, 50, 4)
            time.sleep(0.5)
            print("[机械臂] 打开夹爪")
        except Exception as e:
            print(f"打开夹爪失败: {str(e)}")

    def close_gripper(self) -> None:
        if self.mc is None:
            print("[机械臂] 未连接，跳过闭合夹爪")
            return
        try:
            self.mc.set_gripper_mode(0)
            self.mc.init_electric_gripper()
            self.mc.set_gripper_state(1, 30, 4)
            time.sleep(1.0)
            print("[机械臂] 闭合夹爪")
        except Exception as e:
            print(f"闭合夹爪失败: {str(e)}")

    def get_end2base_matrix(self, use_actual_robot=True):
        def euler_to_rotation_matrix(rx, ry, rz, degrees=True):
            if degrees:
                rx = math.radians(rx)
                ry = math.radians(ry)
                rz = math.radians(rz)

            Rz = np.array([
                [math.cos(rz), -math.sin(rz), 0],
                [math.sin(rz), math.cos(rz), 0],
                [0, 0, 1]
            ])

            Ry = np.array([
                [math.cos(ry), 0, math.sin(ry)],
                [0, 1, 0],
                [-math.sin(ry), 0, math.cos(ry)]
            ])

            Rx = np.array([
                [1, 0, 0],
                [0, math.cos(rx), -math.sin(rx)],
                [0, math.sin(rx), math.cos(rx)]
            ])

            return Rz @ Ry @ Rx

        def get_official_coords():
            if use_actual_robot and self.mc is not None:
                try:
                    self.mc.set_reference_frame(0)
                    self.mc.set_end_type(1)
                    arm_coords = self.mc.get_coords()
                    print("成功获取机械臂位姿数据")
                    return arm_coords
                except Exception as e:
                    print(f"获取实际位姿失败: {e}，将使用示例数据")

            return [100, 200, 300, 90, 0, 45]

        arm_coords = get_official_coords()
        print(f"获取的位姿数据: {arm_coords}")

        x, y, z, rx, ry, rz = arm_coords

        rotation_matrix = euler_to_rotation_matrix(rx, ry, rz)
        translation_vector = np.array([x, y, z])

        return rotation_matrix, translation_vector

    def check_gripper(self):
        if self.mc is None:
            print("[机械臂] 未连接，跳过夹爪测试")
            return False
        try:
            self.mc.set_gripper_mode(0)
            self.mc.init_electric_gripper()

            for i in range(2):
                self.mc.set_gripper_state(1, 100, 4)
                time.sleep(1)
                self.mc.set_gripper_state(0, 100, 4)
                time.sleep(1)

            return True
        except Exception as e:
            print(f"夹爪测试错误: {str(e)}")
            return False

    def back_zero(self):
        if self.mc is None:
            print("[机械臂] 未连接，跳过归零")
            return
        print('机械臂归零')
        self.mc.send_angles([0, 0, 0, 0, 0, 0], 40)
        time.sleep(3)

    def relax_arms(self):
        if self.mc is None:
            print("[机械臂] 未连接，跳过放松关节")
            return
        print('已放松机械臂关节')
        self.mc.release_all_servos()

    def move_to_coords(self, X=150, Y=-130, HEIGHT_SAFE=230):
        if self.mc is None:
            print("[机械臂] 未连接，跳过移动")
            return
        print('移动至指定坐标：X {} Y {}'.format(X, Y))
        self.mc.send_coords([X, Y, HEIGHT_SAFE, 0, 180, 90], 20)
        time.sleep(4)

    def single_joint_move(self, joint_index, angle):
        if self.mc is None:
            print("[机械臂] 未连接，跳过关节移动")
            return
        print('关节 {} 旋转至 {} 度'.format(joint_index, angle))
        self.mc.send_angle(joint_index, angle, 40)
        time.sleep(2)

    def move_to_top_view(self):
        if self.mc is None:
            print("[机械臂] 未连接，跳过俯视姿态")
            return
        print('移动至俯视姿态')
        self.mc.send_angles([-62.13, 8.96, -87.71, -14.41, 2.54, -16.34], 10)
        time.sleep(3)

    def grab_and_place(self, grab_point, place_point):
        if self.mc is None:
            print("[机械臂] 未连接，跳过抓取放置")
            return

        init_angles = [0, 0, 0, 0, 0, 0]

        self.mc.set_gripper_mode(0)
        self.mc.init_electric_gripper()
        self.mc.set_gripper_state(0, 50, 4)
        time.sleep(1)

        self.mc.send_angles(init_angles, 50)
        time.sleep(2)

        self.mc.send_coords([grab_point[0], grab_point[1], grab_point[2] + 70,
                             grab_point[3], grab_point[4], grab_point[5]], 50, 1)
        time.sleep(2)

        self.mc.send_coords(grab_point, 30, 1)
        time.sleep(2)

        self.mc.set_gripper_state(1, 30, 4)
        time.sleep(1.5)

        self.mc.send_coords([grab_point[0], grab_point[1], grab_point[2] + 70,
                             grab_point[3], grab_point[4], grab_point[5]], 40, 1)
        time.sleep(2)

        self.mc.send_coords([place_point[0], place_point[1], place_point[2] + 70,
                             place_point[3], place_point[4], place_point[5]], 50, 1)
        time.sleep(2)

        self.mc.send_coords(place_point, 30, 1)
        time.sleep(2)

        self.mc.set_gripper_state(0, 30, 4)
        time.sleep(1.5)

        self.mc.send_coords([place_point[0], place_point[1], place_point[2] + 70,
                             place_point[3], place_point[4], place_point[5]], 40, 1)
        time.sleep(2)

        self.mc.send_angles(init_angles, 50)
        time.sleep(2)

    def clamp_coords(self, coords):
        x, y, z, rx, ry, rz = coords
        x = float(np.clip(x, -350.0, 350.0))
        y = float(np.clip(y, -350.0, 350.0))
        z = float(np.clip(z, -41.0, 523.9))
        rx = float(np.clip(rx, -180.0, 180.0))
        ry = float(np.clip(ry, -180.0, 180.0))
        rz = float(np.clip(rz, -180.0, 180.0))
        return [x, y, z, rx, ry, rz]

    def get_end2base_transform_matrix(self, use_actual_robot=True) -> np.ndarray:
        """
        获取末端到基坐标系的4x4齐次变换矩阵 T_EB
        
        参数:
            use_actual_robot: 是否使用真实机械臂
            
        返回:
            4x4 齐次变换矩阵
        """
        rotation_matrix, translation_vector = self.get_end2base_matrix(use_actual_robot)
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = translation_vector.flatten()
        return T

    def compute_relative_transform(self, T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
        """
        计算两个位姿的相对变换: A = T2 * inv(T1)
        
        Eye-to-Hand标定中的A矩阵计算
        
        参数:
            T1: 第一个位姿的4x4变换矩阵
            T2: 第二个位姿的4x4变换矩阵
            
        返回:
            4x4 A矩阵
        """
        T1_inv = np.linalg.inv(T1)
        A = T2 @ T1_inv
        return A

    def transform_camera_to_base(self, point_in_camera: np.ndarray, T_BC: np.ndarray) -> np.ndarray:
        """
        相机坐标系坐标转换到基坐标系 (Eye-to-Hand模式)

        T_BC 为「相机在基坐标系下的外参」，即把相机坐标映射到基坐标的变换。

        参数:
            point_in_camera: 4x1 齐次坐标 [X, Y, Z, 1]
            T_BC: 相机在基坐标系下的外参 (4x4)

        返回:
            基坐标系下的4x1齐次坐标
        """
        return T_BC @ point_in_camera

    def transform_base_to_camera(self, point_in_base: np.ndarray, T_BC: np.ndarray) -> np.ndarray:
        """
        基坐标系坐标转换到相机坐标系 (Eye-to-Hand模式)

        参数:
            point_in_base: 4x1 齐次坐标 [X, Y, Z, 1]
            T_BC: 相机在基坐标系下的外参 (4x4)

        返回:
            相机坐标系下的4x1齐次坐标
        """
        T_BC_inv = np.linalg.inv(T_BC)
        return T_BC_inv @ point_in_base