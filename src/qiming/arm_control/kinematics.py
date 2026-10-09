"""机械臂运动学：正运动学 (FK) 与逆运动学 (IK)。

使用改进 DH (Craig) 参数描述 myCobot 320 各连杆的几何关系，
提供「基坐标系位姿 <-> 关节角度」的双向转换。

坐标与姿态约定（与 pymycobot 一致）:
    位姿 pose = [x, y, z, rx, ry, rz]
        x, y, z    : 末端在基坐标系下的位置，单位 mm
        rx, ry, rz : 末端姿态欧拉角 (Z-Y-X，即 R = Rz @ Ry @ Rx)，单位 度
    关节 joints = [j1, j2, j3, j4, j5, j6]，单位 度

逆运动学采用阻尼最小二乘 (Damped Least Squares) 数值迭代求解，
对 320 这种非球腕结构同样适用，通过多次随机重启避免局部极小。

DH 参数取自 Elephant Robotics myCobot 320 官方手册（连杆参数规格），
若与实机存在偏差，可通过 config/robot.yaml 的 kinematics.dh_params 覆盖。
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np


# myCobot 320 改进 DH 参数: [alpha_{i-1}(deg), a_{i-1}(mm), d_i(mm), theta_offset(deg)]
DEFAULT_DH_PARAMS: List[List[float]] = [
    [0.0, 0.0, 173.9, 0.0],
    [90.0, 0.0, 0.0, -90.0],
    [0.0, -135.0, 0.0, 0.0],
    [0.0, -120.0, 95.0, -90.0],
    [90.0, 0.0, 87.78, 0.0],
    [-90.0, 0.0, 65.5, 0.0],
]

# 关节运动范围（单位: 度），来源: myCobot 320 产品手册
DEFAULT_JOINT_LIMITS: List[tuple] = [
    (-168.0, 168.0),
    (-135.0, 135.0),
    (-145.0, 145.0),
    (-148.0, 148.0),
    (-168.0, 168.0),
    (-180.0, 180.0),
]


def _rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, -s, 0.0],
        [0.0, s, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def _rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ])


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [c, -s, 0.0, 0.0],
        [s, c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def _trans_x(distance: float) -> np.ndarray:
    T = np.eye(4)
    T[0, 3] = distance
    return T


def _trans_z(distance: float) -> np.ndarray:
    T = np.eye(4)
    T[2, 3] = distance
    return T


def rotation_vector_to_matrix(rvec: np.ndarray) -> np.ndarray:
    """罗德里格斯公式：旋转向量 -> 3x3 旋转矩阵。"""
    rvec = np.asarray(rvec, dtype=float).reshape(3)
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ])
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def matrix_to_rotation_vector(R: np.ndarray) -> np.ndarray:
    """3x3 旋转矩阵 -> 旋转向量（对数映射）。"""
    R = np.asarray(R, dtype=float)
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-9:
        return np.zeros(3)
    if abs(math.pi - theta) < 1e-6:
        # 接近 180 度，直接由对角元恢复轴向
        diag = np.clip((np.diag(R) + 1.0) / 2.0, 0.0, 1.0)
        axis = np.sqrt(diag)
        if R[2, 1] - R[1, 2] < 0:
            axis[0] = -axis[0]
        if R[0, 2] - R[2, 0] < 0:
            axis[1] = -axis[1]
        if R[1, 0] - R[0, 1] < 0:
            axis[2] = -axis[2]
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            return np.zeros(3)
        return axis / norm * theta
    factor = theta / (2.0 * math.sin(theta))
    return factor * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])


class ArmKinematics:
    """myCobot 320 正/逆运动学求解器。"""

    def __init__(
        self,
        dh_params: Optional[Sequence[Sequence[float]]] = None,
        joint_limits: Optional[Sequence[Sequence[float]]] = None,
    ) -> None:
        dh = [list(row) for row in (dh_params or DEFAULT_DH_PARAMS)]
        if len(dh) != 6 or any(len(row) != 4 for row in dh):
            raise ValueError("DH 参数需要 6 行，每行 4 个值")

        self.dh_params = dh
        self.joint_limits = [tuple(lim) for lim in (joint_limits or DEFAULT_JOINT_LIMITS)]

        self._alpha = np.radians([row[0] for row in dh])
        self._a = np.array([row[1] for row in dh], dtype=float)
        self._d = np.array([row[2] for row in dh], dtype=float)
        self._theta_offset = np.radians([row[3] for row in dh])

    # ------------------------------------------------------------------
    # 位姿 <-> 矩阵
    # ------------------------------------------------------------------
    @staticmethod
    def pose_to_matrix(pose: Sequence[float]) -> np.ndarray:
        """位姿 [x, y, z, rx, ry, rz] -> 4x4 齐次变换矩阵。"""
        if len(pose) != 6:
            raise ValueError("位姿需要 6 个分量 [x, y, z, rx, ry, rz]")
        x, y, z, rx, ry, rz = (float(v) for v in pose)
        R = (_rot_z(math.radians(rz))[:3, :3]
             @ _rot_y(math.radians(ry))
             @ _rot_x(math.radians(rx))[:3, :3])
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        return T

    @staticmethod
    def matrix_to_pose(T: np.ndarray) -> List[float]:
        """4x4 齐次变换矩阵 -> 位姿 [x, y, z, rx, ry, rz]（欧拉角 Z-Y-X，度）。"""
        T = np.asarray(T, dtype=float)
        x, y, z = T[:3, 3]
        R = T[:3, :3]

        # R = Rz(rz) @ Ry(ry) @ Rx(rx)
        sy = -R[2, 0]
        sy = float(np.clip(sy, -1.0, 1.0))
        ry = math.asin(sy)
        if abs(abs(sy) - 1.0) < 1e-6:
            # 万向锁：rx 与 rz 耦合，约定 rz = 0
            rz = 0.0
            rx = math.atan2(-R[1, 2], R[1, 1]) if sy > 0 else math.atan2(R[1, 2], R[1, 1])
        else:
            rx = math.atan2(R[2, 1], R[2, 2])
            rz = math.atan2(R[1, 0], R[0, 0])
        return [float(x), float(y), float(z),
                float(math.degrees(rx)), float(math.degrees(ry)), float(math.degrees(rz))]

    def _link_transform(self, index: int, theta: float) -> np.ndarray:
        """改进 DH 单连杆变换: Rx(alpha) * Tx(a) * Rz(theta) * Tz(d)。"""
        T = _rot_x(self._alpha[index]) @ _trans_x(self._a[index])
        T = T @ _rot_z(theta) @ _trans_z(self._d[index])
        return T

    # ------------------------------------------------------------------
    # 正运动学
    # ------------------------------------------------------------------
    def forward_kinematics(self, joints: Sequence[float]) -> np.ndarray:
        """关节角度 -> 末端到基坐标系的 4x4 变换矩阵。"""
        if len(joints) != 6:
            raise ValueError("关节角度需要 6 个分量")
        T = np.eye(4)
        for i, angle in enumerate(joints):
            theta = math.radians(float(angle)) + self._theta_offset[i]
            T = T @ self._link_transform(i, theta)
        return T

    def joints_to_pose(self, joints: Sequence[float]) -> List[float]:
        """关节角度 -> 基坐标位姿 [x, y, z, rx, ry, rz]。"""
        return self.matrix_to_pose(self.forward_kinematics(joints))

    # ------------------------------------------------------------------
    # 逆运动学
    # ------------------------------------------------------------------
    def _clamp(self, joints: np.ndarray) -> np.ndarray:
        limits = np.array(self.joint_limits, dtype=float)
        return np.clip(joints, limits[:, 0], limits[:, 1])

    @staticmethod
    def _cost(error: np.ndarray) -> float:
        """把位置(mm)与姿态(rad)误差合成单一代价，姿态权重抬高以兼顾两者。"""
        return float(np.linalg.norm(error[:3]) + 100.0 * np.linalg.norm(error[3:]))

    @staticmethod
    def _converged(error: np.ndarray, pos_tol: float, rot_tol: float) -> bool:
        return bool(np.linalg.norm(error[:3]) < pos_tol
                    and np.linalg.norm(error[3:]) < rot_tol)

    @staticmethod
    def _pose_error(T_current: np.ndarray, T_target: np.ndarray) -> np.ndarray:
        """当前位置与目标位姿的误差 [位置误差(mm); 姿态误差(rad)]。"""
        pos_error = T_target[:3, 3] - T_current[:3, 3]
        rot_error = matrix_to_rotation_vector(T_target[:3, :3] @ T_current[:3, :3].T)
        return np.concatenate([pos_error, rot_error])

    def _link_frames(self, joints: Sequence[float]) -> List[np.ndarray]:
        """返回基坐标系到各连杆坐标系的变换 T_0..T_6（T_0 为单位阵）。"""
        frames = [np.eye(4)]
        T = np.eye(4)
        for i, angle in enumerate(joints):
            theta = math.radians(float(angle)) + self._theta_offset[i]
            T = T @ self._link_transform(i, theta)
            frames.append(T)
        return frames

    def _analytic_jacobian(self, joints: Sequence[float],
                           frames: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """末端位姿误差的线性化雅可比 (6x6)，对「关节角度(度)」求导。

        以 e = [p_target - p_current; log(R_target · R_current^T)] 为误差，
        在当前位置附近有 e ≈ J · dq，J 的各列即几何雅可比：
            J_v_i = z_i × (p_end - p_i)，J_w_i = z_i
        其中 z_i 为关节 i 的轴线方向（改进 DH 下为 z_i 轴），p_i 取轴线上一点，
        即 M_i = T_0..i-1 · Rx(alpha_{i-1}) · Tx(a_{i-1}) 的原点。
        几何雅可比以弧度为定义、而关节角以度为单位，故最后乘 pi/180。
        """
        if frames is None:
            frames = self._link_frames(joints)
        p_end = frames[6][:3, 3]

        J = np.zeros((6, 6))
        for i in range(6):
            M = frames[i] @ _rot_x(self._alpha[i]) @ _trans_x(self._a[i])
            axis = M[:3, 2]
            point = M[:3, 3]
            J[:3, i] = np.cross(axis, p_end - point)
            J[3:, i] = axis
        return J * (math.pi / 180.0)

    def _numeric_jacobian(self, joints: np.ndarray, T_current: np.ndarray, delta: float = 1e-5) -> np.ndarray:
        """数值雅可比 (6x6)，用于校验解析雅可比。"""
        J = np.zeros((6, 6))
        for col in range(6):
            perturbed = joints.copy()
            perturbed[col] += delta
            T_next = self.forward_kinematics(perturbed)
            J[:3, col] = (T_next[:3, 3] - T_current[:3, 3]) / delta
            J[3:, col] = matrix_to_rotation_vector(T_next[:3, :3] @ T_current[:3, :3].T) / delta
        return J

    def _solve_once(
        self,
        T_target: np.ndarray,
        seed: np.ndarray,
        max_iter: int,
        pos_tol: float,
        rot_tol: float,
    ) -> tuple:
        """从一个初值出发做 Levenberg-Marquardt 迭代，返回 (关节角, 代价)。"""
        joints = self._clamp(seed.copy())
        frames = self._link_frames(joints)
        error = self._pose_error(frames[6], T_target)
        cost = self._cost(error)

        best_joints = joints.copy()
        best_cost = cost
        damping = 1e-2

        for _ in range(max_iter):
            if self._converged(error, pos_tol, rot_tol):
                break

            J = self._analytic_jacobian(joints, frames)
            JJt = J @ J.T + (damping ** 2) * np.eye(6)
            try:
                dq = J.T @ np.linalg.solve(JJt, error)
            except np.linalg.LinAlgError:
                break

            step_norm = float(np.linalg.norm(dq))
            if step_norm > 30.0:  # 限制单步幅度，避免震荡
                dq *= 30.0 / step_norm

            candidate = self._clamp(joints + dq)
            candidate_frames = self._link_frames(candidate)
            candidate_error = self._pose_error(candidate_frames[6], T_target)
            candidate_cost = self._cost(candidate_error)

            if candidate_cost < cost:
                joints, frames, error, cost = candidate, candidate_frames, candidate_error, candidate_cost
                damping = max(damping * 0.7, 1e-6)
                if cost < best_cost:
                    best_cost = cost
                    best_joints = joints.copy()
            else:
                # 步长使代价增大，加大阻尼重新求解
                damping = min(damping * 2.5, 1.0)

        if cost < best_cost:
            best_cost, best_joints = cost, joints.copy()
        return best_joints, best_cost

    def inverse_kinematics(
        self,
        target: Sequence[float] | np.ndarray,
        seed: Optional[Sequence[float]] = None,
        max_iter: int = 300,
        pos_tol: float = 0.1,
        rot_tol: float = math.radians(0.1),
        num_restarts: int = 32,
    ) -> Optional[List[float]]:
        """求解逆运动学：基坐标位姿 -> 关节角度。

        参数:
            target: 目标位姿 [x, y, z, rx, ry, rz] 或 4x4 齐次矩阵
            seed: 起始关节角度，用于保证解的连续性；为空时使用零位
            pos_tol: 位置收敛阈值 (mm)
            rot_tol: 姿态收敛阈值 (rad)
            num_restarts: 随机重启次数，用于跳出局部极小

        返回:
            关节角度列表，若在关节范围内无解则返回 None
        """
        if isinstance(target, np.ndarray) and target.shape == (4, 4):
            T_target = target.astype(float)
        else:
            T_target = self.pose_to_matrix(target)

        rng = np.random.default_rng(2026)
        seeds: List[np.ndarray] = []
        if seed is not None:
            seeds.append(np.array(seed, dtype=float))
        seeds.append(np.zeros(6))
        for _ in range(num_restarts):
            limits = np.array(self.joint_limits, dtype=float)
            seeds.append(rng.uniform(limits[:, 0], limits[:, 1]))

        best_joints: Optional[np.ndarray] = None
        best_cost = float("inf")
        for start in seeds:
            joints, cost = self._solve_once(T_target, start, max_iter, pos_tol, rot_tol)
            if self._converged(self._pose_error(self.forward_kinematics(joints), T_target),
                               pos_tol, rot_tol):
                # 找到可行解即返回；seeds 首个元素即用户给定的 seed，
                # 因此结果与当前位形连续
                return [float(v) for v in joints]
            if best_joints is None or cost < best_cost:
                best_cost = cost
                best_joints = joints

        if best_joints is None:
            return None

        # 末段精修：近奇异位形下 DLS 收敛较慢，从当前最优解再迭代一轮
        refined, refined_cost = self._solve_once(
            T_target, best_joints, max_iter * 3, pos_tol, rot_tol)
        if refined_cost < best_cost:
            best_joints, best_cost = refined, refined_cost

        joints = self._clamp(best_joints)
        final_error = self._pose_error(self.forward_kinematics(joints), T_target)
        if (np.linalg.norm(final_error[:3]) > pos_tol * 10
                or np.linalg.norm(final_error[3:]) > rot_tol * 10):
            return None
        return [float(v) for v in joints]

    def pose_to_joints(self, pose: Sequence[float], seed: Optional[Sequence[float]] = None) -> Optional[List[float]]:
        """便捷入口：基坐标位姿 -> 关节角度。"""
        return self.inverse_kinematics(pose, seed=seed)

    def is_reachable(self, pose: Sequence[float], tolerance: float = 0.5) -> bool:
        """判断给定位姿是否在关节范围内可达。"""
        joints = self.inverse_kinematics(pose, pos_tol=tolerance, rot_tol=math.radians(tolerance))
        return joints is not None
