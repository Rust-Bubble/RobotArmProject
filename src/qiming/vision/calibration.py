from __future__ import annotations

import cv2
import numpy as np

from src.qiming.vision.depth_camera import DepthCamera
from src.qiming.arm_control.mycobot_driver import MyCobotDriver


class HandEyeCalibrator:
    def __init__(self, depth_camera: DepthCamera, arm_driver: MyCobotDriver):
        self.depth_camera = depth_camera
        self.arm_driver = arm_driver
        self.camera_calibration_times = 20
        self.success_calibration_count = []
        self.T_cam2end = []
        self.CHESSBOARD_SIZE = (9, 6)
        self.SQUARE_SIZE = 0.025
        self.calibration_armgrasper_result = []
        self.r_end2base = []
        self.t_end2base = []

    def going_calibration_capture(self):
        print(f"手眼标定开始！一共要进行{self.camera_calibration_times}次手眼标定")
        for idx in range(self.camera_calibration_times):
            self.depth_camera.HandEye_calibration_capture(idx)
            rotation_matrix, translation_matrix = self.arm_driver.get_end2base_matrix()
            self.calibration_armgrasper_result.append(
                {
                    'idx': idx,
                    'rotation_matrix': rotation_matrix,
                    'translation_matrix': translation_matrix,
                }
            )
        print("手眼标定完成！")

    def going_calibration_calculation(self):
        (self.depth_camera.calibration_objpoints,
         self.depth_camera.calibration_imgpoints,
         self.success_calibration_count) = self.depth_camera.going_camera_calibration(
            self.CHESSBOARD_SIZE, self.SQUARE_SIZE
        )

        for i, (obj_pts, img_pts) in enumerate(zip(
                self.depth_camera.calibration_objpoints,
                self.depth_camera.calibration_imgpoints
        )):
            retval, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts,
                self.depth_camera.intrinsic_matrix,
                self.depth_camera.dist_coeffs,
            )

            if retval:
                self.depth_camera.all_rvec.append(rvec)
                self.depth_camera.all_tvec.append(tvec)
            else:
                print(f"存在第{i}组数据导致solvePnP函数无法产生正确值！")

        for arm_data in self.calibration_armgrasper_result:
            flag = True
            for idx in self.depth_camera.fail_data:
                if idx == arm_data['idx']:
                    flag = False

            if flag:
                self.r_end2base.append(arm_data['rotation_matrix'])
                self.t_end2base.append(arm_data['translation_matrix'])

        r_cam2end, t_cam2end = cv2.calibrateHandEye(
            self.r_end2base,
            self.t_end2base,
            self.depth_camera.all_rvec,
            self.depth_camera.all_tvec,
        )

        zero_vector = np.array([0, 0, 0])

        self.T_cam2end = np.block([
            [r_cam2end, t_cam2end],
            [zero_vector, 1],
        ])

        self.depth_camera.T_cam2end = self.T_cam2end
        print("手眼标定矩阵计算完成！")
        print(f"T_cam2end:\n{self.T_cam2end}")

    def run_full_calibration(self):
        self.going_calibration_capture()
        self.going_calibration_calculation()
        return self.T_cam2end