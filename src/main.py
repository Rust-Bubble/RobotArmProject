# -*- coding: utf-8 -*-
# 功能：项目最小流程入口，串联语音输入、摄像头拍照、MLLM 决策、坐标转换、机械臂执行和语音反馈。

from __future__ import annotations

import argparse
import cv2
import numpy as np

from src.qiming.arm_control.controller import ArmController
from src.qiming.arm_control.mycobot_driver import MyCobotDriver
from src.qiming.arm_control.mock_driver import MockArmDriver
from src.qiming.feedback.tts import TtsFeedback
from src.qiming.mllm.recognizer import MllmRecognizer
from src.qiming.speech.voice import VoiceIO
from src.qiming.utils.config import load_yaml
from src.qiming.vision.camera import Camera
from src.qiming.vision.depth_camera import DepthCamera
from src.qiming.vision.coordinate import CoordinateConverter
from src.qiming.vision.detection import YoloDetector
from src.qiming.vision.calibration import EyeToHandCalibrator


def _setup_calibration_config(depth_camera, converter, calib_cfg):
    """
    从配置文件加载标定参数 (内参矩阵、外参矩阵)
    
    参数:
        depth_camera: DepthCamera 实例
        converter: CoordinateConverter 实例
        calib_cfg: 标定配置字典
    """
    # 加载手眼标定模式
    calib_mode = calib_cfg.get("mode", "eye_to_hand")
    converter.set_calibration_mode(calib_mode)
    
    # 加载内参矩阵
    K_data = calib_cfg.get("intrinsic_matrix")
    dist_data = calib_cfg.get("dist_coeffs")
    
    if K_data is not None and dist_data is not None:
        try:
            K = np.array(K_data)
            dist = np.array(dist_data)
            depth_camera.intrinsic_matrix = K
            depth_camera.dist_coeffs = dist
            print("[配置] 内参矩阵已从配置加载")
        except Exception as e:
            print(f"[警告] 加载内参矩阵失败: {e}")
    else:
        print("[警告] 内参矩阵未配置，请在 config/robot.yaml 中填入或通过标定流程生成")
    
    # 加载 Eye-to-Hand 外参矩阵 T_BC
    T_BC_data = calib_cfg.get("T_BC")
    if T_BC_data is not None:
        try:
            T_BC = np.array(T_BC_data)
            converter.set_T_BC(T_BC)
            depth_camera.T_cam2end = T_BC
            print("[配置] T_BC 矩阵已从配置加载")
        except Exception as e:
            print(f"[警告] 加载 T_BC 矩阵失败: {e}")
    
    # 加载 Eye-in-Hand 外参矩阵 T_CE
    T_CE_data = calib_cfg.get("T_CE")
    if T_CE_data is not None:
        try:
            T_CE = np.array(T_CE_data)
            converter.set_T_CE(T_CE)
            print("[配置] T_CE 矩阵已从配置加载")
        except Exception as e:
            print(f"[警告] 加载 T_CE 矩阵失败: {e}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行启明智手简化流程。")
    parser.add_argument("--text", help="用文本模拟语音指令。")
    parser.add_argument("--calibrate", action="store_true", help="执行手眼标定流程。")
    parser.add_argument("--show-frame", action="store_true", help="显示摄像头画面。")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    robot_cfg = load_yaml("config/robot.yaml")["robot"]
    llm_cfg = load_yaml("config/llm.yaml")["llm"]
    camera_cfg = robot_cfg.get("camera", {})
    calib_cfg = robot_cfg.get("hand_eye_calibration", {})

    voice = VoiceIO()
    feedback = TtsFeedback()

    mode = robot_cfg.get("mode", "mock")
    if mode == "real":
        arm_driver = MyCobotDriver(robot_cfg)
    else:
        arm_driver = MockArmDriver(robot_cfg)

    depth_camera = DepthCamera(camera_cfg)
    detector = YoloDetector()
    mllm = MllmRecognizer(llm_cfg)
    converter = CoordinateConverter(depth_camera, arm_driver)
    arm = ArmController(arm_driver)

    # 加载标定配置 (内参矩阵、外参矩阵)
    _setup_calibration_config(depth_camera, converter, calib_cfg)

    if args.calibrate:
        print("开始Eye-to-Hand手眼标定流程...")
        calibrator = EyeToHandCalibrator(depth_camera, arm_driver)
        calibrator.calibration_times = calib_cfg.get("calibration_times", 20)
        calibrator.CHESSBOARD_SIZE = tuple(calib_cfg.get("chessboard_size", [9, 6]))
        calibrator.SQUARE_SIZE = calib_cfg.get("square_size", 0.025)
        calibrator.run_calibration()
        feedback.say("手眼标定完成。")
        return 0

    if args.show_frame:
        print("显示摄像头画面，按 'q' 退出...")
        if not depth_camera.open_camera():
            print("相机打开失败")
            return 1
        while True:
            color_frame, depth_frame = depth_camera.get_frames()
            if color_frame is None:
                break
            results = detector.yolo_model(color_frame, conf=0.5, verbose=False)
            annotated_frame = results[0].plot()
            cv2.imshow('yolo_detection', annotated_frame)
            dpt_vis = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imshow('depth', dpt_vis)
            key = cv2.waitKey(1)
            if int(key) == ord('q'):
                break
        depth_camera.close_camera()
        cv2.destroyAllWindows()
        return 0

    user_text = args.text or voice.listen()
    image = depth_camera.capture()

    decision = mllm.recognize(user_text, image)

    if decision is None:
        feedback.say("没有识别到可以执行的动作。")
        return 1

    action = converter.to_arm_action(decision, robot_cfg["handover_coords"])
    result = arm.execute(action)
    feedback.say(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())