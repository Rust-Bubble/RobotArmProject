# -*- coding: utf-8 -*-
# 功能：项目最小流程入口，串联语音输入、摄像头拍照、MLLM 决策、坐标转换、机械臂执行和语音反馈。

from __future__ import annotations

import argparse
import cv2

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
from src.qiming.vision.calibration import HandEyeCalibrator


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

    if args.calibrate:
        print("开始手眼标定流程...")
        calibrator = HandEyeCalibrator(depth_camera, arm_driver)
        calibrator.run_full_calibration()
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