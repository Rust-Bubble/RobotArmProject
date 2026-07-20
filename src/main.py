# -*- coding: utf-8 -*-
# 功能：项目最小流程入口，串联语音输入、摄像头拍照、MLLM 决策、坐标转换、机械臂执行和语音反馈。

from __future__ import annotations

import argparse

from src.qiming.arm_control.controller import ArmController
from src.qiming.feedback.tts import TtsFeedback
from src.qiming.mllm.recognizer import MllmRecognizer
from src.qiming.speech.voice import VoiceIO
from src.qiming.utils.config import load_yaml
from src.qiming.vision.camera import Camera
from src.qiming.vision.coordinate import CoordinateConverter


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行启明智手简化流程。")
    parser.add_argument("--text", help="用文本模拟语音指令。")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    robot_cfg = load_yaml("config/robot.yaml")["robot"]
    llm_cfg = load_yaml("config/llm.yaml")["llm"]

    voice = VoiceIO()
    camera = Camera()
    mllm = MllmRecognizer(llm_cfg)
    converter = CoordinateConverter()
    arm = ArmController.from_config(robot_cfg)
    feedback = TtsFeedback()

    user_text = args.text or voice.listen()
    image = camera.capture()
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
