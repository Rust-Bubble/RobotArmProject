"""MLLM 识别与决策智能体入口，负责根据用户文本和图像判断目标物体与动作意图。"""

from __future__ import annotations

from typing import Optional

from src.qiming.models import ImageFrame, MllmDecision


class MllmRecognizer:
    """MLLM 调用占位，当前仅保留接口形状。"""

    def __init__(self, llm_cfg: dict) -> None:
        self.llm_cfg = llm_cfg

    def recognize(self, user_text: str, image: ImageFrame) -> Optional[MllmDecision]:
        _ = image
        if not user_text:
            return None

        return MllmDecision(
            target_name="水杯",
            image_position=None,
            intent="抓取并递给用户",
            reply="已识别到水杯，正在执行递送动作。",
        )
