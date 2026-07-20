"""
TTS 语音反馈占位文件。
"""
from __future__ import annotations


class TtsFeedback:
    """语音播报接口占位。"""

    def say(self, text: str) -> None:
        print(f"[语音播报] {text}")
