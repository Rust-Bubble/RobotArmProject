"""语音交互入口。"""

from __future__ import annotations


class VoiceIO:
    """语音输入占位，后续替换为真实麦克风和 ASR。"""

    def listen(self) -> str:
        return input("请输入语音指令文本：")
