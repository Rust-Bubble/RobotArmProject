"""ASR 语音识别"""

from __future__ import annotations


class AsrRecognizer:
    """语音识别接口占位。"""

    def recognize(self, audio_data: bytes) -> str:
        _ = audio_data
        raise NotImplementedError("后续在这里接入真实 ASR。")
