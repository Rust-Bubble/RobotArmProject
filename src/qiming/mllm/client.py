"""MLLM API 客户端占位文件。"""

from __future__ import annotations

from src.qiming.models import ImageFrame


class MllmClient:
    """多模态大模型请求接口占位。"""

    def ask(self, user_text: str, image: ImageFrame) -> dict:
        _ = (user_text, image)
        raise NotImplementedError("后续在这里接入真实 MLLM API。")
