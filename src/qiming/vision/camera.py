"""功能：摄像头拍照入口，当前返回空图像，后续替换为真实摄像头采集。"""

from __future__ import annotations

from src.qiming.models import ImageFrame


class Camera:
    """摄像头采集占位。"""

    def capture(self) -> ImageFrame:
        return ImageFrame()
