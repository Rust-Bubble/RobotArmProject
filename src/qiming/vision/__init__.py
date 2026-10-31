"""视觉处理模块。"""

from .camera import Camera
from .depth_camera import DepthCamera
from .coordinate import CoordinateConverter
from .calibration import EyeToHandCalibrator

__all__ = [
    "Camera",
    "DepthCamera",
    "CoordinateConverter",
    "EyeToHandCalibrator",
    "YoloDetector",
]


def __getattr__(name: str):
    """YoloDetector 延迟导入。

    目标检测依赖 ultralytics（会带上 torch 等重依赖），而坐标转换、手眼标定、
    仿真测试都不需要它。延迟导入后，未安装 ultralytics 的环境也能正常使用
    视觉坐标链路，只有在真正实例化 YoloDetector 时才要求该依赖。
    """
    if name == "YoloDetector":
        from .detection import YoloDetector

        return YoloDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
