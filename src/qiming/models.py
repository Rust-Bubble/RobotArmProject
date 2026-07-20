# -*- coding: utf-8 -*-
# 功能：定义简化流程中各模块之间传递的基础数据结构。

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple


Pose = Tuple[float, float, float, float, float, float]


@dataclass
class ImageFrame:
    """摄像头采集到的一帧图像。"""

    data: Optional[Any] = None


@dataclass
class MllmDecision:
    """MLLM 对图像和用户指令的理解结果。"""

    target_name: str
    image_position: Optional[Tuple[float, float]]
    intent: str
    reply: str


@dataclass
class ArmAction:
    """传递给机械臂执行层的动作。"""

    target_name: str
    pick_pose: Pose
    place_pose: Pose
    reply: str
