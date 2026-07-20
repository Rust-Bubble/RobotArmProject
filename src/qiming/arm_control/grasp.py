"""功能：抓取流程占位文件，后续可在这里细化“靠近、下降、夹取、抬起、递送”等动作步骤。"""

from __future__ import annotations

from typing import List

from src.qiming.models import ArmAction


class GraspFlow:
    """抓取流程接口占位。"""

    def build_steps(self, action: ArmAction) -> List[str]:
        _ = action
        return ["打开夹爪", "移动到目标", "闭合夹爪", "移动到递送点"]
