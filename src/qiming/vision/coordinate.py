"""功能：坐标转换占位文件，负责把 MLLM/图像位置转换为机械臂可执行坐标。"""

from __future__ import annotations

from typing import List

from src.qiming.models import ArmAction, MllmDecision, Pose


class CoordinateConverter:
    """图像坐标到机械臂坐标的转换接口占位。"""

    def to_arm_action(self, decision: MllmDecision, handover_coords: List[float]) -> ArmAction:
        pick_pose: Pose = (120.0, -120.0, 100.0, 0.0, 0.0, 0.0)
        place_pose: Pose = tuple(handover_coords)  # type: ignore[assignment]
        return ArmAction(
            target_name=decision.target_name,
            pick_pose=pick_pose,
            place_pose=place_pose,
            reply=decision.reply,
        )
