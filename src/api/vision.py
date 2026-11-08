# src/api/vision.py
"""视觉与标定状态的只读接口。

状态直接读 config/robot.yaml 的 hand_eye_calibration 段：
内参与外参都填好，才认为相机链路已具备执行条件。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.dependencies import get_current_user
from src.qiming.utils.config import load_yaml

router = APIRouter(prefix="/vision", tags=["vision"])


def _calibration_status() -> dict:
    """从配置读取当前标定状态。"""
    try:
        cfg = load_yaml("config/robot.yaml") or {}
    except Exception:  # noqa: BLE001 - 配置缺失时按未标定处理
        cfg = {}
    robot = cfg.get("robot", cfg)
    calib = robot.get("hand_eye_calibration") or {}

    mode = calib.get("mode", "eye_to_hand")
    intrinsic = calib.get("intrinsic_matrix")
    dist = calib.get("dist_coeffs")
    # Eye-to-Hand 用 T_BC，Eye-in-Hand 用 T_CE
    extrinsic = calib.get("T_CE") if mode == "eye_in_hand" else calib.get("T_BC")

    intrinsic_ok = intrinsic is not None and dist is not None
    extrinsic_ok = extrinsic is not None

    return {
        "mode": mode,
        "intrinsic_calibrated": intrinsic_ok,
        "extrinsic_calibrated": extrinsic_ok,
        "hand_eye_calibrated": intrinsic_ok and extrinsic_ok,
        "source": "config/robot.yaml",
    }


@router.get(
    "/calibration/status",
    summary="查询手眼标定状态",
    description=(
        "只读查询：内参与手眼外参是否已在 config/robot.yaml 中填写。"
        "当前尚未接入真实相机，标定流程仍通过命令行 `python -m src.main --calibrate` 执行。"
    ),
)
async def calibration_status(current_user=Depends(get_current_user)) -> dict:
    return _calibration_status()
