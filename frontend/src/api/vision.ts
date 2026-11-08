// 视觉/标定状态查询，走 vite 代理转发到 FastAPI /vision

export interface CalibrationStatus {
  mode: string;
  intrinsic_calibrated: boolean;
  extrinsic_calibrated: boolean;
  hand_eye_calibrated: boolean;
  source: string;
}

export async function fetchCalibrationStatus(
  token: string,
): Promise<CalibrationStatus> {
  const res = await fetch("/vision/calibration/status", {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) {
    throw new Error("获取标定状态失败");
  }
  return (await res.json()) as CalibrationStatus;
}
