# -*- coding: utf-8 -*-
"""任务：搭建本地仿真测试环境，验证「算法 -> 指令」的输出准确率。

无需相机与实机：用几何仿真构造基坐标真值点，经相机成像 + 深度图 + 手眼外参，
走真实的 CoordinateConverter 坐标转换，得到机械臂指令位姿，再与真值比较。

运行:
    ./.venv/Scripts/python.exe scripts/simulation_accuracy.py --cases 200

退出码：准确率达到阈值返回 0，否则返回 1，可直接用于 CI 或验收。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.qiming.arm_control.pose_presets import PRESETS  # noqa: E402
from src.qiming.simulation.env import (  # noqa: E402
    SimulatedCamera,
    evaluate_conversion,
    make_extrinsic,
    sample_points,
)


def print_report(report, points, tolerance: float) -> None:
    data = report.to_dict()
    print("=" * 62)
    print("仿真测试环境：算法 -> 指令 输出准确率报告")
    print("=" * 62)
    print(f"采样目标点      : {data['total']} 个")
    print(f"成功转换        : {data['converted']} 个（{data['conversion_rate'] * 100:.2f}%）")
    print(f"位置误差容差    : {tolerance} mm")
    print(f"容差内指令      : {data['within_tolerance']} 个")
    print(f"输出准确率      : {data['accuracy'] * 100:.2f}%")
    print("-" * 62)
    print(f"位置误差 平均   : {data['mean_error_mm']:.4f} mm")
    print(f"位置误差 P95    : {data['p95_error_mm']:.4f} mm")
    print(f"位置误差 最大   : {data['max_error_mm']:.4f} mm")
    print(f"姿态不一致      : {data['orientation_mismatches']} 个")
    print("=" * 62)


def print_worst_cases(report, points, top: int = 5) -> None:
    """打印误差最大的若干个点，便于定位坐标系/单位类问题。"""
    if not report.errors_mm:
        return
    camera = SimulatedCamera()
    pairs = []
    for point in points:
        projected = camera.project_point(point)
        if projected is None:
            continue
        pairs.append((point, projected))

    ranked = sorted(zip(report.errors_mm, pairs), key=lambda item: item[0], reverse=True)[:top]
    print(f"\n误差最大的 {len(ranked)} 个目标点（真值 -> 投影像素）:")
    for error, (point, (u, v, depth)) in ranked:
        print(f"  误差 {error:8.4f} mm | 真值 {tuple(round(p, 2) for p in point)}"
              f" | 像素 ({u:.2f}, {v:.2f}) 深度 {depth:.2f} mm")


def main() -> int:
    parser = argparse.ArgumentParser(description="本地仿真环境：验证算法到指令的输出准确率")
    parser.add_argument("--cases", type=int, default=200, help="采样目标点数量（默认 200）")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子（默认 2026）")
    parser.add_argument("--tolerance", type=float, default=1.0,
                        help="位置误差容差 mm（默认 1.0）")
    parser.add_argument("--min-accuracy", type=float, default=0.98,
                        help="准确率阈值 0~1（默认 0.98）")
    parser.add_argument("--preset", default="top_down",
                        choices=sorted(PRESETS), help="末端姿态预设（默认 top_down）")
    args = parser.parse_args()

    print("相机仿真配置:")
    T_BC = make_extrinsic()
    print(f"  外参 T_BC 平移 = {tuple(round(v, 2) for v in T_BC[:3, 3])} mm")
    print(f"  末端姿态预设   = {args.preset}")
    print()

    points = sample_points(args.cases, args.seed)
    report = evaluate_conversion(
        points, preset_name=args.preset, T_BC=T_BC, tolerance_mm=args.tolerance)
    print_report(report, points, args.tolerance)
    print_worst_cases(report, points)

    if report.converted == 0:
        print("\n结论: 没有任何目标点落在视野内，请检查相机外参配置")
        return 1

    passed = report.accuracy >= args.min_accuracy
    print(f"\n结论: {'通过' if passed else '未达标'}"
          f"（准确率 {report.accuracy * 100:.2f}% vs 阈值 {args.min_accuracy * 100:.2f}%）")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
