# -*- coding: utf-8 -*-
"""机械臂复位脚本：清错、回零位、打开夹爪，恢复到可执行任务的初始状态。

接口参考 docs/MyCobot320_Python_API.md：
    - clear_error_information()  清除固件错误码
    - set_fresh_mode(0)          切回插补模式（指令按队列顺序执行）
    - power_on() / is_power_on() 确保机械臂上电
    - sync_send_angles(...)      同步回到零位，到位后返回
    - set_gripper_mode(0) + init_electric_gripper() + set_gripper_value()
                                夹爪置为打开
    - release_all_servos()       可选：放松所有关节（--relax）

注意：复位会独占串口，运行前请先停止占用串口的后端（uvicorn src.api.app），
否则机械臂读写可能全部返回 -1。

运行:
    python scripts/reset_robot.py
    python scripts/reset_robot.py --home "0,0,0,0,0,0" --speed 30
    python scripts/reset_robot.py --relax --keep-gripper

退出码：复位成功返回 0，否则返回 1。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pymycobot import MyCobot320, PI_PORT  # noqa: E402

from src.qiming.utils.config import load_yaml  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "robot.yaml"
HOME_ANGLES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def parse_angles(text: str) -> list[float]:
    values = [float(v) for v in text.replace("，", ",").split(",") if v.strip()]
    if len(values) != 6:
        raise argparse.ArgumentTypeError(f"需要 6 个关节角度，收到 {len(values)} 个: {text}")
    return values


def read_status(mc: MyCobot320) -> None:
    """打印上电、连接、错误与当前位姿，便于确认复位结果。"""
    try:
        print(f"  上电状态      : {mc.is_power_on()} (1=上电, 0=掉电)")
    except Exception as e:  # noqa: BLE001
        print(f"  上电状态      : 读取失败({e})")
    try:
        print(f"  通信连接      : {mc.is_controller_connected()} (1=连接, 0=未连接)")
    except Exception as e:  # noqa: BLE001
        print(f"  通信连接      : 读取失败({e})")
    try:
        print(f"  固件错误码    : {mc.get_error_information()} (0=正常)")
    except Exception as e:  # noqa: BLE001
        print(f"  固件错误码    : 读取失败({e})")
    try:
        angles = mc.get_angles()
        coords = mc.get_coords()
        print(f"  当前关节角度  : {angles}")
        print(f"  当前末端坐标  : {coords}")
    except Exception as e:  # noqa: BLE001
        print(f"  当前位姿      : 读取失败({e})")


def open_gripper(mc: MyCobot320, gripper_cfg: dict) -> None:
    gripper_type = gripper_cfg.get("type", 1)
    open_value = gripper_cfg.get("open_value", 80)
    open_speed = gripper_cfg.get("open_speed", 50)
    try:
        mc.set_gripper_mode(0)
        mc.init_electric_gripper()
        mc.set_gripper_value(open_value, open_speed, gripper_type)
        time.sleep(0.8)
        print(f"  夹爪          : 已打开（value={open_value}, speed={open_speed}, type={gripper_type}）")
    except Exception as e:  # noqa: BLE001
        print(f"  夹爪          : 打开失败({e})，回退 set_gripper_state")
        try:
            mc.set_gripper_state(0, open_speed, gripper_type)
        except Exception as e2:  # noqa: BLE001
            print(f"  夹爪          : 仍失败({e2})")


def move_home(mc: MyCobot320, angles: list[float], speed: int, timeout: int) -> bool:
    """同步回到零位，超时或异常时回退到异步发送，并用 is_in_position 校验。"""
    try:
        mc.sync_send_angles(angles, speed, timeout)
    except Exception as e:  # noqa: BLE001
        print(f"  同步回零失败({e})，回退到异步 send_angles")
        try:
            mc.send_angles(angles, speed)
        except Exception as e2:  # noqa: BLE001
            print(f"  异步回零失败({e2})")
            return False

    for _ in range(max(1, timeout * 2)):
        try:
            if mc.is_in_position(angles, 0) == 1:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
    print("  警告: 未在超时时间内确认到达零位，请人工检查")
    return True


def reset_robot(mc: MyCobot320, args: argparse.Namespace, gripper_cfg: dict) -> bool:
    print("[1/6] 检查上电与通信")
    if mc.is_power_on() != 1:
        print("  未上电，调用 power_on()")
        mc.power_on()
        time.sleep(1)
    if mc.is_controller_connected() == 0:
        print("  错误: Atom 通信未连接，请检查串口/波特率")
        return False

    print("[2/6] 切回插补模式并清除错误码")
    mc.set_fresh_mode(0)
    mc.clear_error_information()
    time.sleep(0.5)
    code = mc.get_error_information()
    if code not in (None, 0):
        print(f"  错误码仍为 {code}（1~6 关节超限 / 16~19 碰撞 / 32 逆解无解 / 33~34 直线无相邻解）")
        if not args.force:
            print("  已中止复位，排查后可使用 --force 强制执行")
            return False
        print("  已指定 --force，继续执行")

    print(f"[3/6] 回零位 {args.home}（speed={args.speed}, timeout={args.timeout}s）")
    if not move_home(mc, args.home, args.speed, args.timeout):
        return False

    if args.keep_gripper:
        print("[4/6] 跳过夹爪（--keep-gripper）")
    else:
        print("[4/6] 打开夹爪")
        open_gripper(mc, gripper_cfg)

    if args.relax:
        print("[5/6] 放松所有关节")
        mc.release_all_servos()
    else:
        print("[5/6] 保持关节上电（如需放松请加 --relax）")

    print("[6/6] 复位后状态")
    read_status(mc)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="MyCobot 320 复位脚本")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="机械臂配置文件路径")
    parser.add_argument("--port", default=None, help="串口，默认读取配置中的 port")
    parser.add_argument("--baudrate", type=int, default=None, help="波特率，默认读取配置中的 baudrate")
    parser.add_argument("--speed", type=int, default=None, help="回零速度 1~100，默认读取 default_speed")
    parser.add_argument("--timeout", type=int, default=None, help="到位等待超时秒数，默认读取 move_timeout")
    parser.add_argument("--home", type=parse_angles, default=HOME_ANGLES,
                        help='零位关节角度，逗号分隔 6 个值，默认 "0,0,0,0,0,0"')
    parser.add_argument("--relax", action="store_true", help="复位后放松所有关节")
    parser.add_argument("--keep-gripper", action="store_true", help="不操作夹爪")
    parser.add_argument("--force", action="store_true", help="存在固件错误码时仍强制执行")
    parser.add_argument("--check", action="store_true", help="仅读取状态，不做任何复位动作")
    args = parser.parse_args()

    cfg = load_yaml(args.config).get("robot", {})
    port = args.port or cfg.get("port", PI_PORT)
    baudrate = args.baudrate or cfg.get("baudrate", 115200)
    args.speed = args.speed or cfg.get("default_speed", 35)
    args.timeout = args.timeout or cfg.get("move_timeout", 15)
    gripper_cfg = cfg.get("gripper", {})

    print("=" * 60)
    print("机械臂复位")
    print(f"  配置文件: {args.config}")
    print(f"  串口/波特率: {port} @ {baudrate}")
    print("=" * 60)

    try:
        mc = MyCobot320(port, baudrate)
    except Exception as e:  # noqa: BLE001
        print(f"机械臂连接失败: {e}")
        return 1
    print(f"机械臂连接成功: {port} @ {baudrate}\n")

    try:
        if args.check:
            read_status(mc)
            return 0
        ok = reset_robot(mc, args, gripper_cfg)
    finally:
        try:
            mc.close()
        except Exception:  # noqa: BLE001
            pass

    print("\n" + "=" * 60)
    print("复位完成" if ok else "复位未完成，请查看上面的错误信息")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
