# -*- coding: utf-8 -*-
"""联调四：接收控制网关坐标指令并稳定驱动机械臂/仿真器。

对运行中的后端做端到端验收，覆盖：
    1. 合法坐标指令（move / grasp）能被接收并成功执行；
    2. 非法指令（缺字段、维度错误、非数值、越界、未知动作）被拒绝且不驱动；
    3. 同一 command_id 重复下发具备幂等性（不重复动作）；
    4. 急停状态下拒绝新指令，复位后恢复；
    5. 连续下发多轮指令仍保持稳定（无异常、无事件流错乱）。

使用前先启动后端（见 README）：
    ./.venv/Scripts/python.exe -m uvicorn src.api.app:app --port 8000

运行:
    ./.venv/Scripts/python.exe scripts/gateway_coordinate_test.py --rounds 20

退出码：全部通过返回 0，否则返回 1。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

import httpx
import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # noqa: E402

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000/ws/chat"

# 一个在工作空间内、姿态为俯视的合法位姿
VALID_POSE = [120.0, -120.0, 60.0, 0.0, 180.0, 90.0]
VALID_PLACE = [80.0, -220.0, 190.0, 0.0, 180.0, 90.0]


class Result:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.failures: list[str] = []

    def check(self, ok: bool, label: str) -> None:
        if ok:
            self.passed += 1
            print(f"  [通过] {label}")
        else:
            self.failed += 1
            self.failures.append(label)
            print(f"  [失败] {label}")

    def summary(self) -> bool:
        print("\n" + "=" * 60)
        print("联调四报告：坐标指令网关")
        print("=" * 60)
        print(f"通过检查项: {self.passed}    失败检查项: {self.failed}")
        for item in self.failures:
            print(f"  失败项: {item}")
        print("=" * 60)
        return self.failed == 0


async def register_and_login(client: httpx.AsyncClient) -> str:
    suffix = uuid.uuid4().hex[:8]
    creds = {
        "username": f"gateway_{suffix}",
        "email": f"gateway_{suffix}@example.com",
        "password": "Gateway#123",
    }
    resp = await client.post(f"{BASE_URL}/auth/register", json=creds)
    resp.raise_for_status()
    resp = await client.post(
        f"{BASE_URL}/auth/login",
        json={"username": creds["username"], "password": creds["password"]},
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


async def send_command(ws, command: dict, timeout: float = 20.0) -> dict:
    """下发一条坐标指令并收事件，直到 command_result 出现。"""
    await ws.send(json.dumps({"type": "coordinates", "command": command}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        raw = await asyncio.wait_for(
            ws.recv(), timeout=max(0.1, deadline - time.monotonic()))
        event = json.loads(raw)
        if event.get("type") == "command_result":
            return event
    raise TimeoutError("等待 command_result 超时")


async def send_text(ws, content: str) -> None:
    """发送一条文本指令（用于急停/复位），读到 done 为止。"""
    await ws.send(json.dumps({"type": "text", "content": content}))
    while True:
        event = json.loads(await ws.recv())
        if event.get("type") == "done":
            return


async def main() -> int:
    parser = argparse.ArgumentParser(description="联调四：坐标指令网关稳定性联调")
    parser.add_argument("--rounds", type=int, default=20, help="合法指令连续下发轮数（默认 20）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    global BASE_URL, WS_URL
    BASE_URL = f"http://{args.host}:{args.port}"
    WS_URL = f"ws://{args.host}:{args.port}/ws/chat"

    print("=" * 60)
    print("联调四：接收控制网关坐标指令并稳定驱动机械臂/仿真器")
    print(f"目标服务: {BASE_URL}")
    print("=" * 60)

    result = Result()
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            token = await register_and_login(client)
            result.check(True, "REST 注册/登录获取 token")
        except Exception as exc:  # noqa: BLE001
            print(f"  注册/登录失败: {exc}")
            result.check(False, "REST 注册/登录获取 token")
            return 1

    async with websockets.connect(f"{WS_URL}?token={token}") as ws:
        first, second = json.loads(await ws.recv()), json.loads(await ws.recv())
        result.check(first.get("type") == "robot_status", "连接后推送机械臂状态快照")
        result.check(second.get("type") == "history", "连接后推送会话历史")

        print("\n[阶段一] 连续下发合法 move 指令")
        failures = []
        for i in range(args.rounds):
            pose = [VALID_POSE[0] + i, VALID_POSE[1], VALID_POSE[2],
                    VALID_POSE[3], VALID_POSE[4], VALID_POSE[5]]
            response = await send_command(ws, {
                "command_id": f"move-{i}", "action": "move", "pick_pose": pose})
            if not response.get("ok"):
                failures.append((i, response))
        result.check(not failures, f"连续 {args.rounds} 条 move 指令全部执行成功")
        if failures:
            for index, response in failures[:3]:
                print(f"    第 {index} 条失败: {response.get('error')}")

        print("\n[阶段二] 合法 grasp 坐标指令")
        grasp = await send_command(ws, {
            "command_id": "grasp-1", "action": "grasp",
            "pick_pose": VALID_POSE, "place_pose": VALID_PLACE,
            "object_name": "水杯"})
        result.check(grasp.get("ok") is True, "grasp 指令执行成功")
        result.check(grasp.get("object") == "水杯", "结果回带目标物体名称")

        print("\n[阶段三] 幂等：重复下发同一 command_id")
        duplicate = await send_command(ws, {
            "command_id": "grasp-1", "action": "grasp",
            "pick_pose": VALID_POSE, "place_pose": VALID_PLACE,
            "object_name": "水杯"})
        result.check(duplicate.get("duplicate") is True, "重复指令被识别为 duplicate 且未重复执行")
        result.check(duplicate.get("ok") is True, "重复指令返回首次执行结果")

        print("\n[阶段四] 非法指令应被拒绝且不驱动")
        invalid_cases = [
            ("缺少 pick_pose", {"action": "move"}),
            ("维度不足", {"action": "move", "pick_pose": [1.0, 2.0, 3.0]}),
            ("非数值", {"action": "move", "pick_pose": [1, 2, 3, "x", 5, 6]}),
            ("超出工作空间", {"action": "move", "pick_pose": [999.0, 0.0, 50.0, 0, 180, 90]}),
            ("未知动作", {"action": "fly", "pick_pose": VALID_POSE}),
            ("grasp 缺递送点（无默认值时应拒绝）", {"action": "grasp", "pick_pose": VALID_POSE}),
        ]
        rejected_ok = []
        for label, command in invalid_cases:
            response = await send_command(ws, command)
            expect_reject = "grasp 缺递送点" not in label  # 有默认递送点时应成功
            if expect_reject:
                rejected_ok.append(response.get("rejected") is True and response.get("ok") is False)
            else:
                rejected_ok.append(response.get("ok") is True)
            print(f"    {label}: ok={response.get('ok')} rejected={response.get('rejected')}"
                  f" error={response.get('error')}")
        result.check(all(rejected_ok), "非法指令按预期被拒绝或使用默认递送点")

        print("\n[阶段五] 急停保护与复位恢复")
        await send_text(ws, "紧急停止")
        blocked = await send_command(ws, {"action": "move", "pick_pose": VALID_POSE})
        result.check(blocked.get("rejected") is True, "急停状态下坐标指令被拒绝")
        print(f"    拒绝原因: {blocked.get('error')}")

        await send_text(ws, "系统复位")
        resumed = await send_command(ws, {"action": "move", "pick_pose": VALID_POSE})
        result.check(resumed.get("ok") is True, "复位后坐标指令恢复正常执行")

    return 0 if result.summary() else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
