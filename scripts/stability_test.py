# -*- coding: utf-8 -*-
"""联调一：前端语音流转至后端处理的稳定性测试脚本。

模拟前端完整链路，对运行中的后端服务做端到端压测：
    REST 注册/登录拿 token -> WebSocket 连接 -> 连续发送多轮指令
    （即语音识别产出的文本流）-> 逐轮校验事件流完整性 -> 断线重连
    校验历史上下文恢复（Memory 持久化）-> 新建会话校验上下文清空。

使用前先启动后端（见 README）：
    ./robot-arm-env/Scripts/python.exe -m uvicorn src.api.app:app --port 8000

运行本脚本（另开一个终端）：
    ./robot-arm-env/Scripts/python.exe scripts/stability_test.py --rounds 20

退出码：全部通过返回 0，任一环节失败返回 1，适合 CI 或手动验收。
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

# Windows 控制台下让中文报告正常显示
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000/ws/chat"

# 一轮指令 = (指令文本, 预期工具调用次数)。文本会被当作"语音识别结果"发给后端。
# 覆盖三种意图：闲聊（0 工具）/ 动作指令（1~N 工具）/ 系统指令（1 工具）。
VOICE_ROUNDS: list[tuple[str, int]] = [
    ("你好呀，你是谁", 0),
    ("帮我拿一下水杯", 1),
    ("查询状态", 1),
    ("先把苹果拿过来，再把杯子放到桌上", 2),
    ("帮我拿一下遥控器", 1),
]


class StabilityResult:
    """汇总测试结果的简单容器。"""

    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.failures: list[str] = []
        self.latencies: list[float] = []

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
        print("稳定性测试报告")
        print("=" * 60)
        print(f"通过检查项: {self.passed}    失败检查项: {self.failed}")
        if self.latencies:
            lat = sorted(self.latencies)
            avg = sum(lat) / len(lat)
            print(f"单轮往返延迟: 平均 {avg:.2f}s | 最小 {lat[0]:.2f}s | 最大 {lat[-1]:.2f}s")
        for f in self.failures:
            print(f"  失败项: {f}")
        print("=" * 60)
        return self.failed == 0


async def register_and_login(client: httpx.AsyncClient, result: StabilityResult) -> str | None:
    """注册一个随机测试用户并登录，返回 access token。"""
    creds = {
        "username": f"stability_{uuid.uuid4().hex[:8]}",
        "email": f"stability_{uuid.uuid4().hex[:8]}@example.com",
        "password": "Stability#123",
    }
    try:
        resp = await client.post(f"{BASE_URL}/auth/register", json=creds)
        resp.raise_for_status()
        resp = await client.post(
            f"{BASE_URL}/auth/login",
            json={"username": creds["username"], "password": creds["password"]},
        )
        resp.raise_for_status()
        token = resp.json()["access_token"]
        result.check(True, "REST 注册/登录获取 token")
        return token
    except Exception as exc:  # noqa: BLE001
        result.check(False, f"REST 注册/登录获取 token（{exc}）")
        return None


async def recv_until(ws, stop_type: str = "done", timeout: float = 15.0) -> tuple[list[dict], bool]:
    """收消息直到收到 stop_type 类型事件，返回 (事件列表, 是否超时)。"""
    events: list[dict] = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        raw = await asyncio.wait_for(ws.recv(), timeout=max(0.1, deadline - time.monotonic()))
        event = json.loads(raw)
        events.append(event)
        if event.get("type") == stop_type:
            return events, False
        if event.get("type") == "error":
            return events, False
    return events, True


async def run_round(ws, content: str, timeout: float = 15.0) -> tuple[list[dict], bool, float]:
    """发送一轮指令并收集到 done，返回 (事件, 是否超时, 往返延迟秒)。"""
    start = time.monotonic()
    await ws.send(json.dumps({"type": "text", "content": content}))
    events, timeout_hit = await recv_until(ws, timeout=timeout)
    return events, timeout_hit, time.monotonic() - start


def round_is_clean(events: list[dict]) -> bool:
    """一轮事件流必须完整：status 开头、done 收尾、无 error。"""
    types = [e.get("type") for e in events]
    return bool(types) and types[0] == "status" and types[-1] == "done" and "error" not in types


def tool_count(events: list[dict]) -> int:
    return sum(1 for e in events if e.get("type") == "tool_result")


async def test_main_flow(ws, rounds: int, result: StabilityResult) -> None:
    """阶段一：连续多轮语音指令稳定性。"""
    print(f"\n[阶段一] 连续 {rounds} 轮语音指令（混合三种意图 + 心跳穿插）")
    ok_all = True
    for i in range(rounds):
        content, expect_tools = VOICE_ROUNDS[i % len(VOICE_ROUNDS)]
        # 每轮之间穿插一次心跳，模拟前端保活
        await ws.send('{"type": "ping"}')
        pong = json.loads(await ws.recv())
        if pong.get("type") != "pong":
            ok_all = False
            result.check(False, f"第 {i + 1} 轮前心跳 pong 异常: {pong}")
            continue

        events, timeout_hit, latency = await run_round(ws, content)
        result.latencies.append(latency)
        clean = (not timeout_hit) and round_is_clean(events) and tool_count(events) >= expect_tools
        if not clean:
            ok_all = False
            types = [e.get("type") for e in events]
            print(f"    第 {i + 1} 轮异常: 指令={content!r} 事件序列={types} 延迟={latency:.2f}s")
    result.check(ok_all, f"连续 {rounds} 轮指令全部完整收尾且工具调用次数符合预期")


async def test_reconnect_history(token: str, result: StabilityResult) -> None:
    """阶段二：断线重连后历史上下文恢复（Memory 持久化）。"""
    print("\n[阶段二] 断线重连 -> 历史上下文恢复")
    marker = f"帮我拿一下魔方{uuid.uuid4().hex[:4]}"  # 唯一文本，避免历史混淆
    async with websockets.connect(f"{WS_URL}?token={token}") as ws:
        await ws.recv()  # robot_status
        await ws.recv()  # history
        events, timeout_hit, _ = await run_round(ws, marker)
        result.check((not timeout_hit) and round_is_clean(events), "重连前发送一轮指令")

    # 断开后重连（等价于前端刷新页面）
    async with websockets.connect(f"{WS_URL}?token={token}") as ws:
        await ws.recv()  # robot_status
        history = json.loads(await ws.recv())
        texts = [m.get("text", "") for m in history.get("messages", [])]
        result.check(history.get("type") == "history" and marker in texts,
                     "重连后历史对话被完整恢复（含断线前最后一轮）")


async def test_new_session(token: str, result: StabilityResult) -> None:
    """阶段三：新建会话后上下文清空。"""
    print("\n[阶段三] 新建会话 -> 上下文清空")
    async with websockets.connect(f"{WS_URL}?token={token}") as ws:
        await ws.recv()
        await ws.recv()
        await ws.send('{"type": "new_session"}')
        history = json.loads(await ws.recv())
        result.check(history.get("type") == "history" and history.get("messages") == [],
                     "新建会话后返回空历史")

        events, timeout_hit, _ = await run_round(ws, "帮我拿一下苹果")
        result.check((not timeout_hit) and round_is_clean(events) and tool_count(events) >= 1,
                     "新会话中对话正常工作")


async def main() -> int:
    parser = argparse.ArgumentParser(description="前端语音流 -> 后端处理 稳定性联调脚本")
    parser.add_argument("--rounds", type=int, default=20, help="阶段一连续指令轮数（默认 20）")
    parser.add_argument("--host", default="127.0.0.1", help="后端地址（默认 127.0.0.1）")
    parser.add_argument("--port", type=int, default=8000, help="后端端口（默认 8000）")
    args = parser.parse_args()

    global BASE_URL, WS_URL
    BASE_URL = f"http://{args.host}:{args.port}"
    WS_URL = f"ws://{args.host}:{args.port}/ws/chat"

    print("=" * 60)
    print("联调一：前端语音流转至后端处理的稳定性测试")
    print(f"目标服务: {BASE_URL}")
    print("=" * 60)

    result = StabilityResult()
    async with httpx.AsyncClient(timeout=15) as client:
        token = await register_and_login(client, result)
    if token is None:
        return 1 if not result.summary() else 1

    try:
        async with websockets.connect(f"{WS_URL}?token={token}") as ws:
            # 连接建立后固定先收到 robot_status 和 history 两条初始化消息
            first = json.loads(await ws.recv())
            second = json.loads(await ws.recv())
            result.check(first.get("type") == "robot_status", "连接后推送机械臂状态快照")
            result.check(second.get("type") == "history", "连接后推送会话历史（Memory 恢复）")
            await test_main_flow(ws, args.rounds, result)
    except Exception as exc:  # noqa: BLE001
        result.check(False, f"WebSocket 主链路异常中断: {exc}")
        return 1

    await test_reconnect_history(token, result)
    await test_new_session(token, result)

    return 0 if result.summary() else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
