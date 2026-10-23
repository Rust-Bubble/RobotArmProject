# -*- coding: utf-8 -*-
"""联调一：前端语音流转至后端处理的稳定性测试。

模拟前端的完整链路：WebSocket 认证连接 -> 连续发送多轮指令（即语音识别
产出的文本流）-> 逐轮校验事件流完整性（以 done 收尾、无 error、工具调用
次数符合预期）-> 断线重连后校验历史上下文恢复 -> 新建会话后上下文清空。
"""

import time

import pytest

from fastapi.testclient import TestClient

from src.api.app import app
from src.api.auth_utils import create_access_token, get_password_hash
from src.qiming.database.connection import db_manager
from src.qiming.database.models import User


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def token(client):
    async def _create():
        async with db_manager.session() as session:
            user = User(
                username="stability_tester",
                email="stability_tester@example.com",
                hashed_password=get_password_hash("password123"),
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user.id

    import asyncio

    user_id = asyncio.run(_create())
    return create_access_token({"sub": str(user_id)})


def recv_initial(ws) -> dict:
    """连接建立后固定先收到 robot_status + history 两条初始化消息。"""
    status = ws.receive_json()
    assert status["type"] == "robot_status"
    history = ws.receive_json()
    assert history["type"] == "history"
    return history


def run_round(ws, content: str) -> list[dict]:
    """发送一轮指令并收集到 done 为止，返回事件列表。"""
    ws.send_json({"type": "text", "content": content})
    events = []
    while True:
        event = ws.receive_json()
        events.append(event)
        if event["type"] == "done":
            break
    return events


def assert_round_ok(events: list[dict], content: str, min_tools: int = 0):
    """校验一轮事件流的完整性。"""
    types = [e["type"] for e in events]
    assert types[0] == "status", f"[{content}] 首事件应为 status"
    assert types[-1] == "done", f"[{content}] 必须以 done 收尾"
    assert "error" not in types, f"[{content}] 不应出现 error 事件: {events}"
    assert types[1] == "intent" and types[2] == "agent_start", f"[{content}] 事件顺序异常: {types}"
    assert any(e["type"] == "agent_result" for e in events), f"[{content}] 缺少 agent_result"

    tools = [e for e in events if e["type"] == "tool_result"]
    assert len(tools) >= min_tools, f"[{content}] 工具调用次数不足"
    assert all(e["ok"] is True for e in tools), f"[{content}] 存在失败的工具调用"


class TestVoiceFlowStability:
    def test_multi_round_mixed_intents(self, client, token):
        """连续 24 轮混合指令（闲聊/动作/系统 + 心跳穿插），全部完整收尾。"""
        rounds = [
            ("你好呀", 0),
            ("帮我拿一下水杯", 1),
            ("查询状态", 1),
            ("先把苹果拿过来，再把杯子放到桌上", 2),
            ("你是谁呀", 0),
            ("帮我拿一下遥控器", 1),
        ] * 4  # 共 24 轮

        start = time.monotonic()
        with client.websocket_connect(f"/ws/chat?token={token}") as ws:
            recv_initial(ws)
            for i, (content, min_tools) in enumerate(rounds, start=1):
                # 每轮之间穿插一次心跳，模拟前端保活
                ws.send_json({"type": "ping"})
                assert ws.receive_json() == {"type": "pong"}

                events = run_round(ws, content)
                assert_round_ok(events, content, min_tools)

        elapsed = time.monotonic() - start
        # mock 链路下 24 轮应在 30s 内完成（放宽阈值避免 CI 抖动误报）
        assert elapsed < 30, f"24 轮耗时 {elapsed:.1f}s，超出预期"

    def test_roundtrip_latency(self, client, token):
        """单轮往返延迟应稳定在秒级以内（mock 链路）。"""
        with client.websocket_connect(f"/ws/chat?token={token}") as ws:
            recv_initial(ws)
            for _ in range(5):
                start = time.monotonic()
                events = run_round(ws, "帮我拿一下水杯")
                elapsed = time.monotonic() - start
                assert_round_ok(events, "帮我拿一下水杯", 1)
                assert elapsed < 5, f"单轮延迟 {elapsed:.2f}s，超出预期"

    def test_reconnect_restores_history(self, client, token):
        """断线重连（等价于前端刷新页面）后，历史对话被完整恢复。"""
        # 用本用例独有的文本，避免与文件内其他用例的历史记录混淆
        text_a = "帮我拿一下魔方"
        text_b = "今天适合出门散步吗"
        with client.websocket_connect(f"/ws/chat?token={token}") as ws:
            recv_initial(ws)
            run_round(ws, text_a)
            run_round(ws, text_b)

        # 重连
        with client.websocket_connect(f"/ws/chat?token={token}") as ws:
            history = recv_initial(ws)
            texts = [m["text"] for m in history["messages"]]
            assert texts.count(text_a) == 1
            assert texts.count(text_b) == 1

            # 恢复后继续对话不受影响
            events = run_round(ws, "再把魔方递给我")
            assert_round_ok(events, "再把魔方递给我", 1)

    def test_new_session_clears_context(self, client, token):
        """发送 new_session 后上下文清空，从零开始新一轮对话。"""
        with client.websocket_connect(f"/ws/chat?token={token}") as ws:
            recv_initial(ws)
            run_round(ws, "帮我拿一下水杯")

            ws.send_json({"type": "new_session"})
            history = ws.receive_json()
            assert history["type"] == "history"
            assert history["messages"] == []

            # 新会话正常工作
            events = run_round(ws, "帮我拿一下苹果")
            assert_round_ok(events, "帮我拿一下苹果", 1)

    def test_empty_and_invalid_messages_do_not_break_stream(self, client, token):
        """空内容 / 非法类型消息被忽略，后续轮次不受影响。"""
        with client.websocket_connect(f"/ws/chat?token={token}") as ws:
            recv_initial(ws)
            ws.send_json({"type": "text", "content": "   "})
            ws.send_json({"type": "unknown"})
            ws.send_json({"type": "ping"})
            assert ws.receive_json() == {"type": "pong"}

            events = run_round(ws, "帮我拿一下水杯")
            assert_round_ok(events, "帮我拿一下水杯", 1)
