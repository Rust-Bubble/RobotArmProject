# -*- coding: utf-8 -*-
"""前端与后台大模型通信端点 (/ws/chat) 的集成测试。

覆盖：token 认证、ping/pong 心跳、一轮完整对话事件流。
使用 MockOrchestrator（llm.yaml mode: mock），不访问外部大模型。
"""

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
import pytest

from src.api.app import app
from src.api.auth_utils import create_access_token, get_password_hash
from src.qiming.database.connection import db_manager
from src.qiming.database.models import User


@pytest.fixture(scope="module")
def client():
    # lifespan 会初始化数据库并建表（conftest 已把 SQLITE_PATH 指到临时目录）
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def token(client):
    """创建一个测试用户并签发 access token。"""

    async def _create():
        async with db_manager.session() as session:
            user = User(
                username="ws_tester",
                email="ws_tester@example.com",
                hashed_password=get_password_hash("password123"),
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user.id

    import asyncio

    user_id = asyncio.run(_create())
    return create_access_token({"sub": str(user_id)})


class TestWsAuth:
    def test_invalid_token_rejected(self, client):
        # 4401 是约定好的"未认证"关闭码，前端据此跳转登录
        with pytest.raises(WebSocketDisconnect) as exc_info:
            client.websocket_connect("/ws/chat?token=bad.token.here").__enter__()
        assert exc_info.value.code == 4401

    def test_missing_token_rejected(self, client):
        with pytest.raises(WebSocketDisconnect):
            client.websocket_connect("/ws/chat").__enter__()


class TestWsChat:
    def test_ping_pong(self, client, token):
        with client.websocket_connect(f"/ws/chat?token={token}") as ws:
            ws.send_json({"type": "ping"})
            assert ws.receive_json() == {"type": "pong"}

    def test_full_chat_round(self, client, token):
        with client.websocket_connect(f"/ws/chat?token={token}") as ws:
            ws.send_json({"type": "text", "content": "今天天气真不错"})

            events = []
            while True:
                event = ws.receive_json()
                events.append(event)
                if event["type"] == "done":
                    break

        types = [e["type"] for e in events]
        # 闲聊消息不应触发工具调用
        assert types == ["status", "intent", "agent_start", "agent_result", "done"]
        assert events[1]["intent"] == "闲聊"
        assert "mock 回复" in events[3]["message"]

    def test_action_round_dispatches_tool(self, client, token):
        with client.websocket_connect(f"/ws/chat?token={token}") as ws:
            ws.send_json({"type": "text", "content": "帮我拿一下水杯"})

            events = []
            while True:
                event = ws.receive_json()
                events.append(event)
                if event["type"] == "done":
                    break

        tool_events = [e for e in events if e["type"] == "tool_result"]
        assert len(tool_events) == 1
        assert tool_events[0]["tool"] == "execute_grasp"
        assert tool_events[0]["ok"] is True
        assert events[-1]["type"] == "done"

    def test_invalid_message_type_ignored(self, client, token):
        with client.websocket_connect(f"/ws/chat?token={token}") as ws:
            ws.send_json({"type": "unknown_type"})
            ws.send_json({"type": "ping"})
            # 非法消息被忽略，不影响后续心跳
            assert ws.receive_json() == {"type": "pong"}
