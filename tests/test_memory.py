# -*- coding: utf-8 -*-
"""全局上下文管理 (MemoryManager) 的测试。

覆盖：会话新建/恢复、一轮对话持久化、跨管理器实例（模拟服务重启）恢复、
会话隔离、新建会话清空上下文。
"""

import asyncio

import pytest

from fastapi.testclient import TestClient

from src.api.app import app
from src.api.auth_utils import get_password_hash
from src.qiming.database.connection import db_manager
from src.qiming.database.models import User
from src.qiming.memory.manager import MemoryManager


@pytest.fixture(scope="module")
def client():
    # 启动 TestClient 触发 lifespan 建表（conftest 已把 SQLITE_PATH 指到临时目录）
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def user_id(client):
    """创建一个 memory 专用测试用户。"""

    async def _create():
        async with db_manager.session() as session:
            user = User(
                username="memory_tester",
                email="memory_tester@example.com",
                hashed_password=get_password_hash("password123"),
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user.id

    return asyncio.run(_create())


def test_create_session_and_record_turn(user_id):
    mgr = MemoryManager()
    memory = asyncio.run(mgr.get_session(user_id))

    assert memory.user_id == user_id
    assert memory.history == []

    # 模拟一轮对话：Orchestrator 写内存，MemoryManager 落库
    memory.history.append({"role": "user", "text": "帮我拿水杯"})
    memory.history.append({"role": "agent", "text": "好的，正在为您拿取水杯", "agent": "decompose"})
    asyncio.run(mgr.record_turn(memory.session_id, "帮我拿水杯", "好的，正在为您拿取水杯", "decompose"))

    # 进程内缓存命中
    again = asyncio.run(mgr.get_session(user_id, memory.session_id))
    assert again is memory
    asyncio.run(mgr.close())


def test_persistence_across_restart(user_id):
    """新建管理器实例（模拟服务重启）后，历史从数据库恢复。"""
    # 第一段：写入一轮对话（复用上一个用例落库的记录）
    mgr1 = MemoryManager()
    memory1 = asyncio.run(mgr1.get_session(user_id))
    assert memory1.history, "应能从数据库恢复上一用例的记录"
    asyncio.run(mgr1.close())

    # 第二段：重启后恢复
    mgr2 = MemoryManager()
    memory2 = asyncio.run(mgr2.get_session(user_id))
    assert memory2.session_id == memory1.session_id
    assert memory2.history == memory1.history
    assert memory2.history[0] == {"role": "user", "text": "帮我拿水杯", "agent": ""}
    asyncio.run(mgr2.close())


def test_reset_session_starts_fresh(user_id):
    mgr = MemoryManager()
    old = asyncio.run(mgr.get_session(user_id))
    # 模拟 Orchestrator 写内存 + MemoryManager 落库
    old.history.append({"role": "user", "text": "你好"})
    asyncio.run(mgr.record_turn(old.session_id, "你好", "你好呀"))

    fresh = asyncio.run(mgr.reset_session(user_id))
    assert fresh.session_id != old.session_id
    assert fresh.history == []

    # 旧会话仍可按 session_id 恢复（数据库记录保留）
    restored = asyncio.run(mgr.get_session(user_id, old.session_id))
    assert restored.session_id == old.session_id
    assert any(h["text"] == "你好" for h in restored.history)
    asyncio.run(mgr.close())


def test_sessions_isolated_between_users(user_id):
    async def _create():
        async with db_manager.session() as session:
            user = User(
                username="memory_tester2",
                email="memory_tester2@example.com",
                hashed_password=get_password_hash("password123"),
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user.id

    other_id = asyncio.run(_create())
    mgr = MemoryManager()

    async def _scenario():
        m1 = await mgr.get_session(user_id)
        m2 = await mgr.get_session(other_id)
        assert m1.session_id != m2.session_id
        m1.history.append({"role": "user", "text": "只属于用户1的消息"})
        await mgr.record_turn(m1.session_id, "只属于用户1的消息", "收到")
        # 用户2的会话不受影响
        assert all("只属于用户1" not in h["text"] for h in m2.history)
        # 用户2重连后拿到的仍是自己的会话
        m2b = await mgr.get_session(other_id)
        assert m2b.session_id == m2.session_id

    asyncio.run(_scenario())
    asyncio.run(mgr.close())
