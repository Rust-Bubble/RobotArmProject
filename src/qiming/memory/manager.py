# -*- coding: utf-8 -*-
"""系统运行全局上下文管理器 (MemoryManager)。

职责：
    1. 进程内维护所有对话会话的运行时上下文（对话历史列表），
       Orchestrator / 子 Agent 直接读写它来保证多轮上下文；
    2. 每轮对话结束后把消息持久化到数据库（conversation_sessions /
       conversation_messages 两张表），服务重启或前端断线重连后可恢复；
    3. 管理会话生命周期：恢复最近会话 / 新建会话 / 记录一轮对话。

使用方式（见 src/api/chat.py）：
    memory = await memory_manager.get_session(user.id)   # 恢复或新建会话
    orchestrator.history = memory.history                # 与调度器共享同一份历史
    await memory_manager.record_turn(memory.session_id, text, reply, agent)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from sqlalchemy import select, update

from src.qiming.database.connection import db_manager
from src.qiming.database.models import ConversationMessage, ConversationSession

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    # 与 database/models.py 保持一致的 naive UTC 时间
    return datetime.utcnow()


@dataclass
class SessionMemory:
    """单个对话会话的运行时上下文。

    history 与 Orchestrator.history 是同一份列表引用：
    调度器每轮往里 append，这里不做重复维护。
    """

    session_id: str
    user_id: int
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)
    history: list[dict] = field(default_factory=list)  # [{"role", "text", "agent"?}]

    def to_payload(self) -> list[dict]:
        """转成可发给前端的精简结构（见 /ws/chat 的 history 事件）。"""
        return [
            {"role": h.get("role", "user"), "text": h.get("text", ""), "agent": h.get("agent") or ""}
            for h in self.history
        ]


class MemoryManager:
    """系统运行全局上下文管理器（进程级单例，见模块底部 memory_manager）。"""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionMemory] = {}
        # 防止同一用户并发连接时重复建会话
        self._lock = asyncio.Lock()

    # ---------- 会话生命周期 ----------

    async def get_session(self, user_id: int, session_id: Optional[str] = None) -> SessionMemory:
        """恢复（或新建）指定用户的会话上下文。

        查找顺序：
            1. session_id 命中进程内缓存 -> 直接返回；
            2. session_id 命中数据库（且属于该用户）-> 从库中加载历史；
            3. 未指定/无效 session_id -> 取该用户最近更新的会话（跨重启恢复）；
            4. 都没有 -> 新建一个会话。
        """
        if session_id:
            memory = self._sessions.get(session_id)
            if memory is not None and memory.user_id == user_id:
                return memory

        async with self._lock:
            # 双检：等锁期间可能已被同一用户的另一个连接创建
            if session_id:
                memory = self._sessions.get(session_id)
                if memory is not None and memory.user_id == user_id:
                    return memory

            async with db_manager.session() as session:
                row: ConversationSession | None = None
                if session_id:
                    stmt = select(ConversationSession).where(
                        ConversationSession.id == session_id,
                        ConversationSession.user_id == user_id,
                    )
                    row = (await session.execute(stmt)).scalars().first()

                if row is None:
                    stmt = (
                        select(ConversationSession)
                        .where(ConversationSession.user_id == user_id)
                        .order_by(ConversationSession.updated_at.desc())
                        .limit(1)
                    )
                    row = (await session.execute(stmt)).scalars().first()

                if row is None:
                    row = ConversationSession(
                        id=uuid.uuid4().hex, user_id=user_id, created_at=_utcnow(), updated_at=_utcnow()
                    )
                    session.add(row)
                    await session.commit()
                    memory = SessionMemory(session_id=row.id, user_id=user_id)
                    logger.info("为用户 %s 新建会话 %s", user_id, row.id)
                else:
                    memory = SessionMemory(
                        session_id=row.id, user_id=user_id,
                        created_at=row.created_at, updated_at=row.updated_at,
                    )
                    stmt = (
                        select(ConversationMessage)
                        .where(ConversationMessage.session_id == row.id)
                        .order_by(ConversationMessage.id.asc())
                    )
                    rows = (await session.execute(stmt)).scalars().all()
                    memory.history = [
                        {"role": m.role, "text": m.content, "agent": m.agent or ""}
                        for m in rows
                    ]
                    logger.info("恢复用户 %s 会话 %s（%d 条历史消息）", user_id, row.id, len(rows))

            self._sessions[memory.session_id] = memory
            return memory

    async def reset_session(self, user_id: int) -> SessionMemory:
        """新建一个全新会话（旧会话记录保留在数据库中，可追溯）。"""
        async with self._lock:
            async with db_manager.session() as session:
                row = ConversationSession(
                    id=uuid.uuid4().hex, user_id=user_id, created_at=_utcnow(), updated_at=_utcnow()
                )
                session.add(row)
                await session.commit()
                memory = SessionMemory(session_id=row.id, user_id=user_id)
            self._sessions[memory.session_id] = memory
            logger.info("为用户 %s 新建会话 %s", user_id, row.id)
            return memory

    # ---------- 持久化 ----------

    async def record_turn(
        self, session_id: str, user_text: str, reply: str, agent: Optional[str] = None
    ) -> None:
        """把一轮对话（用户输入 + 助手回复）持久化到数据库。

        进程内历史由 Orchestrator 维护，这里只负责落库；
        落库失败只记日志，不影响本轮对话正常返回。
        """
        async with db_manager.session() as session:
            session.add(
                ConversationMessage(session_id=session_id, role="user", content=user_text)
            )
            session.add(
                ConversationMessage(
                    session_id=session_id, role="agent", content=reply, agent=agent
                )
            )
            await session.execute(
                update(ConversationSession)
                .where(ConversationSession.id == session_id)
                .values(updated_at=_utcnow())
            )
            await session.commit()
        logger.debug("会话 %s 已持久化一轮对话", session_id)

    async def close(self) -> None:
        """服务关闭时清空进程内缓存（数据库中的记录保留）。"""
        self._sessions.clear()
        logger.info("MemoryManager 已关闭，进程内会话缓存已清空")


# 进程级全局单例：整个后端共享这一份全局上下文
memory_manager = MemoryManager()
