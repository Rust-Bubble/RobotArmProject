# src/qiming/database/connection.py
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base

# 加载 .env
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

Base = declarative_base()

class DatabaseManager:
    def __init__(self):
        self._engine = None
        self._sessionmaker = None

    async def initialize(self):
        if self._engine is not None:
            return  # 幂等：重复初始化直接跳过（断线恢复 / 测试场景都会再调）

        db_type = os.getenv("DATABASE_TYPE", "sqlite").lower()
        if db_type == "postgresql":
            host = os.getenv("DATABASE_HOST", "localhost")
            port = os.getenv("DATABASE_PORT", "5432")
            user = os.getenv("DATABASE_USER", "postgres")
            password = os.getenv("DATABASE_PASSWORD", "")
            dbname = os.getenv("DATABASE_NAME", "qiming_db")
            dsn = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{dbname}"
        else:
            # SQLite
            db_path = os.getenv("SQLITE_PATH", "./qiming.db")
            dsn = f"sqlite+aiosqlite:///{db_path}"

        sqlite_kwargs = {}
        if db_type != "postgresql":
            # 对话历史每轮都会写库，SQLite 需要放宽锁等待，否则并发写入会报 database is locked
            sqlite_kwargs["connect_args"] = {"timeout": 30}

        # SQL 回显默认关闭（避免刷爆日志），需要调试 SQL 时设 DATABASE_ECHO=true
        echo = os.getenv("DATABASE_ECHO", "false").lower() in {"1", "true", "yes"}

        self._engine = create_async_engine(
            dsn,
            echo=echo,
            future=True,
            pool_pre_ping=True,  # 取连接前先探活，数据库重启后自动恢复连接
            **sqlite_kwargs,
        )

        if db_type != "postgresql":
            # WAL 模式：读写不互斥；busy_timeout：写锁被占用时等待而不是立刻报错
            @event.listens_for(self._engine.sync_engine, "connect")
            def _set_sqlite_pragma(dbapi_conn, _record):  # noqa: ANN001
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA busy_timeout=30000")
                cursor.close()

        self._sessionmaker = async_sessionmaker(
            self._engine, expire_on_commit=False, class_=AsyncSession
        )
        logger.info("数据库已初始化（%s）", db_type)

    async def create_all(self):
        """根据 ORM 元数据建表（已存在则跳过）。"""
        if self._engine is None:
            await self.initialize()
        from src.qiming.database import models  # noqa: F401 - 确保模型已注册到 Base

        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("数据库表结构已就绪")

    async def check(self) -> bool:
        """连通性检查（供 /health 健康检查端点使用）。"""
        try:
            if self._engine is None:
                await self.initialize()
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception:  # noqa: BLE001 - 健康检查只关心通不通
            logger.exception("数据库健康检查失败")
            return False

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        if self._sessionmaker is None:
            await self.initialize()
        async with self._sessionmaker() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def close(self):
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._sessionmaker = None
            logger.info("数据库连接已关闭")

# 全局单例
db_manager = DatabaseManager()
