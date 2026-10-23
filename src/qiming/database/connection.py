# src/qiming/database/connection.py
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base

# 加载 .env
from dotenv import load_dotenv
load_dotenv()

Base = declarative_base()

class DatabaseManager:
    def __init__(self):
        self._engine = None
        self._sessionmaker = None

    async def initialize(self):
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

        self._engine = create_async_engine(
            dsn,
            echo=True,      # 开发时打印 SQL，生产环境可设为 False
            future=True,
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

# 全局单例
db_manager = DatabaseManager()