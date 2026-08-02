# src/qiming/database/connection.py
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

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

        self._engine = create_async_engine(
            dsn,
            echo=True,      # 开发时打印 SQL，生产环境可设为 False
            future=True,
        )
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