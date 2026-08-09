# src/api/app.py
from fastapi import FastAPI
from contextlib import asynccontextmanager

from src.qiming.database.connection import db_manager, Base
from src.api.auth import router as auth_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化数据库连接并创建表
    await db_manager.initialize()
    async with db_manager._engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    # 关闭时释放资源
    await db_manager.close()

app = FastAPI(title="启明智手 API", version="0.1.0", lifespan=lifespan)

# 注册认证路由
app.include_router(auth_router)

# 根路径，用于健康检查
@app.get("/")
async def root():
    return {"message": "启明智手 API 服务运行中"}