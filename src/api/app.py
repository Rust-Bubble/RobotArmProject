# src/api/app.py
"""FastAPI 应用入口：路由注册、日志/数据库初始化、全局异常捕获与文档配置。

- Swagger UI: /docs   （各端点的请求/响应结构自动生成）
- ReDoc:      /redoc
- OpenAPI JSON: /openapi.json
- 健康检查:   /health（含数据库连通性，供运维探活与前端断线提示）
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.auth import router as auth_router
from src.api.chat import router as chat_router
from src.api.docs import ERROR_RESPONSES, TAGS_METADATA, ErrorResponse
from src.qiming.database.connection import db_manager
from src.qiming.memory.manager import memory_manager
from src.qiming.utils.logging import configure_logging

logger = logging.getLogger(__name__)

API_VERSION = "0.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时初始化日志与数据库连接并创建表
    configure_logging()
    logger.info("启明智手 API 启动中（v%s）…", API_VERSION)
    try:
        await db_manager.initialize()
        await db_manager.create_all()
        logger.info("服务就绪")
    except Exception:
        logger.exception("启动阶段初始化失败")
        raise
    yield
    await db_manager.close()
    logger.info("服务已关闭")


app = FastAPI(
    title="启明智手 API",
    description=(
        "面向视障用户的多模态大模型机械臂助手后端。\n\n"
        "能力：用户认证（JWT）、对话 WebSocket（`/ws/chat`）、机械臂状态推送、\n"
        "坐标指令控制网关、会话上下文持久化与断线恢复。"
    ),
    version=API_VERSION,
    lifespan=lifespan,
    openapi_tags=TAGS_METADATA,
    # Swagger UI / ReDoc 默认地址即 /docs 与 /redoc，显式写出便于检索
    docs_url="/docs",
    redoc_url="/redoc",
)

app = FastAPI(title="启明智手 API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(chat_router)


# ---------- 中间件：请求日志 + 兜底异常捕获 ----------

@app.middleware("http")
async def log_requests_and_catch_errors(request: Request, call_next):
    """记录每个 HTTP 请求的耗时与状态码；未捕获异常统一转成 500 JSON 并记录堆栈。"""
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("请求 %s %s 处理时发生未捕获异常", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"code": 500, "message": "服务内部错误，请稍后重试"},
        )
    elapsed_ms = (time.perf_counter() - start) * 1000
    # 健康检查探活请求不打日志，避免刷屏
    if request.url.path != "/health":
        logger.info(
            "%s %s -> %d (%.1fms)", request.method, request.url.path,
            response.status_code, elapsed_ms,
        )
    return response


# ---------- 全局异常处理器（结构化错误响应） ----------

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """HTTPException（401/404/400…）统一为 {code, message} 结构。"""
    # 401/403 属于正常业务流程，只记 info；其余记 warning 便于排查
    log = logger.info if exc.status_code < 500 else logger.warning
    log("HTTP %d 于 %s %s: %s", exc.status_code, request.method, request.url.path, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.status_code, "message": str(exc.detail)},
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """请求体校验失败（422）返回简化后的字段错误信息。"""
    errors = [
        {"field": ".".join(str(loc) for loc in err["loc"][1:]), "msg": err["msg"]}
        for err in exc.errors()
    ]
    logger.info("请求 %s %s 参数校验失败: %s", request.method, request.url.path, errors)
    return JSONResponse(
        status_code=422,
        content={"code": 422, "message": "请求参数不合法", "detail": errors},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """中间件之外的兜底：任何未处理异常都记录完整堆栈，不向外泄漏内部细节。"""
    logger.exception("未处理异常于 %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"code": 500, "message": "服务内部错误，请稍后重试"},
    )


# ---------- 系统端点 ----------

@app.get(
    "/",
    tags=["system"],
    summary="服务欢迎信息",
    responses=ERROR_RESPONSES,
)
async def root():
    return {"message": "启明智手 API 服务运行中"}


@app.get(
    "/health",
    tags=["system"],
    summary="健康检查",
    description="检查服务自身与数据库连通性。任一不健康返回 503，供运维探活。",
    response_model=dict,
    responses={
        503: {"model": ErrorResponse, "description": "数据库不可达"},
        **ERROR_RESPONSES,
    },
)
async def health():
    db_ok = await db_manager.check()
    body = {"status": "ok" if db_ok else "degraded", "database": "up" if db_ok else "down"}
    return JSONResponse(status_code=200 if db_ok else 503, content=body)
