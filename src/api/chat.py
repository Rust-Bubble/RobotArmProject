# src/api/chat.py
"""前端与后台大模型通信的 WebSocket 端点。

浏览器 WebSocket 无法自定义请求头，因此 JWT 通过 query 参数 ?token=... 传递。
消息协议（均为 JSON）：

前端 -> 后端:
    {"type": "text", "content": "帮我拿水杯"}   # 用户文本输入
    {"type": "ping"}

后端 -> 前端 (事件流，见 Orchestrator.handle):
    {"type": "status",       "stage": "...", "message": "..."}
    {"type": "intent",       "intent": "动作指令", "agent": "action"}
    {"type": "agent_start",  "agent": "action"}
    {"type": "tool_result",  "tool": "execute_grasp", "ok": true, "result": {...}}
    {"type": "agent_result", "agent": "action", "message": "...", "task": {...}}
    {"type": "error",        "message": "..."}
    {"type": "done"}
    {"type": "pong"}
"""

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from src.api.auth_utils import decode_token
from src.qiming.database.connection import db_manager
from src.qiming.database.models import User
from src.qiming.orchestrator.orchestrator import MockOrchestrator, Orchestrator
from src.qiming.utils.config import load_yaml

router = APIRouter(tags=["chat"])


def _build_orchestrator():
    """根据 config/llm.yaml 的 mode 决定用真实 MLLM 调度器还是本地 mock 调度器。"""
    try:
        mode = load_yaml("config/llm.yaml")["llm"].get("mode", "mock")
    except Exception:  # noqa: BLE001 - 配置缺失时退回 mock，保证服务可启动
        mode = "mock"
    if mode == "real":
        return Orchestrator()
    return MockOrchestrator()


async def _authenticate(token: str) -> User | None:
    """校验 WebSocket 连接的 access token，返回对应用户。"""
    try:
        payload = decode_token(token)
    except Exception:  # noqa: BLE001 - token 过期/格式错误一律视为未认证
        return None
    if payload.get("type") != "access" or not payload.get("sub"):
        return None
    async with db_manager.session() as session:
        stmt = select(User).where(User.id == int(payload["sub"]))
        result = await session.execute(stmt)
        return result.scalars().first()


@router.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket, token: str = Query(default="")):
    user = await _authenticate(token)
    if user is None:
        # 4401: 自定义关闭码，前端据此跳转登录
        await websocket.close(code=4401)
        return

    await websocket.accept()
    # 每个连接一个独立会话（独立对话历史）
    orchestrator = _build_orchestrator()

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if data.get("type") != "text":
                continue

            content = str(data.get("content") or "").strip()
            if not content:
                continue

            async for event in orchestrator.handle(content):
                await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001 - 单轮异常不断开连接，向前端报错
        try:
            await websocket.send_json({"type": "error", "message": f"服务内部错误: {exc}"})
        except Exception:  # noqa: BLE001
            pass
