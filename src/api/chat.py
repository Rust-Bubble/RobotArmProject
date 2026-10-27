# src/api/chat.py
"""前端与后台大模型通信的 WebSocket 端点。

浏览器 WebSocket 无法自定义请求头，因此 JWT 通过 query 参数 ?token=... 传递。
消息协议（均为 JSON）：

前端 -> 后端:
    {"type": "text", "content": "帮我拿水杯"}   # 用户文本输入（语音识别结果走同一条消息）
    {"type": "coordinates", "command": {...}}    # 控制网关下发的坐标指令（见 gateway.py）
    {"type": "new_session"}                     # 新建会话（清空上下文重新开始）
    {"type": "ping"}

后端 -> 前端 (事件流，见 Orchestrator.handle):
    {"type": "history",      "session_id": "...", "messages": [{role, text, agent}, ...]}
    {"type": "robot_status", "state": "busy", "phase": "...", ...}
    {"type": "status",       "stage": "intent", "message": "..."}
    {"type": "intent",       "intent": "动作指令", "agent": "action"}
    {"type": "agent_start",  "agent": "action"}
    {"type": "step",         "index": 1, "total": 2, "action_type": "grasp", "target_object": "水杯"}
    {"type": "tool_result",  "tool": "execute_grasp", "ok": true, "result": {...}}
    {"type": "command_result", "ok": true, "action": "move", ...}   # 坐标指令执行结果
    {"type": "agent_result", "agent": "action", "message": "...", "task": {...}}
    {"type": "error",        "message": "..."}
    {"type": "done"}
    {"type": "pong"}

连接建立后会先推送一条 robot_status 快照和一条 history 消息（由全局上下文
管理器 MemoryManager 从数据库恢复该用户最近的会话），供前端初始化。
断线重连：前端重连后后端会重新推送历史与状态快照，会话上下文不丢失。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from src.api.auth_utils import decode_token
from src.qiming.arm_control.status import status_hub
from src.qiming.database.connection import db_manager
from src.qiming.database.models import User
from src.qiming.memory.manager import SessionMemory, memory_manager
from src.qiming.orchestrator.orchestrator import MockOrchestrator, Orchestrator
from src.qiming.tools.builtin import get_coordinate_gateway
from src.qiming.utils.async_utils import run_with_status_events
from src.qiming.utils.config import load_yaml

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


def _build_orchestrator():
    """根据 config/llm.yaml 的 mode 决定用真实 MLLM 调度器还是本地 mock 调度器。"""
    try:
        mode = load_yaml("config/llm.yaml")["llm"].get("mode", "mock")
    except Exception:  # noqa: BLE001 - 配置缺失时退回 mock，保证服务可启动
        logger.warning("config/llm.yaml 读取失败，回退到 mock 调度器")
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


async def _push_history(websocket: WebSocket, memory: SessionMemory) -> None:
    """把会话历史推给前端，供其恢复对话界面。"""
    await websocket.send_json(
        {"type": "history", "session_id": memory.session_id, "messages": memory.to_payload()}
    )


@router.websocket("/ws/chat")
async def chat_ws(websocket: WebSocket, token: str = Query(default="")):
    user = await _authenticate(token)
    if user is None:
        # 必须先 accept 再携带 4401 关闭：uvicorn 等服务器对"accept 之前 close"
        # 会直接以 HTTP 403 拒绝握手，浏览器拿不到 4401 关闭码，无法区分
        # "token 失效"和"网络故障"，会导致前端无限重连
        await websocket.accept()
        await websocket.close(code=4401)
        return

    await websocket.accept()
    # 每个连接一个独立调度器，但对话历史挂在全局上下文管理器的会话上：
    # 重连 / 服务重启后历史都能从数据库恢复
    orchestrator = _build_orchestrator()
    memory = await memory_manager.get_session(user.id)
    orchestrator.history = memory.history
    logger.info("用户 %s 的 WebSocket 已连接（会话 %s）", user.id, memory.session_id)

    # 先推一条机械臂状态快照，让前端状态卡片在连接建立时就有内容
    await websocket.send_json({"type": "robot_status", **status_hub.snapshot()})
    # 再推一条会话历史，让前端恢复之前的对话
    await _push_history(websocket, memory)

    async def _send(event: dict) -> bool:
        """发送事件；连接已断开时返回 False 而不是抛异常。"""
        try:
            await websocket.send_json(event)
            return True
        except Exception:  # noqa: BLE001 - 断线后发送失败属正常流程
            return False

    try:
        while True:
            try:
                data = await websocket.receive_json()
            except ValueError:
                # 非 JSON 消息：提示前端但不断开连接
                logger.warning("收到非 JSON 消息，已忽略")
                await _send({"type": "error", "message": "消息必须是 JSON 格式"})
                continue

            if data.get("type") == "ping":
                await _send({"type": "pong"})
                continue

            if data.get("type") == "new_session":
                # 新建会话：清空上下文重新开始（旧会话记录保留在数据库中）
                memory = await memory_manager.reset_session(user.id)
                orchestrator.history = memory.history
                await _push_history(websocket, memory)
                continue

            if data.get("type") == "coordinates":
                # 控制网关坐标指令：校验、串行驱动、幂等由 CoordinateGateway 负责，
                # 执行期间同样把机械臂状态实时推给前端
                command = data.get("command") or {}
                holder: dict = {}
                try:
                    async for event in run_with_status_events(
                        get_coordinate_gateway().submit, holder, command
                    ):
                        await _send(event)
                    await _send({"type": "command_result", **holder["result"]})
                except Exception as exc:  # noqa: BLE001 - 单条指令失败不影响连接
                    logger.exception("坐标指令执行失败: %s", command)
                    await _send({"type": "error", "message": f"坐标指令执行失败: {exc}"})
                continue

            if data.get("type") != "text":
                # 未知的消息类型直接忽略（与既有约定一致，兼容前端旧版本）
                continue

            content = str(data.get("content") or "").strip()
            if not content:
                continue

            reply, agent_name = "", ""
            async for event in orchestrator.handle(content):
                if event.get("type") == "agent_result":
                    reply = str(event.get("message") or "")
                    agent_name = str(event.get("agent") or "")
                if event.get("type") == "done" and reply:
                    # 在发送 done 之前先落库：保证前端收到"本轮结束"时
                    # 上下文已持久化，即使它随即断开（如刷新页面）也不会丢这一轮
                    try:
                        await memory_manager.record_turn(
                            memory.session_id, content, reply, agent_name or None)
                    except Exception:  # noqa: BLE001
                        logger.exception("会话 %s 持久化对话失败", memory.session_id)
                if not await _send(event):
                    # 发送失败说明前端已断开，结束本轮循环去走重连恢复流程
                    break
    except WebSocketDisconnect:
        logger.info("用户 %s 的 WebSocket 已断开（会话 %s）", user.id, memory.session_id)
    except Exception:
        # 连接级异常：记录完整堆栈，向前端报错后关闭，等待前端自动重连
        logger.exception("用户 %s 的 WebSocket 发生异常", user.id)
        try:
            await websocket.close(code=1011)
        except Exception:  # noqa: BLE001 - 连接可能已经断开
            pass
