# -*- coding: utf-8 -*-
"""Swagger/OpenAPI 文档元数据。

各端点的 summary / responses 写在路由装饰器上，这里集中管理
全局信息（标签分组说明、错误响应模型），由 app.py 组装进 FastAPI 实例。
"""

from __future__ import annotations

from pydantic import BaseModel

# 标签分组：/docs 页面按此分组并展示说明
TAGS_METADATA = [
    {
        "name": "chat",
        "description": (
            "对话 WebSocket 端点 `/ws/chat`（浏览器无法自定义请求头，JWT 通过 "
            "`?token=` 传递）。消息协议见 `src/api/chat.py` 模块注释；"
            "连接建立后端会先推送 `robot_status` 快照与 `history` 会话历史，"
            "断线重连后同样恢复。"
        ),
    },
    {
        "name": "authentication",
        "description": "用户注册 / 登录 / Token 刷新 / 当前用户信息。",
    },
    {
        "name": "system",
        "description": "服务健康检查等系统级端点。",
    },
]


class ErrorResponse(BaseModel):
    """统一错误响应体。"""

    code: int
    message: str


# 各路由通用的错误响应说明，路由里用 responses=ERROR_RESPONSES 引用
ERROR_RESPONSES = {
    401: {"model": ErrorResponse, "description": "未认证或 Token 无效"},
    500: {"model": ErrorResponse, "description": "服务内部错误"},
}
