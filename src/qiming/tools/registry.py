# -*- coding: utf-8 -*-
"""Function Calling 工具注册表。

把外部系统能力（机械臂、语音、系统状态等）封装成带 JSON Schema 的工具，
供大模型 Function Calling 调用，也可由 Orchestrator 按任务直接分发执行。

用法：
    registry = ToolRegistry()

    @registry.tool(
        name="query_robot_status",
        description="查询机械臂当前状态",
        parameters={"type": "object", "properties": {}, "required": []},
    )
    def query_robot_status() -> dict:
        return {"status": "idle"}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    """一个可被大模型 Function Calling 调用的外部工具。"""

    name: str
    description: str
    # JSON Schema 形式的参数定义，如 {"type": "object", "properties": {...}, "required": [...]}
    parameters: dict[str, Any]
    handler: Callable[..., Any]

    def to_openai_schema(self) -> dict[str, Any]:
        """转成 OpenAI Function Calling 的 tools 条目格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """工具注册表：注册、查询、执行工具。

    execute 统一返回 {"ok": bool, "result"/"error": ...}，
    保证单个工具抛异常不会打断整轮对话。
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # ---------- 注册 ----------

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"工具重复注册: {tool.name}")
        self._tools[tool.name] = tool

    def tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any] | None = None,
    ) -> Callable[[Callable], Callable]:
        """装饰器形式注册工具。parameters 缺省为无参对象。"""

        def decorator(func: Callable) -> Callable:
            self.register(
                Tool(
                    name=name,
                    description=description,
                    parameters=parameters or {"type": "object", "properties": {}, "required": []},
                    handler=func,
                )
            )
            return func

        return decorator

    # ---------- 查询 ----------

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    # ---------- 给大模型的 Function Calling 描述 ----------

    def to_openai_schemas(self) -> list[dict[str, Any]]:
        """全部工具的 OpenAI tools 列表。"""
        return [t.to_openai_schema() for t in self._tools.values()]

    def to_langchain_tools(self) -> list:
        """转成 langchain StructuredTool，可直接传给 create_agent(tools=...)。"""
        from langchain_core.tools import StructuredTool

        return [
            StructuredTool.from_function(
                func=t.handler,
                name=t.name,
                description=t.description,
            )
            for t in self._tools.values()
        ]

    # ---------- 执行 ----------

    def execute(self, name: str, arguments: dict[str, Any] | str | None = None) -> dict[str, Any]:
        """按名称执行工具。

        :param arguments: 参数字典；也兼容大模型返回的 JSON 字符串
        :return: {"ok": True, "result": ...} 或 {"ok": False, "error": ...}
        """
        tool = self._tools.get(name)
        if tool is None:
            return {"ok": False, "error": f"未注册的工具: {name}"}

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError:
                return {"ok": False, "error": f"工具 {name} 的参数不是合法 JSON"}
        arguments = arguments or {}

        try:
            result = tool.handler(**arguments)
            return {"ok": True, "result": result}
        except TypeError as exc:
            return {"ok": False, "error": f"工具 {name} 参数不匹配: {exc}"}
        except Exception as exc:  # noqa: BLE001 - 工具异常统一兜底，不上抛
            return {"ok": False, "error": f"工具 {name} 执行失败: {exc}"}
