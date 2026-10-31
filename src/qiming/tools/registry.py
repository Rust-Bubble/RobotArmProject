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
from typing import Any, Callable, Literal

# JSON Schema 基础类型 -> Python 类型（用于生成 Function Calling 的 args_schema）
_SCHEMA_TYPES: dict[str, Any] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _schema_to_model(tool_name: str, schema: dict[str, Any]):
    """把工具注册时的 JSON Schema 转成 pydantic 模型，作为 Function Calling 参数结构。"""
    from pydantic import create_model

    required = set(schema.get("required", []))
    fields: dict[str, Any] = {}
    for key, prop in (schema.get("properties") or {}).items():
        if "enum" in prop:
            value_type: Any = Literal[tuple(prop["enum"])]  # type: ignore[valid-type]
        else:
            value_type = _SCHEMA_TYPES.get(prop.get("type"), str)
        fields[key] = (value_type, ... if key in required else None)
    return create_model(f"{tool_name}_Args", **fields)


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

    def to_langchain_tools(self, trace: list[dict[str, Any]] | None = None,
                           names: tuple[str, ...] | list[str] | None = None) -> list:
        """转成 langchain StructuredTool，可直接传给 create_agent(tools=...)。

        :param trace: 传入列表时，每次调用追加 {"tool", "args", "ok", "result"/"error"}
        :param names: 只导出这些工具（按 Agent 职责收窄权限），None 表示全部
        """
        from langchain_core.tools import StructuredTool

        selected = [t for t in self._tools.values()
                    if names is None or t.name in names]
        tools = []
        for tool in selected:
            args_schema = _schema_to_model(tool.name, tool.parameters)

            def _run(_name: str = tool.name, _trace=trace, **kwargs: Any) -> Any:
                # pydantic 会把可选参数补成 None，未显式传值的应回落到函数默认值
                actual = {k: v for k, v in kwargs.items() if v is not None}
                result = self.execute(_name, actual)
                if _trace is not None:
                    _trace.append({"tool": _name, "args": actual, **result})
                if not result.get("ok"):
                    # 返回失败结构而不是抛异常：模型能看到具体原因并自行重试
                    # （例如脚本名写错时改用 list_motion_scripts 返回的名称）
                    return {"ok": False, "error": result.get("error")}
                return result.get("result")

            tools.append(
                StructuredTool.from_function(
                    func=_run,
                    name=tool.name,
                    description=tool.description,
                    args_schema=args_schema,
                )
            )
        return tools

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
