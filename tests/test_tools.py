# -*- coding: utf-8 -*-
"""工具注册表 (Function Calling 基础 API) 的单元测试。"""

import pytest

from src.qiming.tools.builtin import create_default_registry
from src.qiming.tools.registry import Tool, ToolRegistry


@pytest.fixture()
def registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool(
        name="add",
        description="两数相加",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
    )
    def add(a: float, b: float) -> dict:
        return {"sum": a + b}

    @reg.tool(name="boom", description="总是抛异常", parameters={"type": "object", "properties": {}})
    def boom() -> dict:
        raise RuntimeError("炸了")

    return reg


class TestToolRegistry:
    def test_register_and_get(self, registry: ToolRegistry):
        tool = registry.get("add")
        assert tool is not None
        assert tool.description == "两数相加"
        assert set(registry.names()) == {"add", "boom"}
        assert "add" in registry
        assert "nope" not in registry

    def test_execute_success(self, registry: ToolRegistry):
        result = registry.execute("add", {"a": 1, "b": 2})
        assert result == {"ok": True, "result": {"sum": 3}}

    def test_execute_accepts_json_string(self, registry: ToolRegistry):
        # 大模型 Function Calling 返回的是 JSON 字符串参数
        result = registry.execute("add", '{"a": 1.5, "b": 2.5}')
        assert result["ok"] is True
        assert result["result"]["sum"] == 4

    def test_execute_unknown_tool(self, registry: ToolRegistry):
        result = registry.execute("not_exist", {})
        assert result["ok"] is False
        assert "未注册" in result["error"]

    def test_execute_tool_exception_is_captured(self, registry: ToolRegistry):
        result = registry.execute("boom", {})
        assert result["ok"] is False
        assert "炸了" in result["error"]

    def test_execute_bad_json_args(self, registry: ToolRegistry):
        result = registry.execute("add", "{not json")
        assert result["ok"] is False
        assert "JSON" in result["error"]

    def test_execute_wrong_params(self, registry: ToolRegistry):
        result = registry.execute("add", {"a": 1})  # 缺少 b
        assert result["ok"] is False
        assert "参数" in result["error"]

    def test_duplicate_register_raises(self, registry: ToolRegistry):
        with pytest.raises(ValueError):
            registry.register(
                Tool(name="add", description="重复", parameters={}, handler=lambda: None)
            )

    def test_openai_schema_shape(self, registry: ToolRegistry):
        schemas = registry.to_openai_schemas()
        add_schema = next(s for s in schemas if s["function"]["name"] == "add")
        assert add_schema["type"] == "function"
        assert add_schema["function"]["parameters"]["required"] == ["a", "b"]

    def test_to_langchain_tools(self, registry: ToolRegistry):
        lc_tools = registry.to_langchain_tools()
        add_tool = next(t for t in lc_tools if t.name == "add")
        assert add_tool.invoke({"a": 2, "b": 3}) == {"sum": 5}


class TestBuiltinRegistry:
    def test_builtin_tools_registered(self):
        reg = create_default_registry()
        assert {
            "query_robot_status",
            "emergency_stop",
            "execute_grasp",
            "speak",
            "reset_system",
        }.issubset(set(reg.names()))

    def test_query_robot_status(self):
        reg = create_default_registry()
        result = reg.execute("query_robot_status", {})
        assert result["ok"] is True
        assert result["result"]["status"] == "idle"

    def test_emergency_stop(self):
        reg = create_default_registry()
        result = reg.execute("emergency_stop", {})
        assert result["ok"] is True
        assert result["result"]["stopped"] is True

    def test_execute_grasp_mock(self):
        reg = create_default_registry()
        result = reg.execute("execute_grasp", {"object_name": "水杯"})
        assert result["ok"] is True
        assert result["result"]["executed"] is True
        assert result["result"]["object"] == "水杯"

    def test_speak(self, capsys):
        reg = create_default_registry()
        result = reg.execute("speak", {"text": "你好"})
        assert result["ok"] is True
        assert "你好" in capsys.readouterr().out
