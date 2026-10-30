# -*- coding: utf-8 -*-
"""多 Agent 调度器 (Orchestrator) 分发逻辑的单元测试。

真实调度器用假 Recognizer 驱动，不访问任何外部大模型服务。
"""

import asyncio

import pytest

from src.qiming.orchestrator.agents import ActionPlan
from src.qiming.orchestrator.orchestrator import MockOrchestrator, Orchestrator


def collect_events(orchestrator, text: str) -> list[dict]:
    async def run():
        return [event async for event in orchestrator.handle(text)]

    return asyncio.run(run())


class FakeRecognizer:
    """替身 Recognizer：意图识别返回预设值，并模拟模型自主调用工具。

    Function Calling 架构下工具由 create_agent 自主调用；测试用 tool_calls
    预设「模型决定调用哪些工具」，再通过 StructuredTool.invoke 真实执行，
    因此仍覆盖参数 schema 校验与 trace 记录。
    """

    def __init__(self, intent: str = "闲聊", tool_calls: list | None = None) -> None:
        self.intent = intent
        self.tool_calls = tool_calls or []
        self.last_tools: list[str] = []

    def tendRecognizer(self, user_text: str) -> str:
        return self.intent

    def agentCommunication(self, model, system_prompt, userInfo, imageInfo=None,
                           responseForm=None, tools=None):
        self.last_tools = [t.name for t in (tools or [])]
        by_name = {t.name: t for t in (tools or [])}
        for name, args in self.tool_calls:
            if name in by_name:
                by_name[name].invoke(args)
        return "好的，已执行"


class TestMockOrchestrator:
    def test_action_intent_routes_to_decompose_agent_and_tool(self):
        orch = MockOrchestrator()
        events = collect_events(orch, "帮我拿一下水杯")

        # 动作执行期间会穿插 robot_status 事件（供前端机械臂状态卡片实时刷新），
        # 这里校验其余事件仍保持原有顺序与内容。
        types = [e["type"] for e in events if e["type"] != "robot_status"]
        assert types == [
            "status", "intent", "agent_start",
            "step", "tool_result", "agent_result", "done",
        ]
        intent_event = events[1]
        assert intent_event["intent"] == "动作指令"
        assert intent_event["agent"] == "decompose"

        # 机械臂状态被逐步上报：动作执行中有 busy，动作结束后回到 idle
        status_events = [e for e in events if e["type"] == "robot_status"]
        assert status_events, "应上报机械臂状态"
        assert any(e["state"] == "busy" for e in status_events)
        assert status_events[-1]["state"] == "idle"
        assert any(e.get("phase") == "闭合夹爪" for e in status_events)

        # 动作任务被分发到 execute_grasp 工具
        tool_event = next(e for e in events if e["type"] == "tool_result")
        assert tool_event["tool"] == "execute_grasp"
        assert tool_event["ok"] is True

        result = next(e for e in events if e["type"] == "agent_result")
        assert result["task"]["type"] == "plan"
        assert len(result["task"]["steps"]) == 1
        # 历史被记录
        assert orch.history[-2]["role"] == "user"

    def test_system_intent_query_status(self):
        orch = MockOrchestrator()
        events = collect_events(orch, "查询一下状态")

        assert events[1]["intent"] == "系统指令"
        assert events[1]["agent"] == "system"
        tool_event = next(e for e in events if e["type"] == "tool_result")
        assert tool_event["tool"] == "query_robot_status"
        assert tool_event["ok"] is True

    def test_system_intent_emergency_stop(self):
        orch = MockOrchestrator()
        events = collect_events(orch, "紧急停止")
        tool_event = next(e for e in events if e["type"] == "tool_result")
        assert tool_event["tool"] == "emergency_stop"
        assert tool_event["result"]["stopped"] is True

    def test_chat_intent_has_no_tool_call(self):
        orch = MockOrchestrator()
        events = collect_events(orch, "今天天气真不错")
        assert events[1]["intent"] == "闲聊"
        assert events[1]["agent"] == "chat"
        assert all(e["type"] != "tool_result" for e in events)
        assert events[-1]["type"] == "done"

    def test_history_accumulates(self):
        orch = MockOrchestrator()
        collect_events(orch, "你好")
        collect_events(orch, "再次你好")
        assert len(orch.history) == 4

    def test_compound_action_decomposes_into_multiple_steps(self):
        orch = MockOrchestrator()
        events = collect_events(orch, "先把苹果拿过来，再把杯子放到桌上")

        steps = [e for e in events if e["type"] == "step"]
        assert len(steps) == 2
        assert steps[0]["index"] == 1 and steps[0]["total"] == 2
        assert steps[1]["index"] == 2 and steps[1]["total"] == 2

        tool_events = [e for e in events if e["type"] == "tool_result"]
        assert len(tool_events) == 2
        assert all(e["tool"] == "execute_grasp" for e in tool_events)
        assert all(e["ok"] is True for e in tool_events)

        result = next(e for e in events if e["type"] == "agent_result")
        assert result["task"]["type"] == "plan"
        assert len(result["task"]["steps"]) == 2
        assert isinstance(result["tool_result"], list)
        assert len(result["tool_result"]) == 2


class TestOrchestrator:
    def test_real_orchestrator_dispatch_by_intent(self):
        """意图识别 -> 子 Agent -> 模型自主调用工具 全链路（假 Recognizer）。"""
        recognizer = FakeRecognizer(
            intent="系统指令", tool_calls=[("query_robot_status", {})])
        orch = Orchestrator(recognizer=recognizer)
        events = collect_events(orch, "现在什么状态")

        assert events[1]["intent"] == "系统指令"
        assert events[1]["agent"] == "system"
        # SystemAgent 只向模型暴露系统类工具
        assert set(recognizer.last_tools) == {
            "query_robot_status", "emergency_stop", "reset_system"}

        tool_event = next(e for e in events if e["type"] == "tool_result")
        assert tool_event["tool"] == "query_robot_status"
        assert tool_event["ok"] is True

        agent_result = next(e for e in events if e["type"] == "agent_result")
        assert agent_result["task"]["action_type"] == "query_robot_status"

    def test_real_orchestrator_unknown_intent_falls_back_to_chat(self):
        recognizer = FakeRecognizer(intent="其他")
        orch = Orchestrator(recognizer=recognizer)
        events = collect_events(orch, "随便说点什么")
        # "其他" 不在 AGENT_MAP 中，回落到 chat agent
        assert events[1]["agent"] == "chat"
        assert all(e["type"] != "tool_result" for e in events)

    def test_real_orchestrator_recognizer_error_yields_error_event(self):
        class BrokenRecognizer:
            def tendRecognizer(self, user_text):
                raise RuntimeError("模型超时")

            def agentCommunication(self, *args, **kwargs):  # pragma: no cover
                raise AssertionError("不应被调用")

        orch = Orchestrator(recognizer=BrokenRecognizer())
        events = collect_events(orch, "你好")
        assert events[0]["type"] == "status"
        error_event = next(e for e in events if e["type"] == "error")
        assert "意图识别失败" in error_event["message"]
        assert events[-1]["type"] == "done"

    def test_tool_calls_mapping(self):
        # plan 任务按步骤映射为多个 execute_grasp
        assert Orchestrator._tool_calls_for_task(
            {
                "type": "plan",
                "steps": [
                    {"action_type": "抓取", "target_object": "苹果", "params": {"x": 1, "y": 2}},
                    {"action_type": "放置", "target_object": "杯子", "params": {}},
                ],
            }
        ) == [
            ("execute_grasp", {"object_name": "苹果", "x": 1, "y": 2}),
            ("execute_grasp", {"object_name": "杯子"}),
        ]
        # 单步 action 任务退化为单元素列表
        assert Orchestrator._tool_calls_for_task(
            {"type": "action", "action_type": "抓取", "target_object": "水杯", "params": {}}
        ) == [("execute_grasp", {"object_name": "水杯"})]
        # 放置类动作也映射到 execute_grasp（当前以抓取-递送近似）
        assert Orchestrator._tool_calls_for_task(
            {"type": "action", "action_type": "放置", "target_object": "水杯", "params": {}}
        ) == [("execute_grasp", {"object_name": "水杯"})]
        # 系统任务映射
        assert Orchestrator._tool_calls_for_task({"type": "system", "command": "查询状态"}) == [
            ("query_robot_status", {}),
        ]
        # 闲聊无任务
        assert Orchestrator._tool_calls_for_task(None) == []
        assert Orchestrator._tool_calls_for_task({"type": "chat"}) == []

    def test_real_orchestrator_decomposes_complex_action(self):
        recognizer = FakeRecognizer(
            intent="动作指令",
            tool_calls=[
                ("execute_grasp", {"object_name": "苹果"}),
                ("execute_grasp", {"object_name": "杯子"}),
            ],
        )
        orch = Orchestrator(recognizer=recognizer)
        events = collect_events(orch, "先把苹果拿过来，再把杯子放到桌上")

        assert events[1]["intent"] == "动作指令"
        assert events[1]["agent"] == "decompose"
        # DecomposeAgent 暴露机械臂动作类工具与预设脚本工具
        assert set(recognizer.last_tools) == {
            "execute_grasp", "execute_coords", "list_motion_scripts", "run_motion_script"}

        tool_events = [e for e in events if e["type"] == "tool_result"]
        assert len(tool_events) == 2
        assert all(e["tool"] == "execute_grasp" for e in tool_events)
        assert all(e["ok"] is True for e in tool_events)

        result = next(e for e in events if e["type"] == "agent_result")
        assert isinstance(result["tool_result"], list)
        assert len(result["tool_result"]) == 2

    def test_real_orchestrator_chat_agent_only_reads_script_list(self):
        recognizer = FakeRecognizer(
            intent="闲聊", tool_calls=[("list_motion_scripts", {})])
        orch = Orchestrator(recognizer=recognizer)
        events = collect_events(orch, "你有哪些脚本")

        # 闲聊 Agent 只暴露只读的脚本列表工具，不能执行动作
        assert recognizer.last_tools == ["list_motion_scripts"]

        tool_event = next(e for e in events if e["type"] == "tool_result")
        assert tool_event["tool"] == "list_motion_scripts"
        assert tool_event["ok"] is True
