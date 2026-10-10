# -*- coding: utf-8 -*-
"""多 Agent 调度器 (Orchestrator) 分发逻辑的单元测试。

真实调度器用假 Recognizer 驱动，不访问任何外部大模型服务。
"""

import asyncio

import pytest

from src.qiming.orchestrator.orchestrator import MockOrchestrator, Orchestrator


def collect_events(orchestrator, text: str) -> list[dict]:
    async def run():
        return [event async for event in orchestrator.handle(text)]

    return asyncio.run(run())


class FakeRecognizer:
    """替身 Recognizer：意图识别与 Agent 调用都返回预设值。"""

    def __init__(self, intent: str = "闲聊") -> None:
        self.intent = intent
        self.last_response_form = None

    def tendRecognizer(self, user_text: str) -> str:
        return self.intent

    def agentCommunication(self, model, system_prompt, userInfo, imageInfo=None, responseForm=None):
        self.last_response_form = responseForm
        if responseForm is not None:
            if getattr(responseForm, "__name__", "") == "ActionPlan":
                return responseForm(
                    action_type="抓取",
                    target_object="水杯",
                    params={},
                    reply="好的，正在为您拿取水杯",
                )
            return responseForm(command="查询状态", reply="正在查询机械臂状态")
        return "这是一条闲聊回复"


class TestMockOrchestrator:
    def test_action_intent_routes_to_action_agent_and_tool(self):
        orch = MockOrchestrator()
        events = collect_events(orch, "帮我拿一下水杯")

        types = [e["type"] for e in events]
        assert types == [
            "status", "intent", "agent_start",
            "tool_result", "agent_result", "done",
        ]
        intent_event = events[1]
        assert intent_event["intent"] == "动作指令"
        assert intent_event["agent"] == "action"

        # 动作任务被分发到 execute_grasp 工具
        tool_event = events[3]
        assert tool_event["tool"] == "execute_grasp"
        assert tool_event["ok"] is True

        result = events[4]
        assert result["task"]["type"] == "action"
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


class TestOrchestrator:
    def test_real_orchestrator_dispatch_by_intent(self):
        """意图识别 -> 子 Agent -> 工具分发 全链路（假 Recognizer）。"""
        recognizer = FakeRecognizer(intent="系统指令")
        orch = Orchestrator(recognizer=recognizer)
        events = collect_events(orch, "现在什么状态")

        assert events[1]["intent"] == "系统指令"
        assert events[1]["agent"] == "system"
        tool_event = next(e for e in events if e["type"] == "tool_result")
        assert tool_event["tool"] == "query_robot_status"
        assert tool_event["ok"] is True
        agent_result = next(e for e in events if e["type"] == "agent_result")
        assert agent_result["task"]["type"] == "system"

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

    def test_tool_call_mapping(self):
        # 动作任务映射
        assert Orchestrator._tool_call_for_task(
            {"type": "action", "action_type": "抓取", "target_object": "水杯", "params": {}}
        ) == ("execute_grasp", {"object_name": "水杯"})
        # 放置类动作暂无对应工具
        assert Orchestrator._tool_call_for_task(
            {"type": "action", "action_type": "放置", "target_object": "水杯", "params": {}}
        ) is None
        # 系统任务映射
        assert Orchestrator._tool_call_for_task({"type": "system", "command": "查询状态"}) == (
            "query_robot_status",
            {},
        )
        # 闲聊无任务
        assert Orchestrator._tool_call_for_task(None) is None
        assert Orchestrator._tool_call_for_task({"type": "chat"}) is None
