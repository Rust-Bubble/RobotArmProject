# -*- coding: utf-8 -*-
"""多 Agent 调度器 (Orchestrator) 核心分发逻辑。

流程：用户文本 -> 意图识别 -> 按意图分发给对应子 Agent -> 子 Agent 产出结构化
任务 -> 按任务类型分发到外部系统工具 (Function Calling) -> 以事件流的形式
逐阶段产出结果，供 WebSocket 端点推送给前端。
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from src.qiming.mllm.recognizer import MllmRecognizer
from src.qiming.orchestrator.agents import (
    ActionAgent,
    BaseAgent,
    ChatAgent,
    SystemAgent,
)
from src.qiming.tools.registry import ToolRegistry
from src.qiming.tools.builtin import create_default_registry

# 意图识别可返回的类型 -> 对应子 Agent
AGENT_MAP = {
    "闲聊": ChatAgent,
    "动作指令": ActionAgent,
    "系统指令": SystemAgent,
}


class Orchestrator:
    """调度器：一个连接（会话）对应一个实例，内部维护对话历史。"""

    def __init__(
        self,
        recognizer: MllmRecognizer | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.recognizer = recognizer or MllmRecognizer()
        self.tool_registry = tool_registry or create_default_registry()
        # 按意图类型（AGENT_MAP 的 key）索引子 Agent，分发时用意图直接查表
        self.agents: dict[str, BaseAgent] = {
            intent: agent_cls(recognizer=self.recognizer)
            for intent, agent_cls in AGENT_MAP.items()
        }
        # 闲聊兜底也复用 ChatAgent
        self.fallback_agent = self.agents["闲聊"]
        self.history: list[dict] = []

    async def handle(self, user_text: str) -> AsyncIterator[dict]:
        """处理一轮用户输入，逐阶段产出事件。

        事件类型（与前端约定）：
            status       进度提示        {stage, message}
            intent       意图识别结果    {intent, agent}
            agent_start  子 Agent 开始   {agent}
            tool_result  工具执行结果    {tool, ok, result|error}
            agent_result 子 Agent 结束   {agent, message, task}
            error        出错            {message}
            done         本轮结束        {}
        """
        yield {"type": "status", "stage": "intent", "message": "正在识别意图…"}

        try:
            # 意图识别是同步阻塞调用，放到线程里避免卡住事件循环
            intent = await asyncio.to_thread(self._recognize_intent, user_text)
        except Exception as exc:  # noqa: BLE001
            yield {"type": "error", "message": f"意图识别失败: {exc}"}
            yield {"type": "done"}
            return

        agent = self.agents.get(intent, self.fallback_agent)
        yield {"type": "intent", "intent": intent, "agent": agent.name}
        yield {"type": "agent_start", "agent": agent.name}
        yield {
            "type": "status",
            "stage": "agent",
            "message": f"正在由 {agent.name} agent 处理…",
        }

        try:
            result = await asyncio.to_thread(agent.run, user_text, self.history)
        except Exception as exc:  # noqa: BLE001
            yield {"type": "error", "message": f"{agent.name} agent 处理失败: {exc}"}
            yield {"type": "done"}
            return

        # 子 Agent 产出结构化任务时，分发到对应的外部系统工具
        tool_result = None
        tool_call = self._tool_call_for_task(result["task"])
        if tool_call is not None:
            tool_name, tool_args = tool_call
            tool_result = await asyncio.to_thread(
                self.tool_registry.execute, tool_name, tool_args
            )
            yield {"type": "tool_result", "tool": tool_name, **tool_result}

        self.history.append({"role": "user", "text": user_text})
        self.history.append({"role": "agent", "text": result["message"]})

        yield {
            "type": "agent_result",
            "agent": agent.name,
            "message": result["message"],
            "task": result["task"],
            "tool_result": tool_result,
        }
        yield {"type": "done"}

    def _recognize_intent(self, user_text: str) -> str:
        """调用 MLLM 做意图识别，返回意图字符串。"""
        intent = self.recognizer.tendRecognizer(user_text)
        # 模型偶尔会带引号/空白，做一次清洗
        return str(intent).strip().strip("\"'“”")

    @staticmethod
    def _tool_call_for_task(task: dict[str, Any] | None) -> tuple[str, dict] | None:
        """把子 Agent 产出的结构化任务映射为工具调用 (name, args)。

        没有对应工具（如闲聊）时返回 None。
        """
        if not task:
            return None
        if task.get("type") == "action":
            action_type = str(task.get("action_type", ""))
            if any(k in action_type for k in ("抓", "拿", "递", "取")):
                params = task.get("params") or {}
                args: dict[str, Any] = {
                    "object_name": str(task.get("target_object", "物体"))
                }
                if "x" in params:
                    args["x"] = params["x"]
                if "y" in params:
                    args["y"] = params["y"]
                return "execute_grasp", args
            return None
        if task.get("type") == "system":
            command = str(task.get("command", ""))
            if "状态" in command:
                return "query_robot_status", {}
            if "停" in command:
                return "emergency_stop", {}
            if "复位" in command or "重启" in command:
                return "reset_system", {}
            return None
        return None


class MockOrchestrator(Orchestrator):
    """不走 MLLM 的本地调度器，用于联调和测试。

    通过 llm.yaml 的 mode: mock 或构造参数启用。
    """

    INTENT_KEYWORDS = {
        "动作指令": ("拿", "抓", "递", "放", "移动", "给我"),
        "系统指令": ("状态", "停止", "急停", "复位", "重启"),
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    async def handle(self, user_text: str) -> AsyncIterator[dict]:
        yield {"type": "status", "stage": "intent", "message": "正在识别意图…"}
        intent = "闲聊"
        for candidate, keywords in self.INTENT_KEYWORDS.items():
            if any(k in user_text for k in keywords):
                intent = candidate
                break
        agent = self.agents.get(intent, self.fallback_agent)
        yield {"type": "intent", "intent": intent, "agent": agent.name}
        yield {"type": "agent_start", "agent": agent.name}

        if intent == "动作指令":
            message = f"好的，准备执行动作：{user_text}（mock）"
            task = {"type": "action", "action_type": "抓取", "target_object": user_text, "params": {}}
        elif intent == "系统指令":
            message = f"系统指令已收到：{user_text}（mock，尚未接入硬件）"
            task = {"type": "system", "command": user_text}
        else:
            message = f"收到你的消息：{user_text}（mock 回复）"
            task = None

        # mock 调度器同样走工具分发，保证全链路可联调
        tool_result = None
        tool_call = self._tool_call_for_task(task)
        if tool_call is not None:
            tool_name, tool_args = tool_call
            tool_result = await asyncio.to_thread(
                self.tool_registry.execute, tool_name, tool_args
            )
            yield {"type": "tool_result", "tool": tool_name, **tool_result}

        self.history.append({"role": "user", "text": user_text})
        self.history.append({"role": "agent", "text": message})
        yield {
            "type": "agent_result",
            "agent": agent.name,
            "message": message,
            "task": task,
            "tool_result": tool_result,
        }
        yield {"type": "done"}

    def _recognize_intent(self, user_text: str) -> str:  # pragma: no cover
        raise NotImplementedError("MockOrchestrator 不使用 MLLM 意图识别")
