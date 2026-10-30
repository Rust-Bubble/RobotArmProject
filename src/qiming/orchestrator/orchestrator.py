# -*- coding: utf-8 -*-
"""多 Agent 调度器 (Orchestrator) 核心分发逻辑。

流程：用户文本 -> 意图识别 -> 按意图分发给对应子 Agent -> 子 Agent 产出结构化
任务 -> 按任务类型分发到外部系统工具 (Function Calling) -> 以事件流的形式
逐阶段产出结果，供 WebSocket 端点推送给前端。
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, AsyncIterator

from src.qiming.mllm.recognizer import MllmRecognizer
from src.qiming.utils.async_utils import run_with_status_events
from src.qiming.orchestrator.agents import (
    BaseAgent,
    ChatAgent,
    DecomposeAgent,
    SystemAgent,
)
from src.qiming.tools.registry import ToolRegistry
from src.qiming.tools.builtin import create_default_registry

# 意图识别可返回的类型 -> 对应子 Agent
AGENT_MAP = {
    "闲聊": ChatAgent,
    "动作指令": DecomposeAgent,
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
        # 按意图类型（AGENT_MAP 的 key）索引子 Agent，分发时用意图直接查表；
        # 工具注册表一同注入，供 Agent 通过 Function Calling 自主调用
        self.agents: dict[str, BaseAgent] = {
            intent: agent_cls(recognizer=self.recognizer, tool_registry=self.tool_registry)
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
            step         分解步骤开始    {index, total, action_type, target_object, message}
            robot_status 机械臂状态更新  {state, phase, driver, target, updated_at}
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

        # Agent 内部由模型自主决定调用哪些工具，这里只负责在动作执行期间
        # 把机械臂状态实时转成事件，并在迭代结束后取回调用记录
        try:
            holder: dict[str, Any] = {}
            async for event in run_with_status_events(
                agent.run, holder, user_text, self.history
            ):
                yield event
            result = holder["result"]
        except Exception as exc:  # noqa: BLE001
            yield {"type": "error", "message": f"{agent.name} agent 处理失败: {exc}"}
            yield {"type": "done"}
            return

        # 模型自主调用产生的工具结果，按调用顺序回传前端
        traces: list[dict] = result.get("tool_traces") or []
        tool_result = None
        for trace in traces:
            event: dict[str, Any] = {
                "type": "tool_result",
                "tool": trace.get("tool", ""),
                "ok": bool(trace.get("ok")),
            }
            if trace.get("ok"):
                event["result"] = trace.get("result")
            else:
                event["error"] = trace.get("error")
            yield event
        if traces:
            tool_result = traces if len(traces) > 1 else traces[0]

        self.history.append({"role": "user", "text": user_text})
        self.history.append({"role": "agent", "text": result["message"]})

        yield {
            "type": "agent_result",
            "agent": agent.name,
            "message": result["message"],
            "task": result.get("task"),
            "tool_result": tool_result,
        }
        yield {"type": "done"}

    def _recognize_intent(self, user_text: str) -> str:
        """调用 MLLM 做意图识别，返回意图字符串。"""
        intent = self.recognizer.tendRecognizer(user_text)
        # 模型偶尔会带引号/空白，做一次清洗
        return str(intent).strip().strip("\"'“”")

    _ACTION_TOOL_KEYWORDS = ("抓", "拿", "递", "取", "放", "置", "移")

    @staticmethod
    def _action_to_tool_call(action: dict[str, Any]) -> tuple[str, dict] | None:
        """把单个原子动作步骤映射为工具调用 (name, args)，无对应工具时返回 None。"""
        action_type = str(action.get("action_type", ""))
        if any(k in action_type for k in Orchestrator._ACTION_TOOL_KEYWORDS):
            params = action.get("params") or {}
            args: dict[str, Any] = {"object_name": str(action.get("target_object", "物体"))}
            if "x" in params:
                args["x"] = params["x"]
            if "y" in params:
                args["y"] = params["y"]
            return "execute_grasp", args
        return None

    @staticmethod
    def _system_to_tool_call(task: dict[str, Any]) -> tuple[str, dict] | None:
        """把系统指令任务映射为工具调用 (name, args)，无对应工具时返回 None。"""
        command = str(task.get("command", ""))
        if "状态" in command:
            return "query_robot_status", {}
        if "停" in command:
            return "emergency_stop", {}
        if "复位" in command or "重启" in command:
            return "reset_system", {}
        return None

    @classmethod
    def _tool_calls_for_task(cls, task: dict[str, Any] | None) -> list[tuple[str, dict]]:
        """把子 Agent 产出的结构化任务映射为一组工具调用 [(name, args), ...]。

        plan 任务按步骤展开；单步 action / system 任务退化为单元素列表；
        没有对应工具（如闲聊）时返回空列表。
        """
        if not task:
            return []
        if task.get("type") == "plan":
            return [call for step in (task.get("steps") or [])
                    if (call := cls._action_to_tool_call(step))]
        if task.get("type") == "action":
            call = cls._action_to_tool_call(task)
            return [call] if call else []
        if task.get("type") == "system":
            call = cls._system_to_tool_call(task)
            return [call] if call else []
        return []


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
            steps = self._mock_decompose(user_text)
            message = f"好的，准备依次执行 {len(steps)} 个动作（mock）"
            task = {"type": "plan", "steps": steps}
        elif intent == "系统指令":
            message = f"系统指令已收到：{user_text}（mock，尚未接入硬件）"
            task = {"type": "system", "command": user_text}
        else:
            message = f"收到你的消息：{user_text}（mock 回复）"
            task = None

        # mock 调度器同样走工具分发，保证全链路可联调
        tool_calls = self._tool_calls_for_task(task)
        is_plan = bool(task and task.get("type") == "plan")
        steps = task.get("steps", []) if is_plan else []

        tool_results: list[dict] = []
        for idx, (tool_name, tool_args) in enumerate(tool_calls, start=1):
            if is_plan:
                step = steps[idx - 1] if idx - 1 < len(steps) else {}
                yield {
                    "type": "step",
                    "index": idx,
                    "total": len(tool_calls),
                    "action_type": step.get("action_type", ""),
                    "target_object": step.get("target_object", ""),
                    "message": step.get("reply", ""),
                }
            holder: dict[str, Any] = {}
            async for event in run_with_status_events(
                self.tool_registry.execute, holder, tool_name, tool_args
            ):
                yield event
            tool_results.append(holder["result"])
            yield {"type": "tool_result", "tool": tool_name, **holder["result"]}

        tool_result = tool_results if is_plan else (tool_results[0] if tool_results else None)

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

    @staticmethod
    def _mock_decompose(user_text: str) -> list[dict]:
        """mock 版动作拆解：按标点/连接词切分子句，每子句一个原子步骤。"""
        clauses = [c.strip() for c in re.split(r"[，,。；;、]|然后|再", user_text) if c.strip()]
        steps = [
            {
                "action_type": "抓取" if any(k in c for k in ("拿", "抓", "取", "递", "给")) else "放置",
                "target_object": c,
                "params": {},
                "reply": f"执行：{c}",
            }
            for c in clauses
        ]
        if not steps:
            steps = [{"action_type": "抓取", "target_object": user_text, "params": {}, "reply": f"执行：{user_text}"}]
        return steps

    def _recognize_intent(self, user_text: str) -> str:  # pragma: no cover
        raise NotImplementedError("MockOrchestrator 不使用 MLLM 意图识别")
