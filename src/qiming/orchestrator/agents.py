# -*- coding: utf-8 -*-
"""Orchestrator 下属的各个子 Agent。

每个 Agent 封装一种意图的处理方式，统一对外输出：
    {"message": 给用户看的回复文本, "task": 结构化任务(可选)}
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from src.qiming.mllm.recognizer import MllmRecognizer

# 与前端控制台约定的任务结构
Task = dict[str, Any] | None


class _AgentResult(BaseModel):
    """所有 Agent 的统一输出形状。"""

    message: str
    task: Task = None


class ActionPlan(BaseModel):
    """单个原子动作步骤的结构化输出（同时作为 DecomposePlan 的步骤类型）。"""

    action_type: str  # 动作类型，如 抓取、放置、递给用户
    target_object: str  # 目标物体
    params: dict[str, Any] = {}  # 其他动作参数
    reply: str  # 给用户的确认回复


class DecomposePlan(BaseModel):
    """复杂动作分解 Agent 的结构化输出：有序原子步骤序列。"""

    steps: list[ActionPlan]  # 按执行先后顺序排列的原子步骤
    reply: str  # 给用户的整体确认回复


class SystemPlan(BaseModel):
    """系统指令 Agent 的结构化输出。"""

    command: str  # 系统命令，如 查询状态、紧急停止
    reply: str  # 给用户的回复


class BaseAgent:
    """子 Agent 基类：持有共享的 MllmRecognizer，定义统一接口。"""

    name = "base"
    model = "qwen-flash"
    system_prompt = ""

    def __init__(self, recognizer: MllmRecognizer) -> None:
        self.recognizer = recognizer

    def run(self, user_text: str, history: list[dict]) -> dict:
        """处理一轮用户输入，返回 {"message": ..., "task": ...}。

        均为同步阻塞调用，由调度器放到线程里执行。
        :param user_text: 用户本轮输入
        :param history: 历史对话 [{"role": "user"|"agent", "text": ...}]
        """
        result = self.recognizer.agentCommunication(
            self.model, self.system_prompt, self._build_user_info(user_text, history)
        )
        return {"message": str(result), "task": None}

    def _build_user_info(self, user_text: str, history: list[dict]) -> str:
        """把最近几轮历史拼进用户消息，保证多轮上下文。"""
        if not history:
            return user_text
        lines = ["以下是之前的对话记录（供参考，可能为空缺省）："]
        for turn in history[-6:]:
            who = "用户" if turn["role"] == "user" else "助手"
            lines.append(f"{who}: {turn['text']}")
        lines.append("---")
        lines.append(f"用户本轮消息: {user_text}")
        return "\n".join(lines)


class ChatAgent(BaseAgent):
    """闲聊 Agent：普通对话回复，不产生结构化任务。"""

    name = "chat"
    model = "qwen-flash"
    system_prompt = """
    #身份
    你是智能机械臂"启明智手"的对话助手，名叫启明。
    #指令
    请用简短的中文自然回应用户，保持友好。你不需要执行任何机械臂动作。
    """


class DecomposeAgent(BaseAgent):
    """动作分解 Agent：把用户指令拆成有序的原子抓取/放置步骤。"""

    name = "decompose"
    model = "qwen-flash"
    system_prompt = """
    #身份
    你是智能机械臂系统的动作分解规划智能体。
    #指令
    用户会给出想让机械臂执行的动作指令，可能包含多个连续动作（例如"先把苹果拿过来，再把杯子放到桌上"）。
    请把指令拆解为按执行先后顺序排列的原子步骤，每一步都是单次"抓取/放置"级别的动作：
    - steps: 有序步骤列表，每步包含：
        - action_type: 动作类型，如 抓取、放置、移动、递给用户
        - target_object: 该步要操作的单个物体名称
        - params: 其他动作参数（如位置、方向等），没有则留空字典
        - reply: 用一句简短中文描述这一步的动作
    - reply: 用一句简短中文向用户确认即将依次执行的整个任务
    注意：只包含单个动作的指令也要返回仅含一个 step 的 steps 列表。
    """

    def run(self, user_text: str, history: list[dict]) -> dict:
        plan: DecomposePlan = self.recognizer.agentCommunication(
            self.model,
            self.system_prompt,
            self._build_user_info(user_text, history),
            responseForm=DecomposePlan,
        )
        return {
            "message": plan.reply,
            "task": {
                "type": "plan",
                "steps": [
                    {
                        "action_type": s.action_type,
                        "target_object": s.target_object,
                        "params": s.params,
                        "reply": s.reply,
                    }
                    for s in plan.steps
                ],
            },
        }


class SystemAgent(BaseAgent):
    """系统指令 Agent：处理查询状态、紧急停止等系统级命令。"""

    name = "system"
    model = "qwen-flash"
    system_prompt = """
    #身份
    你是智能机械臂系统的系统指令处理智能体。
    #指令
    用户会给出系统级指令（如查询状态、紧急停止、回到初始位等），请解析出：
    - command: 系统命令名称
    - reply: 用一句简短的中文向用户说明系统将如何处理
    目前系统仅处于联调阶段，除"紧急停止"外命令暂未接入真实硬件，请在 reply 中如实说明。
    """

    def run(self, user_text: str, history: list[dict]) -> dict:
        plan: SystemPlan = self.recognizer.agentCommunication(
            self.model,
            self.system_prompt,
            self._build_user_info(user_text, history),
            responseForm=SystemPlan,
        )
        return {
            "message": plan.reply,
            "task": {"type": "system", "command": plan.command},
        }
