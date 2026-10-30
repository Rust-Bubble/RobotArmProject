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
    """子 Agent 基类：持有共享的 MllmRecognizer 与工具注册表，定义统一接口。"""

    name = "base"
    model = "qwen-flash"
    system_prompt = ""
    # 该 Agent 允许模型自主调用的工具（None 表示不允许调用任何工具）
    tool_names: tuple[str, ...] | None = None

    def __init__(self, recognizer: MllmRecognizer, tool_registry=None) -> None:
        self.recognizer = recognizer
        self.tool_registry = tool_registry

    def _build_tools(self, trace: list[dict]) -> list:
        """按 tool_names 收窄可用工具，并把调用记录写入 trace。"""
        if not self.tool_names or self.tool_registry is None:
            return []
        return self.tool_registry.to_langchain_tools(trace=trace, names=self.tool_names)

    def run(self, user_text: str, history: list[dict]) -> dict:
        """处理一轮用户输入，返回 {"message": ..., "task": ..., "tool_traces": [...]}。

        均为同步阻塞调用，由调度器放到线程里执行。
        :param user_text: 用户本轮输入
        :param history: 历史对话 [{"role": "user"|"agent", "text": ...}]
        """
        trace: list[dict] = []
        result = self.recognizer.agentCommunication(
            self.model,
            self.system_prompt,
            self._build_user_info(user_text, history),
            tools=self._build_tools(trace),
        )
        return {"message": str(result), "task": None, "tool_traces": trace}

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


def task_from_traces(traces: list[dict]) -> dict | None:
    """把模型的工具调用记录转成前端控制台展示的任务结构。"""
    if not traces:
        return None
    first = traces[0]
    args = first.get("args") or {}
    return {
        "type": "action",
        "action_type": first.get("tool", ""),
        "target_object": str(args.get("object_name") or "—"),
        "params": args,
    }


class ChatAgent(BaseAgent):
    """闲聊 Agent：普通对话回复，只能调用只读的脚本列表工具。

    没有动作权限，但可以列出系统能力与预设脚本，
    避免被问到"有哪些脚本"时凭空回答"暂无"。
    """

    name = "chat"
    model = "qwen-flash"
    tool_names = ("list_motion_scripts",)
    system_prompt = """
    #身份
    你是智能机械臂"启明智手"的对话助手，名叫启明。
    #系统能力（仅供介绍，你本人不执行机械臂动作）
    本系统提供以下能力：
    - 预设运动脚本（不含视觉，只有运动与夹爪开合），列表由 list_motion_scripts 工具返回
    - execute_grasp：抓取物体并递送到用户手边
    - execute_coords：按坐标移动，或在指定坐标抓取递送
    - query_robot_status：查询机械臂工作状态
    - emergency_stop：紧急停止
    - reset_system：软件复位
    - speak：语音播报
    注意：这些是进程内的 Python 工具，不是 scripts/ 目录下的脚本；
    系统没有执行任意脚本或 Shell 命令的能力。
    #指令
    1. 用简短的中文自然回应用户，保持友好。
    2. 你本人不执行机械臂动作，动作类请求由动作 Agent 处理。
    3. 用户问"有哪些脚本/能做什么"时，先调用 list_motion_scripts 获取真实列表，
       再按返回结果如实说明脚本名称与用途；不要回答"暂无"，也不要编造清单之外的能力。
    """


class DecomposeAgent(BaseAgent):
    """动作分解 Agent：由模型自主决定调用哪个机械臂工具，并支持多步任务。

    工具（execute_grasp 抓取递送 / execute_coords 坐标移动或抓取）的
    JSON Schema 由 Function Calling 自动提供给模型，模型自行决定调用顺序，
    不再依赖 Python 关键词映射。
    """

    name = "decompose"
    model = "qwen3-vl-plus"  # 多步工具编排对模型能力要求较高，使用更强模型
    tool_names = ("execute_grasp", "execute_coords", "list_motion_scripts", "run_motion_script")
    system_prompt = """
    #身份
    你是智能机械臂系统的动作规划智能体，可以直接调用机械臂工具完成任务。
    #可用工具
    - list_motion_scripts: 列出所有预设运动脚本（不含视觉，只有运动和夹爪开合）。
    - run_motion_script: 按名称执行预设运动脚本。参数 name 必填；
      不确定名称时先调用 list_motion_scripts 获取。
    - execute_grasp: 抓取物体并递送到用户手边。参数 object_name 必填，x/y 可选（单位 mm）。
    - execute_coords: 按坐标指令执行动作。action=move 移动到坐标；
      action=grasp 在指定坐标抓取并递送；pick_pose 为 [x, y, z, rx, ry, rz]。
    #指令
    1. 用户点名某个预设脚本（如"挥手""回位""抓取放置演示""方形轨迹""夹爪开合"）时，
       调用 run_motion_script 执行，不要自己编造路径。
    2. 用户说"随便执行一个/演示一下"时，先列出脚本，再选一个执行。
    3. 用户要求操作多个物品时，必须逐个调用工具，处理完一个再处理下一个，
       全部完成后才允许输出最终回复；严禁中途回复"已完成"。
    4. 只有无法映射到任何工具时，才直接回复文本说明无法执行。
    5. 调用工具后，用一句简短的中文告诉用户执行结果。
    """

    def run(self, user_text: str, history: list[dict]) -> dict:
        trace: list[dict] = []
        message = self.recognizer.agentCommunication(
            self.model,
            self.system_prompt,
            self._build_user_info(user_text, history),
            tools=self._build_tools(trace),
        )
        return {
            "message": str(message),
            "task": task_from_traces(trace),
            "tool_traces": trace,
        }


class SystemAgent(BaseAgent):
    """系统指令 Agent：由模型自主调用状态查询、急停、复位等系统工具。"""

    name = "system"
    model = "qwen-flash"
    tool_names = ("query_robot_status", "emergency_stop", "reset_system")
    system_prompt = """
    #身份
    你是智能机械臂系统的系统指令处理智能体，可以直接调用系统工具。
    #可用工具
    - query_robot_status: 查询机械臂当前的工作状态。
    - emergency_stop: 紧急停止机械臂的一切动作（仅在用户明确要求停止时调用）。
    - reset_system: 软件层面的系统复位。
    #指令
    1. 用户要求查询状态、停止、复位时，调用对应工具，不要凭空回答状态。
    2. 调用工具后，用一句简短的中文向用户说明结果。
    3. 目前系统处于联调阶段，除上述工具外没有其它硬件能力。
    """

    def run(self, user_text: str, history: list[dict]) -> dict:
        trace: list[dict] = []
        message = self.recognizer.agentCommunication(
            self.model,
            self.system_prompt,
            self._build_user_info(user_text, history),
            tools=self._build_tools(trace),
        )
        return {
            "message": str(message),
            "task": task_from_traces(trace),
            "tool_traces": trace,
        }
