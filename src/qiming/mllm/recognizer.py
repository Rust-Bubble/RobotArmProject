"""MLLM 识别与决策入口。"""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from pydantic import BaseModel

from src.qiming.models import ImageFrame, MllmDecision


class Action(BaseModel):
    """模型识别出的目标和图像像素中心。"""

    obj: str
    pos: tuple[float, float] | None = None


class MllmRecognizer:
    """根据文本和图像识别目标，返回坐标转换层使用的决策类型。"""

    DEFAULT_MODEL = "qwen3-vl-plus"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        load_dotenv(Path(__file__).with_name(".env"))
        config = config or {}
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.base_url = config.get("endpoint") or os.getenv("DASHSCOPE_BASE_URL")
        self.model_name = config.get("model_name") or self.DEFAULT_MODEL
        self.request_timeout_s = float(config.get("request_timeout_s", 8))

    @staticmethod
    def _image_payload(image_info: ImageFrame | str | Path | Any) -> tuple[str, str]:
        """Return base64 image data and its MIME type from a frame, array, or path."""
        if isinstance(image_info, ImageFrame):
            image_info = image_info.color_frame if image_info.color_frame is not None else image_info.data
            if image_info is None:
                raise ValueError("图像帧为空，无法调用 MLLM")

        if isinstance(image_info, (str, Path)):
            path = Path(image_info)
            mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            return base64.b64encode(path.read_bytes()).decode("ascii"), mime_type

        if isinstance(image_info, bytes):
            return base64.b64encode(image_info).decode("ascii"), "image/jpeg"

        try:
            import cv2

            success, encoded = cv2.imencode(".jpg", image_info)
        except Exception as exc:
            raise ValueError(f"无法编码输入图像: {exc}") from exc
        if not success:
            raise ValueError("无法编码输入图像")
        return base64.b64encode(encoded.tobytes()).decode("ascii"), "image/jpeg"

    def agentCommunication(
        self,
        model_type: str,
        system_prompt: str,
        userInfo: str,
        imageInfo: ImageFrame | str | Path | Any = None,
        responseForm: type[BaseModel] | None = None,
    ) -> Any:
        """Invoke the configured chat model with text and an optional image."""
        qwen_model = init_chat_model(
            model=model_type or self.model_name,
            model_provider="openai",
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.request_timeout_s,
        )
        agent = create_agent(
            model=qwen_model,
            system_prompt=system_prompt,
            response_format=responseForm,
        )

        content: list[dict[str, Any]] = [{"type": "text", "text": userInfo}]
        if imageInfo is not None:
            image, mime_type = self._image_payload(imageInfo)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image}"},
            })

        response = agent.invoke({"messages": [{"role": "user", "content": content}]})
        if responseForm:
            return response["structured_response"]

        messages = response.get("messages", [])
        for message in reversed(messages):
            if getattr(message, "type", None) == "ai":
                return message.content
        raise RuntimeError("MLLM 响应中没有助手消息")

    def recognize(
        self,
        userInfo: str,
        imageInfo: ImageFrame | str | Path | Any,
    ) -> MllmDecision | None:
        """Recognize a target and return None when it cannot be located."""
        system_prompt = """
            # 身份
            你是一个负责识别物体位置的智能体。
            # 指令
            根据用户指令和图像识别用户指定的目标，返回目标名称和目标几何中心的像素坐标。
            坐标格式为 [x, y]，以图像左上角为原点。如果未找到目标，obj 返回空字符串，pos 返回 null。
        """

        action = self.agentCommunication(
            self.model_name, system_prompt, userInfo, imageInfo, Action
        )
        if not isinstance(action, Action):
            action = Action.model_validate(action)
        target_name = action.obj.strip()
        position = action.pos
        if not target_name or position is None or position[0] < 0 or position[1] < 0:
            return None

        return MllmDecision(
            target_name=target_name,
            image_position=(float(position[0]), float(position[1])),
            intent="grasp",
            reply=f"正在为您拿取{target_name}",
        )

    def tendRecognizer(self, userInfo: str) -> str:
        """Classify text as chat, system, or action intent."""
        system_prompt = """
        请判断用户消息属于以下哪种意图：闲聊、系统指令、动作指令或其他。
        只返回一个意图名称，不要添加解释。
        """
        return str(self.agentCommunication(self.model_name, system_prompt, userInfo)).strip()
