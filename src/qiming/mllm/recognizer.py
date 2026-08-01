"""MLLM 识别与决策智能体入口，负责根据用户文本和图像判断目标物体与动作意图。"""

from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from  langchain.agents import create_agent
from pydantic import BaseModel


class Action(BaseModel):
    obj: str
    pos: tuple [float,float]


class MllmRecognizer:
    """MLLM 调用占位，当前仅保留接口形状。"""

    def __init__(self) -> None:
         pass

    def recognize(self,userInfo,imageInfo) :

        load_dotenv()

        system_prompt = """

            #身份
            你是一个负责识别物体位置的智能体。
            
            #指令
            现在需要你根据给出的物体名称，识别该物体在给定图片中的位置，请根据返回其几何中心点的平面坐标，坐标的单位以像素计算。如果图片中包含多个同种目标物体，请依次列出它们的坐标；若不包含目标物体，则返回[-1]。
            请以[(x坐标,y坐标)]的格式输出。
            
           
        """



        api_key=os.getenv("DASHSCOPE_API_KEY")
        base_url=os.getenv("DASHSCOPE_BASE_URL")

        qwen_model = init_chat_model(
            model="qwen3-vl-flash",
            model_provider="openai",
            api_key=api_key,
            base_url=base_url
        )

        # 创建智能体
        agent = create_agent(
            model=qwen_model,
            system_prompt=system_prompt,
            response_format=Action
        )

        # 获取本地图片
        with open(imageInfo, 'rb') as f:
            image_bytes = f.read()

        # 转换为base64编码
        import base64
        image = base64.b64encode(image_bytes).decode("utf-8")

        # 生成消息
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": userInfo},
                {
                    "type": "image",
                    "base64": image,
                    "mime_type": "image/jpeg",
                }
            ]
        }

        # 阻塞调用
        response = agent.invoke({"messages": [message]})
        action = response['structured_response']
        print(action)

        return action



