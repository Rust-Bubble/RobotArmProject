"""MLLM 识别与决策智能体入口，负责根据用户文本和图像判断目标物体与动作意图。"""

from dotenv import load_dotenv
import os
from langchain.chat_models import init_chat_model
from  langchain.agents import create_agent
from openai.types.responses import response_status
from pydantic import BaseModel


class Action(BaseModel):
    obj: str
    pos: tuple [float,float]


class MllmRecognizer:
    """MLLM 调用占位。"""

    def __init__(self) -> None:
        load_dotenv()
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.base_url = os.getenv("DASHSCOPE_BASE_URL")


    def agentCommunication(self,model_type,system_prompt,userInfo,imageInfo = None,responseForm = None):
        """
        单轮信息处理
        :param model_type: 模型类型
        :param system_prompt: 系统提示词
        :param userInfo: 用户消息
        :param imageInfo: 图片（地址）
        :return:模型输出
        """
        qwen_model = init_chat_model(
            model=model_type,
            model_provider="openai",
            api_key=self.api_key,
            base_url=self.base_url
        )

        # 创建智能体
        agent = create_agent(
            model=qwen_model,
            system_prompt=system_prompt,
            response_format=responseForm
        )


        if imageInfo:
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
        else:
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": userInfo},
                ]
            }

        # 阻塞调用
        response = agent.invoke({"messages": [message]})
        if responseForm:
            res = response['structured_response']
        else:
            res = response['messages'][1].content

        print(res)

        return res


    def recognize(self,userInfo,imageInfo) :
        """
        根据图片信息和指令返回物体位置（无法测定深度）
        :param userInfo:用户消息
        :param imageInfo:图片地址，如"C:/Users/under/Pictures/Saved Pictures/样本3.jpg"
        :return:Action类对象
        """

        system_prompt = """
            #身份
            你是一个负责识别物体位置的智能体。
            #指令
            现在需要你根据给出的物体名称，识别该物体在给定图片中的位置，请根据返回其几何中心点的平面坐标，坐标的单位以像素计算。如果图片中包含多个同种目标物体，请依次列出它们的坐标；若不包含目标物体，则返回[-1]。
            请以[(x坐标,y坐标)]的格式输出。
        """

        action = self.agentCommunication("qwen3-vl-flash",system_prompt,userInfo,imageInfo,Action)
        return action

    def tendRecognizer(self,userInfo):
        """
        意图识别
        :param userInfo: 用户消息
        :return: 意图类型（闲聊、系统指令、动作指令、其他）返回类型为字符串
        """
        system_prompt = """
        #指令
        请根据用户的消息判断该信息的意图，可供选择的意图类型有四种：闲聊、系统指令、动作指令、其他。
        请在以上四种类型中选择最接近用户指令意图的类型，并返回该类型。
        请勿返回四种类型以外的其他内容。
        """
        res = self.agentCommunication("qwen-flash",system_prompt,userInfo)
        return res

