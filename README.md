# 启明智手

这是一个简易原型工程，目标是先搭建整体代码框架：

```text
语音指令 -> 图像采集 -> MLLM 理解 -> 机械臂动作 -> 语音反馈
```

当前阶段不展开复杂实现，只保留核心模块边界，方便后续逐步接入真实硬件和模型。

## 目录结构

```text
config/
  robot.yaml        机械臂配置
  llm.yaml          MLLM 配置

src/
  main.py           主流程入口
  qiming/
    speech/         语音输入与播报
    vision/         摄像头图像采集
    mllm/           MLLM 识别与动作决策
    arm_control/    机械臂控制
    models.py       核心数据结构
    utils/          配置与日志工具

scripts/
  demo_text_command.py    文本模拟语音指令
```

## 当前运行方式

现在还没有接入真实麦克风、摄像头和 MLLM，可以先用文本模拟语音输入：

```powershell
python -m src.main --text "帮我拿桌上的水杯"
```

## Web 录音与 STT

项目已支持在浏览器中录音，并通过后端代理调用 OpenAI-compatible
音频转写接口。先安装依赖，再根据 `.env.example` 配置环境变量：

```powershell
python -m pip install -r requirements.txt
$env:STT_API_URL = "https://your-stt-service.example/v1/audio/transcriptions"
$env:STT_API_KEY = "replace-me"
$env:STT_MODEL = "whisper-1"
python -m uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
```

另开一个终端启动前端：

```powershell
cd frontend
pnpm install
pnpm dev
```

访问 `http://127.0.0.1:5173`，允许麦克风权限后即可录音转写。API Key
只保存在后端环境变量中，不会下发到浏览器。

## 后续开发重点

```text
1. 接入 ASR，把麦克风语音转成文本。
2. 接入摄像头，获取真实图像。
3. 接入 MLLM，让模型识别图像并输出动作。
4. 接入 MyCobot，让机械臂执行抓取或移动。
```
