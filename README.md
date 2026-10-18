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

## Web 录音、STT 与 TTS

项目已支持在浏览器中录音，通过后端代理调用 OpenAI-compatible
音频转写接口，并把 TTS 音频流直接转发到浏览器播放。先安装依赖，再根据
`.env.example` 配置环境变量：

```powershell
python -m pip install -r requirements.txt
$env:STT_API_URL = "https://your-stt-service.example/v1/audio/transcriptions"
$env:STT_API_KEY = "replace-me"
$env:STT_MODEL = "whisper-1"
$env:TTS_API_URL = "https://api.openai.com/v1/audio/speech"
$env:TTS_API_KEY = "replace-me"
$env:TTS_MODEL = "gpt-4o-mini-tts"
$env:TTS_VOICE = "alloy"
python -m uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
```

如果使用 `src/qiming/mllm/.env` 中已有的 DashScope 配置，TTS 会自动复用
`DASHSCOPE_API_KEY` 和 `DASHSCOPE_BASE_URL`。在该文件中增加以下配置即可：

```dotenv
TTS_PROVIDER=dashscope
TTS_MODEL=qwen-audio-3.0-tts-flash
TTS_VOICE=longanhuan_v3.6
TTS_RESPONSE_FORMAT=mp3
STT_PROVIDER=dashscope
STT_MODEL=qwen3-asr-flash
```

另开一个终端启动前端：

```powershell
cd frontend
pnpm install
pnpm dev
```

访问 `http://127.0.0.1:5173`，允许麦克风权限后即可录音转写。转写完成后，
页面会自动请求 TTS 并优先使用 MediaSource 边接收边播放；不支持 MediaSource
音频流的浏览器会降级为完整下载后播放。API Key 只保存在后端环境变量中，
不会下发到浏览器。

## 后续开发重点

```text
1. 接入 ASR，把麦克风语音转成文本。
2. 接入摄像头，获取真实图像。
3. 接入 MLLM，让模型识别图像并输出动作。
4. 接入 MyCobot，让机械臂执行抓取或移动。
```
