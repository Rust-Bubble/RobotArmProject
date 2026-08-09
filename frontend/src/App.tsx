import {
  AudioLines,
  Bot,
  ChevronLeft,
  CircleStop,
  Cpu,
  Mic,
  RefreshCw,
  Volume2,
  Waves,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";

type SessionState = "idle" | "recording" | "waiting";

export default function App() {
  const [state, setState] = useState<SessionState>("idle");
  const [elapsed, setElapsed] = useState(0);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [notice, setNotice] = useState("");
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<number | null>(null);

  useEffect(
    () => () => {
      if (timerRef.current) window.clearInterval(timerRef.current);
      streamRef.current?.getTracks().forEach((track) => track.stop());
    },
    [],
  );

  const resetSession = () => {
    if (timerRef.current) window.clearInterval(timerRef.current);
    streamRef.current?.getTracks().forEach((track) => track.stop());
    recorderRef.current = null;
    streamRef.current = null;
    timerRef.current = null;
    setElapsed(0);
    setNotice("");
    setState("idle");
  };

  const stopRecording = () => {
    recorderRef.current?.stop();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    if (timerRef.current) window.clearInterval(timerRef.current);
    timerRef.current = null;
    setState("waiting");
  };

  const toggleRecording = async () => {
    setNotice("");
    if (state === "recording") {
      stopRecording();
      return;
    }
    if (state === "waiting") {
      resetSession();
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setNotice("当前浏览器不支持麦克风录入。");
      return;
    }

    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: true,
      });
      streamRef.current = mediaStream;
      recorderRef.current = new MediaRecorder(mediaStream);
      recorderRef.current.start();
      setState("recording");
      setElapsed(0);
      timerRef.current = window.setInterval(
        () => setElapsed((value) => value + 1),
        1000,
      );
    } catch {
      setNotice("未获得麦克风权限，请允许访问后重试。");
    }
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#" aria-label="启明智手首页">
          <span className="brand-mark">
            <Waves size={21} />
          </span>
          <span>
            <strong>启明智手</strong>
            <small>语音智能机械臂</small>
          </span>
        </a>

        <div className="header-actions">
          <span className="connection">
            <i />
            前端已就绪
          </span>
          <button className="emergency" type="button">
            <CircleStop size={17} />
            紧急停止
          </button>
        </div>
      </header>

      <section className="conversation-layout">
        <div className="conversation-header">
          <div>
            <span>CONVERSATION</span>
            <h1>与启明对话</h1>
          </div>
          <button type="button" onClick={resetSession}>
            <RefreshCw size={15} />
            新建会话
          </button>
        </div>

        <div className="conversation-stream" aria-live="polite">
          {state === "idle" && (
            <div className="empty-conversation">
              <span>
                <AudioLines size={27} />
              </span>
              <h2>从一次语音开始</h2>
              <p>录音转写、Agent 回复和必要的确认信息会显示在这里。</p>
            </div>
          )}

          {state !== "idle" && (
            <>
              <div className="turn user-turn">
                <div className="turn-meta">
                  <span>你</span>
                  <small>{state === "recording" ? "正在输入" : "语音输入"}</small>
                </div>
                <div className="turn-content">
                  {state === "recording" ? (
                    <div className="listening-line">
                      <span className="listening-dot" />
                      正在录音，等待 STT 返回实时转写文本
                    </div>
                  ) : (
                    <p className="pending-copy">语音已录入，等待 STT 服务返回文本。</p>
                  )}
                </div>
              </div>

              {state === "waiting" && (
                <div className="turn agent-turn">
                  <div className="turn-meta">
                    <span className="agent-name">
                      <Bot size={14} />
                      启明
                    </span>
                    <small>系统提示</small>
                  </div>
                  <div className="turn-content">
                    <p>
                      STT 与 Agent 服务尚未接入，当前无法生成转写结果或回复。
                    </p>
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        <div className={`voice-composer ${state}`}>
          <div className="voice-status">
            <span className="voice-bars" aria-hidden="true">
              {Array.from({ length: 9 }, (_, index) => (
                <i key={index} style={{ "--i": index } as React.CSSProperties} />
              ))}
            </span>
            <span>
              <strong>
                {state === "recording"
                  ? `${String(Math.floor(elapsed / 60)).padStart(2, "0")}:${String(
                      elapsed % 60,
                    ).padStart(2, "0")}`
                  : state === "waiting"
                    ? "音频已接收"
                    : "语音输入"}
              </strong>
              <small>
                {state === "recording"
                  ? "再次点击结束录音"
                  : state === "waiting"
                    ? "点击重新录入"
                    : "点击麦克风开始说话"}
              </small>
            </span>
          </div>

          <button
            type="button"
            className="mic-button"
            onClick={toggleRecording}
            aria-label={
              state === "recording"
                ? "结束语音录入"
                : state === "waiting"
                  ? "重新录入"
                  : "开始语音录入"
            }
          >
            {state === "recording" ? (
              <span className="stop-square" />
            ) : state === "waiting" ? (
              <RefreshCw size={20} />
            ) : (
              <Mic size={21} />
            )}
          </button>
        </div>

        {notice && (
          <p className="permission-notice" role="status">
            {notice}
          </p>
        )}

        <footer className="interface-note">
          <span>STT · 未接入</span>
          <span>Agent · 未接入</span>
          <span>
            <Volume2 size={12} />
            TTS · 未接入
          </span>
        </footer>
      </section>

      <aside
        className={`control-drawer ${drawerOpen ? "open" : ""}`}
        onMouseEnter={() => setDrawerOpen(true)}
        onMouseLeave={() => setDrawerOpen(false)}
      >
        <button
          type="button"
          className="drawer-handle"
          onClick={() => setDrawerOpen((value) => !value)}
          onFocus={() => setDrawerOpen(true)}
          aria-expanded={drawerOpen}
          aria-controls="agent-console"
        >
          <ChevronLeft size={17} />
          <span>控制台</span>
        </button>

        <div className="drawer-content" id="agent-console">
          <div className="drawer-title">
            <span>AGENT CONSOLE</span>
            <h2>任务控制台</h2>
          </div>

          <section className="console-section">
            <div className="section-label">
              <span>当前任务</span>
              <small>
                {state === "idle"
                  ? "暂无"
                  : state === "recording"
                    ? "录入中"
                    : "等待后端"}
              </small>
            </div>
            <dl className="task-fields">
              <div>
                <dt>意图</dt>
                <dd>—</dd>
              </div>
              <div>
                <dt>目标</dt>
                <dd>—</dd>
              </div>
              <div>
                <dt>动作参数</dt>
                <dd>—</dd>
              </div>
            </dl>
            <p className="empty-task">
              {state === "idle"
                ? "收到有效的 Agent 输出后，这里将展示结构化任务。"
                : state === "recording"
                  ? "正在采集语音，尚未生成任务。"
                  : "等待 STT 与 Agent 接口返回任务数据。"}
            </p>
          </section>

          <section className="console-section">
            <div className="section-label">
              <span>执行事件</span>
              <small>实时</small>
            </div>
            <div className="event-stream">
              <div className={state !== "idle" ? "complete" : ""}>
                <i />
                <span>
                  <strong>等待语音输入</strong>
                  <small>{state !== "idle" ? "已完成" : "当前状态"}</small>
                </span>
              </div>
              <div className={state === "recording" ? "active" : state === "waiting" ? "complete" : ""}>
                <i />
                <span>
                  <strong>采集麦克风音频</strong>
                  <small>
                    {state === "recording"
                      ? "进行中"
                      : state === "waiting"
                        ? "已完成"
                        : "等待"}
                  </small>
                </span>
              </div>
              <div className={state === "waiting" ? "active" : ""}>
                <i />
                <span>
                  <strong>语音转写与意图识别</strong>
                  <small>{state === "waiting" ? "等待接口" : "等待"}</small>
                </span>
              </div>
              <div>
                <i />
                <span>
                  <strong>机械臂执行</strong>
                  <small>等待</small>
                </span>
              </div>
            </div>
          </section>

          <section className="console-section devices">
            <div className="section-label">
              <span>服务状态</span>
            </div>
            <div>
              <span>
                <Cpu size={15} />
                后端服务
              </span>
              <small>未连接</small>
            </div>
            <div>
              <span>
                <Bot size={15} />
                机械臂
              </span>
              <small>状态不可用</small>
            </div>
          </section>
        </div>
      </aside>
    </main>
  );
}
