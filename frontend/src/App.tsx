import {
  AudioLines,
  Bot,
  CheckCircle2,
  ChevronLeft,
  CircleStop,
  Cpu,
  LoaderCircle,
  Mic,
  RefreshCw,
  TriangleAlert,
  Volume2,
  Waves,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";

type SessionState =
  | "idle"
  | "recording"
  | "transcribing"
  | "complete"
  | "error";
type ServiceState = "checking" | "ready" | "unconfigured" | "offline";

const AUDIO_TYPES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/mp4",
  "audio/ogg;codecs=opus",
];

function supportedAudioType() {
  return AUDIO_TYPES.find((type) => MediaRecorder.isTypeSupported(type)) ?? "";
}

function recordingFilename(mimeType: string) {
  if (mimeType.includes("mp4")) return "recording.mp4";
  if (mimeType.includes("ogg")) return "recording.ogg";
  return "recording.webm";
}

async function readError(response: Response) {
  try {
    const payload = (await response.json()) as { detail?: string };
    return payload.detail || `转写失败（${response.status}）`;
  } catch {
    return `转写失败（${response.status}）`;
  }
}

export default function App() {
  const [state, setState] = useState<SessionState>("idle");
  const [serviceState, setServiceState] = useState<ServiceState>("checking");
  const [elapsed, setElapsed] = useState(0);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [notice, setNotice] = useState("");
  const [transcript, setTranscript] = useState("");
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<number | null>(null);
  const requestRef = useRef<AbortController | null>(null);

  const clearTimer = () => {
    if (timerRef.current !== null) window.clearInterval(timerRef.current);
    timerRef.current = null;
  };

  const stopTracks = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  };

  const resetSession = () => {
    clearTimer();
    requestRef.current?.abort();
    requestRef.current = null;
    if (recorderRef.current?.state === "recording") {
      recorderRef.current.onstop = null;
      recorderRef.current.stop();
    }
    stopTracks();
    recorderRef.current = null;
    chunksRef.current = [];
    setElapsed(0);
    setNotice("");
    setTranscript("");
    setState("idle");
  };

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/health", { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error("health check failed");
        return (await response.json()) as { stt_configured?: boolean };
      })
      .then((health) =>
        setServiceState(health.stt_configured ? "ready" : "unconfigured"),
      )
      .catch((error: unknown) => {
        if (!(error instanceof DOMException && error.name === "AbortError")) {
          setServiceState("offline");
        }
      });

    return () => {
      controller.abort();
      clearTimer();
      requestRef.current?.abort();
      if (recorderRef.current?.state === "recording") {
        recorderRef.current.onstop = null;
        recorderRef.current.stop();
      }
      stopTracks();
    };
  }, []);

  const submitRecording = async (audio: Blob, mimeType: string) => {
    if (!audio.size) {
      setNotice("未录到有效音频，请重试。");
      setState("error");
      return;
    }

    const controller = new AbortController();
    requestRef.current = controller;
    const formData = new FormData();
    formData.append("file", audio, recordingFilename(mimeType));

    try {
      const response = await fetch("/api/stt/transcribe", {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(await readError(response));

      const payload = (await response.json()) as { text?: unknown };
      if (typeof payload.text !== "string" || !payload.text.trim()) {
        throw new Error("STT 服务未返回转写文本。");
      }
      setTranscript(payload.text.trim());
      setNotice("");
      setState("complete");
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      setNotice(error instanceof Error ? error.message : "转写失败，请重试。");
      setState("error");
    } finally {
      if (requestRef.current === controller) requestRef.current = null;
    }
  };

  const stopRecording = () => {
    const recorder = recorderRef.current;
    if (!recorder || recorder.state !== "recording") return;
    clearTimer();
    setState("transcribing");
    recorder.stop();
    stopTracks();
  };

  const startRecording = async () => {
    setNotice("");
    setTranscript("");
    setElapsed(0);

    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      setNotice("当前浏览器不支持麦克风录入。");
      setState("error");
      return;
    }

    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      const mimeType = supportedAudioType();
      const recorder = new MediaRecorder(
        mediaStream,
        mimeType ? { mimeType } : undefined,
      );
      streamRef.current = mediaStream;
      recorderRef.current = recorder;
      chunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.onerror = () => {
        clearTimer();
        stopTracks();
        setNotice("录音过程中发生错误，请重试。");
        setState("error");
      };
      recorder.onstop = () => {
        const resolvedType = recorder.mimeType || mimeType || "audio/webm";
        const audio = new Blob(chunksRef.current, { type: resolvedType });
        recorderRef.current = null;
        chunksRef.current = [];
        void submitRecording(audio, resolvedType);
      };
      recorder.start(250);
      setState("recording");
      timerRef.current = window.setInterval(
        () => setElapsed((value) => value + 1),
        1000,
      );
    } catch (error) {
      stopTracks();
      const denied =
        error instanceof DOMException &&
        (error.name === "NotAllowedError" || error.name === "SecurityError");
      setNotice(
        denied
          ? "未获得麦克风权限，请允许访问后重试。"
          : "无法启动麦克风，请检查设备后重试。",
      );
      setState("error");
    }
  };

  const toggleRecording = async () => {
    if (state === "recording") {
      stopRecording();
      return;
    }
    if (state === "transcribing") return;
    if (state === "complete" || state === "error") resetSession();
    await startRecording();
  };

  const serviceCopy = {
    checking: "检查中",
    ready: "已就绪",
    unconfigured: "待配置",
    offline: "未连接",
  }[serviceState];

  const statusTitle =
    state === "recording"
      ? `${String(Math.floor(elapsed / 60)).padStart(2, "0")}:${String(
          elapsed % 60,
        ).padStart(2, "0")}`
      : state === "transcribing"
        ? "正在转写"
        : state === "complete"
          ? "转写完成"
          : state === "error"
            ? "请重试"
            : "语音输入";

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#" aria-label="启明智手首页">
          <span className="brand-mark"><Waves size={21} /></span>
          <span><strong>启明智手</strong><small>语音智能机械臂</small></span>
        </a>
        <div className="header-actions">
          <span className={`connection ${serviceState}`}><i />STT {serviceCopy}</span>
          <button className="emergency" type="button">
            <CircleStop size={17} />紧急停止
          </button>
        </div>
      </header>

      <section className="conversation-layout">
        <div className="conversation-header">
          <div><span>CONVERSATION</span><h1>与启明对话</h1></div>
          <button type="button" onClick={resetSession}><RefreshCw size={15} />新建会话</button>
        </div>

        <div className="conversation-stream" aria-live="polite">
          {state === "idle" ? (
            <div className="empty-conversation">
              <span><AudioLines size={27} /></span>
              <h2>从一次语音开始</h2>
              <p>点击麦克风说出指令，录音会自动上传并转写。</p>
            </div>
          ) : (
            <>
              <div className="turn user-turn">
                <div className="turn-meta"><span>你</span><small>语音输入</small></div>
                <div className="turn-content">
                  {state === "recording" && <div className="listening-line"><span className="listening-dot" />正在录音，说完后再次点击麦克风</div>}
                  {state === "transcribing" && <div className="listening-line"><LoaderCircle className="spin" size={16} />音频已上传，STT 服务正在转写…</div>}
                  {state === "complete" && <p className="transcript-copy">{transcript}</p>}
                  {state === "error" && <div className="error-line"><TriangleAlert size={16} />{notice}</div>}
                </div>
              </div>
              {state === "complete" && (
                <div className="turn agent-turn">
                  <div className="turn-meta"><span className="agent-name"><Bot size={14} />启明</span><small>系统提示</small></div>
                  <div className="turn-content"><p className="pending-copy">语音转写已完成，文本可继续传入 Agent 执行。</p></div>
                </div>
              )}
            </>
          )}
        </div>

        <div className={`voice-composer ${state}`}>
          <div className="voice-status">
            <span className="voice-bars" aria-hidden="true">
              {Array.from({ length: 9 }, (_, index) => <i key={index} style={{ "--i": index } as React.CSSProperties} />)}
            </span>
            <span>
              <strong>{statusTitle}</strong>
              <small>{state === "recording" ? "再次点击结束录音" : state === "transcribing" ? "请稍候，不要关闭页面" : state === "complete" || state === "error" ? "点击重新录入" : "点击麦克风开始说话"}</small>
            </span>
          </div>
          <button type="button" className="mic-button" onClick={toggleRecording} disabled={state === "transcribing"} aria-label={state === "recording" ? "结束语音录入" : "开始语音录入"}>
            {state === "recording" ? <span className="stop-square" /> : state === "transcribing" ? <LoaderCircle className="spin" size={20} /> : state === "complete" ? <CheckCircle2 size={21} /> : state === "error" ? <RefreshCw size={20} /> : <Mic size={21} />}
          </button>
        </div>

        {notice && state !== "error" && <p className="permission-notice" role="status">{notice}</p>}
        <footer className="interface-note">
          <span>STT · {serviceCopy}</span><span>Agent · 待接入</span>
          <span><Volume2 size={12} />TTS · 待接入</span>
        </footer>
      </section>

      <aside className={`control-drawer ${drawerOpen ? "open" : ""}`} onMouseEnter={() => setDrawerOpen(true)} onMouseLeave={() => setDrawerOpen(false)}>
        <button type="button" className="drawer-handle" onClick={() => setDrawerOpen((value) => !value)} onFocus={() => setDrawerOpen(true)} aria-expanded={drawerOpen} aria-controls="agent-console">
          <ChevronLeft size={17} /><span>控制台</span>
        </button>
        <div className="drawer-content" id="agent-console">
          <div className="drawer-title"><span>AGENT CONSOLE</span><h2>任务控制台</h2></div>
          <section className="console-section">
            <div className="section-label"><span>当前任务</span><small>{state === "idle" ? "暂无" : state === "recording" ? "录入中" : state === "transcribing" ? "转写中" : state === "complete" ? "已转写" : "异常"}</small></div>
            <dl className="task-fields"><div><dt>意图</dt><dd>—</dd></div><div><dt>目标</dt><dd>—</dd></div><div><dt>动作参数</dt><dd>—</dd></div></dl>
            <p className="empty-task">{state === "complete" ? `已识别文本：${transcript}` : "完成语音转写后，文本将作为 Agent 的指令输入。"}</p>
          </section>
          <section className="console-section">
            <div className="section-label"><span>执行事件</span><small>实时</small></div>
            <div className="event-stream">
              <div className={state !== "idle" ? "complete" : "active"}><i /><span><strong>等待语音输入</strong><small>{state !== "idle" ? "已完成" : "当前状态"}</small></span></div>
              <div className={state === "recording" ? "active" : ["transcribing", "complete"].includes(state) ? "complete" : ""}><i /><span><strong>采集麦克风音频</strong><small>{state === "recording" ? "进行中" : ["transcribing", "complete"].includes(state) ? "已完成" : "等待"}</small></span></div>
              <div className={state === "transcribing" ? "active" : state === "complete" ? "complete" : ""}><i /><span><strong>STT 语音转写</strong><small>{state === "transcribing" ? "进行中" : state === "complete" ? "已完成" : state === "error" ? "失败" : "等待"}</small></span></div>
              <div><i /><span><strong>机械臂执行</strong><small>等待</small></span></div>
            </div>
          </section>
          <section className="console-section devices">
            <div className="section-label"><span>服务状态</span></div>
            <div><span><Cpu size={15} />STT 服务</span><small className={serviceState === "ready" ? "status-ready" : ""}>{serviceCopy}</small></div>
            <div><span><Bot size={15} />机械臂</span><small>状态不可用</small></div>
          </section>
        </div>
      </aside>
    </main>
  );
}
