import { useCallback, useEffect, useRef, useState } from "react";
import {
  AudioLines,
  Bot,
  CircleStop,
  Cpu,
  LogOut,
  Mic,
  RefreshCw,
  Send,
  Volume2,
  Waves,
} from "lucide-react";
import {
  ChatClient,
  type ChatEvent,
  type HistoryEvent,
  type RobotStatusEvent,
  type ToolResultEvent,
} from "./api/chatClient";
import {
  clearToken,
  getStoredToken,
  isStoredTokenExpired,
  login,
  register,
} from "./api/auth";

interface Message {
  id: number;
  role: "user" | "agent";
  text: string;
  meta?: string;
}

interface EventRow {
  label: string;
  sub: string;
  cls: "" | "active" | "complete" | "error";
}

interface TaskInfo {
  intent: string;
  target: string;
  params: string;
}

const EMPTY_TASK: TaskInfo = { intent: "—", target: "—", params: "—" };

/** 机械臂工作状态（后端 arm_control/status.py 的 RobotStatusHub 快照） */
interface RobotStatus {
  state: string; // idle / busy / estop
  phase: string;
  driver: string;
  target: string;
  updated_at: string;
}

const EMPTY_ROBOT: RobotStatus = {
  state: "idle",
  phase: "待机中",
  driver: "-",
  target: "",
  updated_at: "",
};

const ROBOT_STATE_LABELS: Record<string, string> = {
  idle: "空闲",
  busy: "执行中",
  estop: "已急停",
};

let messageSeq = 0;
const nextId = () => ++messageSeq;

// ---------- 浏览器语音识别（Web Speech API）最小类型声明 ----------
interface SpeechRecognitionResultLike {
  transcript: string;
}
interface SpeechRecognitionEventLike {
  results: ArrayLike<ArrayLike<SpeechRecognitionResultLike>>;
}
interface SpeechRecognitionLike {
  lang: string;
  interimResults: boolean;
  maxAlternatives: number;
  onresult: ((e: SpeechRecognitionEventLike) => void) | null;
  onerror: ((e: unknown) => void) | null;
  onend: (() => void) | null;
  start(): void;
  stop(): void;
}
type SpeechRecognitionCtor = new () => SpeechRecognitionLike;
const getSpeechRecognition = (): SpeechRecognitionCtor | null =>
  (window as unknown as { SpeechRecognition?: SpeechRecognitionCtor; webkitSpeechRecognition?: SpeechRecognitionCtor })
    .SpeechRecognition ??
  (window as unknown as { webkitSpeechRecognition?: SpeechRecognitionCtor }).webkitSpeechRecognition ??
  null;

export default function App() {
  const token = getStoredToken();

  // ---------- 登录态 ----------
  const [authed, setAuthed] = useState<boolean>(
    Boolean(token) && !isStoredTokenExpired(),
  );
  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const [authForm, setAuthForm] = useState({ username: "", email: "", password: "" });
  const [authError, setAuthError] = useState("");
  const [authBusy, setAuthBusy] = useState(false);

  // ---------- 会话状态 ----------
  const [messages, setMessages] = useState<Message[]>([]);
  const [waiting, setWaiting] = useState(false);
  const [input, setInput] = useState("");
  const [notice, setNotice] = useState("");
  const [wsConnected, setWsConnected] = useState(false);
  const [task, setTask] = useState<TaskInfo>(EMPTY_TASK);
  const [events, setEvents] = useState<EventRow[]>([]);
  const [robot, setRobot] = useState<RobotStatus>(EMPTY_ROBOT);

  const clientRef = useRef<ChatClient | null>(null);
  const streamRef = useRef<HTMLDivElement | null>(null);
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const [listening, setListening] = useState(false);

  // ---------- WebSocket 事件处理 ----------
  const handleEvent = useCallback((event: ChatEvent) => {
    switch (event.type) {
      case "history": {
        // 连接建立 / 新建会话时后端推来的会话历史（MemoryManager 持久化恢复）
        const historyEvent = event as HistoryEvent;
        setMessages(
          (historyEvent.messages ?? []).map((m) => ({
            id: nextId(),
            role: m.role,
            text: m.text,
            meta: m.agent ? `${m.agent} agent` : "历史消息",
          })),
        );
        break;
      }
      case "intent":
        setTask((prev) => ({ ...prev, intent: event.intent ?? "—" }));
        setEvents((prev) => [
          ...prev.map((e) => (e.cls === "active" ? { ...e, cls: "complete" as const, sub: event.intent ?? "" } : e)),
          { label: `意图识别：${event.intent ?? ""}`, sub: `分发到 ${event.agent ?? ""} agent`, cls: "complete" },
        ]);
        break;
      case "agent_start":
        setEvents((prev) => [
          ...prev,
          { label: `${event.agent ?? ""} agent 处理中`, sub: "调用大模型", cls: "active" },
        ]);
        break;
      case "agent_result":
        setMessages((prev) => [
          ...prev,
          {
            id: nextId(),
            role: "agent",
            text: event.message ?? "",
            meta: `${event.agent ?? ""} agent`,
          },
        ]);
        if (event.task) {
          const t = event.task;
          setTask({
            intent: t.type === "action" ? t.action_type ?? "动作" : t.command ?? t.type,
            target: t.target_object ?? "—",
            params: t.params && Object.keys(t.params).length > 0 ? JSON.stringify(t.params) : "—",
          });
        }
        setEvents((prev) => prev.map((e) => (e.cls === "active" ? { ...e, cls: "complete" as const, sub: "已完成" } : e)));
        break;
      case "tool_result": {
        const toolEvent = event as ToolResultEvent;
        const detail = toolEvent.ok
          ? JSON.stringify(toolEvent.result ?? {})
          : `失败：${toolEvent.error ?? "未知错误"}`;
        setEvents((prev) => [
          ...prev,
          {
            label: `工具调用：${toolEvent.tool}`,
            sub: `${toolEvent.ok ? "成功" : "失败"} · ${detail}`,
            cls: toolEvent.ok ? "complete" : "error",
          },
        ]);
        break;
      }
      case "robot_status": {
        const statusEvent = event as RobotStatusEvent;
        setRobot({
          state: statusEvent.state ?? "idle",
          phase: statusEvent.phase ?? "",
          driver: statusEvent.driver ?? "-",
          target: statusEvent.target ?? "",
          updated_at: statusEvent.updated_at ?? "",
        });
        break;
      }
      case "error":
        setMessages((prev) => [
          ...prev,
          { id: nextId(), role: "agent", text: `出错了：${event.message ?? "未知错误"}`, meta: "系统提示" },
        ]);
        setEvents((prev) => prev.map((e) => (e.cls === "active" ? { ...e, cls: "error" as const, sub: "失败" } : e)));
        break;
      case "status":
        setEvents((prev) => [
          ...prev,
          { label: event.message ?? "", sub: "进行中", cls: "active" },
        ]);
        break;
      default:
        break;
    }
    if (event.type === "done") setWaiting(false);
  }, []);

  const handleWsClose = useCallback(
    (code: number) => {
      setWsConnected(false);
      if (code === 4401) {
        // token 失效，回到登录页
        clearToken();
        setAuthed(false);
        return;
      }
      // 非鉴权原因断开：ChatClient 会自动重连，这里把关闭码暴露出来便于定位
      setNotice(`与后端断开（code ${code}），正在自动重连…`);
    },
    [],
  );

  // 建立连接：进入会话页且未连接时
  useEffect(() => {
    if (!authed) return;
    const currentToken = getStoredToken();
    if (!currentToken || isStoredTokenExpired()) {
      // token 缺失或已过期：清掉回到登录页，避免拿着旧 token 无限重连
      clearToken();
      setAuthed(false);
      return;
    }
    const client = new ChatClient({
      onOpen: () => {
        setWsConnected(true);
        setNotice("");
      },
      onClose: handleWsClose,
      onEvent: handleEvent,
      onReconnecting: (attempt) => {
        setWsConnected(false);
        setNotice(`连接中断，正在自动重连（第 ${attempt} 次）…`);
      },
    });
    client.connect(currentToken);
    clientRef.current = client;
    return () => {
      client.close();
      clientRef.current = null;
      setWsConnected(false);
    };
  }, [authed, handleEvent, handleWsClose]);

  // 对话流自动滚动到底部
  useEffect(() => {
    streamRef.current?.scrollTo({ top: streamRef.current.scrollHeight });
  }, [messages, waiting]);

  // ---------- 登录/注册 ----------
  const submitAuth = async () => {
    setAuthError("");
    setAuthBusy(true);
    try {
      if (authMode === "register") {
        await register(authForm.username, authForm.email, authForm.password);
        await login(authForm.username, authForm.password);
      } else {
        await login(authForm.username, authForm.password);
      }
      setMessages([]);
      setEvents([]);
      setTask(EMPTY_TASK);
      setRobot(EMPTY_ROBOT);
      setAuthed(true);
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : "操作失败");
    } finally {
      setAuthBusy(false);
    }
  };

  const logout = () => {
    clientRef.current?.close();
    clearToken();
    setAuthed(false);
    setMessages([]);
    setEvents([]);
    setTask(EMPTY_TASK);
    setRobot(EMPTY_ROBOT);
    setWsConnected(false);
  };

  // ---------- 发送文本 ----------
  const sendMessage = (content: string, meta: string) => {
    const text = content.trim();
    if (!text || waiting || !clientRef.current?.ready) return;
    setNotice("");
    setMessages((prev) => [...prev, { id: nextId(), role: "user", text, meta }]);
    setEvents([]);
    setWaiting(true);
    clientRef.current.sendText(text);
  };

  const sendText = () => sendMessage(input, "文本输入");

  const resetSession = () => {
    // 通知后端新建会话（服务端会回推一条空的 history），本地状态随之清空
    clientRef.current?.newSession();
    setEvents([]);
    setTask(EMPTY_TASK);
    setWaiting(false);
    setNotice("");
  };

  // ---------- 麦克风（浏览器语音识别 -> 文本 -> WebSocket） ----------
  const toggleMic = () => {
    if (listening) {
      recognitionRef.current?.stop();
      return;
    }
    const SRClass = getSpeechRecognition();
    if (!SRClass) {
      setNotice("当前浏览器不支持语音识别（推荐使用 Chrome / Edge），请先用文本输入。");
      return;
    }
    if (!wsConnected || waiting) {
      setNotice(waiting ? "上一条消息还在处理中，请稍候。" : "后端未连接，无法发送语音消息。");
      return;
    }
    const recognition = new SRClass();
    recognition.lang = "zh-CN";
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.onresult = (e) => {
      const transcript = e.results?.[0]?.[0]?.transcript ?? "";
      if (transcript.trim()) sendMessage(transcript.trim(), "语音输入");
    };
    recognition.onerror = () => {
      setListening(false);
      setNotice("语音识别失败，请重试或改用文本输入。");
    };
    recognition.onend = () => setListening(false);
    recognitionRef.current = recognition;
    try {
      recognition.start();
      setListening(true);
      setNotice("");
    } catch {
      setListening(false);
      setNotice("麦克风启动失败，请检查浏览器麦克风权限。");
    }
  };

  // ================= 登录视图 =================
  if (!authed) {
    return (
      <main className="app-shell auth-shell">
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
        </header>

        <section className="auth-panel">
          <div className="auth-title">
            <span>AGENT ACCESS</span>
            <h1>{authMode === "login" ? "登录以开始对话" : "创建新账号"}</h1>
            <p>登录后前端将通过 WebSocket 与后台大模型通信。</p>
          </div>

          <div className="auth-form">
            <label>
              <span>用户名</span>
              <input
                value={authForm.username}
                onChange={(e) => setAuthForm({ ...authForm, username: e.target.value })}
                placeholder="username"
                autoComplete="username"
              />
            </label>
            {authMode === "register" && (
              <label>
                <span>邮箱</span>
                <input
                  value={authForm.email}
                  onChange={(e) => setAuthForm({ ...authForm, email: e.target.value })}
                  placeholder="user@example.com"
                  autoComplete="email"
                />
              </label>
            )}
            <label>
              <span>密码</span>
              <input
                type="password"
                value={authForm.password}
                onChange={(e) => setAuthForm({ ...authForm, password: e.target.value })}
                placeholder="password"
                autoComplete={authMode === "login" ? "current-password" : "new-password"}
              />
            </label>

            {authError && <p className="auth-error" role="alert">{authError}</p>}

            <button type="button" onClick={submitAuth} disabled={authBusy}>
              {authBusy ? "请稍候…" : authMode === "login" ? "登录" : "注册并登录"}
            </button>
            <button
              type="button"
              className="auth-switch"
              onClick={() => {
                setAuthMode(authMode === "login" ? "register" : "login");
                setAuthError("");
              }}
            >
              {authMode === "login" ? "没有账号？去注册" : "已有账号？去登录"}
            </button>
          </div>
        </section>
      </main>
    );
  }

  // ================= 会话视图 =================
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
          <span className={`connection ${wsConnected ? "" : "down"}`}>
            <i />
            {wsConnected ? "后端已连接" : "后端未连接"}
          </span>
          <button className="emergency" type="button">
            <CircleStop size={17} />
            紧急停止
          </button>
          <button className="logout" type="button" onClick={logout} aria-label="退出登录">
            <LogOut size={15} />
            退出
          </button>
        </div>
      </header>

      <div className="workspace">
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

          <div className="conversation-stream" aria-live="polite" ref={streamRef}>
            {messages.length === 0 && !waiting && (
              <div className="empty-conversation">
                <span>
                  <AudioLines size={27} />
                </span>
                <h2>从一句话开始</h2>
                <p>
                  试试输入"帮我拿桌上的水杯"（动作指令）、"你是谁"（闲聊）或"查询状态"（系统指令），
                  Orchestrator 会自动识别意图并分发给对应的 Agent。
                </p>
              </div>
            )}

            {messages.map((msg) => (
              <div key={msg.id} className={`turn ${msg.role === "user" ? "user-turn" : "agent-turn"}`}>
                <div className="turn-meta">
                  {msg.role === "user" ? (
                    <span>你</span>
                  ) : (
                    <span className="agent-name">
                      <Bot size={14} />
                      启明
                    </span>
                  )}
                  <small>{msg.meta ?? ""}</small>
                </div>
                <div className="turn-content">
                  <p>{msg.text}</p>
                </div>
              </div>
            ))}

            {waiting && (
              <div className="turn agent-turn">
                <div className="turn-meta">
                  <span className="agent-name">
                    <Bot size={14} />
                    启明
                  </span>
                  <small>思考中</small>
                </div>
                <div className="turn-content">
                  <div className="listening-line">
                    <span className="listening-dot" />
                    Orchestrator 正在调度 Agent…
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="text-composer">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") sendText();
              }}
              placeholder={wsConnected ? "输入消息，回车发送…" : "后端未连接，无法发送"}
              disabled={!wsConnected}
            />
            <button
              type="button"
              onClick={sendText}
              disabled={!wsConnected || waiting || !input.trim()}
              aria-label="发送"
            >
              <Send size={16} />
            </button>
            <button
              type="button"
              className={`mic-button small${listening ? " listening" : ""}`}
              onClick={toggleMic}
              aria-label={listening ? "停止语音输入" : "语音输入"}
            >
              <Mic size={18} />
            </button>
          </div>

          {notice && (
            <p className="permission-notice" role="status">
              {notice}
            </p>
          )}

          <footer className="interface-note">
            <span>STT · 浏览器语音识别</span>
            <span>{wsConnected ? "Agent · 已连接" : "Agent · 未连接"}</span>
            <span>
              <Volume2 size={12} />
              TTS · 未接入
            </span>
          </footer>
        </section>

        <aside className="control-panel" aria-label="任务控制台">
          <div className="panel-content" id="agent-console">
            <div className="panel-title">
              <span>AGENT CONSOLE</span>
              <h2>任务控制台</h2>
            </div>

            <section className="console-section">
              <div className="section-label">
                <span>当前任务</span>
                <small>{waiting ? "执行中" : messages.length > 0 ? "空闲" : "暂无"}</small>
              </div>
              <dl className="task-fields">
                <div>
                  <dt>意图</dt>
                  <dd>{task.intent}</dd>
                </div>
                <div>
                  <dt>目标</dt>
                  <dd>{task.target}</dd>
                </div>
                <div>
                  <dt>动作参数</dt>
                  <dd>{task.params}</dd>
                </div>
              </dl>
              <p className="empty-task">
                {messages.length === 0
                  ? "收到有效的 Agent 输出后，这里将展示结构化任务。"
                  : waiting
                    ? "Orchestrator 正在识别意图并分发任务。"
                    : "上一轮任务已结束，等待新的指令。"}
              </p>
            </section>

            <section className="console-section robot-card">
              <div className="section-label">
                <span>机械臂状态</span>
                <small>{robot.updated_at || "—"}</small>
              </div>

              <div className={`robot-state ${robot.state}`}>
                <i />
                <strong>{ROBOT_STATE_LABELS[robot.state] ?? robot.state}</strong>
                <small>
                  {robot.state === "busy" ? "动作进行中" : robot.state === "estop" ? "等待复位" : "可接受指令"}
                </small>
              </div>

              <dl className="task-fields">
                <div>
                  <dt>当前阶段</dt>
                  <dd>{robot.phase || "—"}</dd>
                </div>
                <div>
                  <dt>驱动模式</dt>
                  <dd>{robot.driver || "—"}</dd>
                </div>
                <div>
                  <dt>目标物体</dt>
                  <dd>{robot.target || "—"}</dd>
                </div>
              </dl>
            </section>

            <section className="console-section">
              <div className="section-label">
                <span>执行事件</span>
                <small>实时</small>
              </div>
              <div className="event-stream">
                {events.length === 0 ? (
                  <div>
                    <i />
                    <span>
                      <strong>等待用户输入</strong>
                      <small>当前状态</small>
                    </span>
                  </div>
                ) : (
                  events.map((ev, idx) => (
                    <div key={idx} className={ev.cls}>
                      <i />
                      <span>
                        <strong>{ev.label}</strong>
                        <small>{ev.sub}</small>
                      </span>
                    </div>
                  ))
                )}
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
                <small>{wsConnected ? "WebSocket 已连接" : "未连接"}</small>
              </div>
              <div>
                <span>
                  <Bot size={15} />
                  Orchestrator
                </span>
                <small>{wsConnected ? "就绪" : "不可用"}</small>
              </div>
            </section>
          </div>
        </aside>
      </div>
    </main>
  );
}
