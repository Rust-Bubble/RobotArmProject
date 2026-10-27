// 与后端 /ws/chat 的 WebSocket 通信客户端。
// 事件类型与后端 Orchestrator.handle 约定一致。

export interface ToolResultEvent {
  type: "tool_result";
  tool: string;
  ok?: boolean;
  result?: unknown;
  error?: string;
}

/** 机械臂状态事件：由 ArmController 逐步上报，见后端 arm_control/status.py */
export interface RobotStatusEvent {
  type: "robot_status";
  state?: string; // idle / busy / estop
  phase?: string;
  driver?: string;
  target?: string;
  updated_at?: string;
}

/** 会话历史消息：由后端 MemoryManager 从数据库恢复（见 qiming/memory/manager.py） */
export interface HistoryMessage {
  role: "user" | "agent";
  text: string;
  agent?: string;
}

export interface HistoryEvent {
  type: "history";
  session_id?: string;
  messages?: HistoryMessage[];
}

export type ChatEvent =
  | ({
      type: "status" | "intent" | "agent_start" | "agent_result" | "error" | "done" | "pong";
      stage?: string;
      message?: string;
      intent?: string;
      agent?: string;
      tool_result?: {
        tool: string;
        ok?: boolean;
        result?: unknown;
        error?: string;
      } | null;
      task?: {
        type: string;
        action_type?: string;
        target_object?: string;
        params?: Record<string, unknown>;
        command?: string;
      } | null;
    })
  | ToolResultEvent
  | RobotStatusEvent
  | HistoryEvent;

export interface ChatClientHandlers {
  onOpen?: () => void;
  /** 主动断开或重连放弃后触发（4401 = token 失效，应回登录页） */
  onClose?: (code: number) => void;
  onEvent: (event: ChatEvent) => void;
  /** 断线后开始自动重连时触发，前端可据此显示"重连中"提示 */
  onReconnecting?: (attempt: number, delayMs: number) => void;
}

/** 自动重连参数 */
const RECONNECT_BASE_DELAY = 1000; // 首次重连等待 1s
const RECONNECT_MAX_DELAY = 15000; // 退避上限 15s
const RECONNECT_MAX_ATTEMPTS = 8; // 连续失败 8 次后放弃，交给用户手动刷新
const HEARTBEAT_INTERVAL = 30000; // 每 30s 发一次 ping 保活

export class ChatClient {
  private ws: WebSocket | null = null;
  private handlers: ChatClientHandlers;
  // 防止 close() 后触发自动重连
  private intentionallyClosed = false;
  private token = "";
  private attempts = 0;
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(handlers: ChatClientHandlers) {
    this.handlers = handlers;
  }

  connect(token: string): void {
    this.intentionallyClosed = false;
    this.token = token;
    this.openSocket();
  }

  private openSocket(): void {
    const scheme = window.location.protocol === "https:" ? "wss" : "ws";
    this.ws = new WebSocket(
      `${scheme}://${window.location.host}/ws/chat?token=${encodeURIComponent(this.token)}`,
    );
    this.ws.onopen = () => {
      this.attempts = 0;
      this.startHeartbeat();
      this.handlers.onOpen?.();
    };
    this.ws.onclose = (e) => {
      this.stopHeartbeat();
      // token 失效（4401）重连也没有意义，直接上抛给登录流程
      if (e.code === 4401 || this.intentionallyClosed) {
        this.handlers.onClose?.(e.code);
        return;
      }
      this.scheduleReconnect();
    };
    this.ws.onmessage = (e) => {
      try {
        this.handlers.onEvent(JSON.parse(e.data) as ChatEvent);
      } catch {
        // 忽略非 JSON 消息
      }
    };
  }

  /** 指数退避自动重连：1s -> 2s -> 4s … 上限 15s */
  private scheduleReconnect(): void {
    if (this.intentionallyClosed) return;
    if (this.attempts >= RECONNECT_MAX_ATTEMPTS) {
      this.handlers.onClose?.(1006);
      return;
    }
    this.attempts += 1;
    const delay = Math.min(
      RECONNECT_BASE_DELAY * 2 ** (this.attempts - 1),
      RECONNECT_MAX_DELAY,
    );
    this.handlers.onReconnecting?.(this.attempts, delay);
    this.reconnectTimer = setTimeout(() => {
      if (!this.intentionallyClosed) this.openSocket();
    }, delay);
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      if (this.ready) this.ws?.send(JSON.stringify({ type: "ping" }));
    }, HEARTBEAT_INTERVAL);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  get ready(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  sendText(content: string): void {
    if (!this.ready) return;
    this.ws?.send(JSON.stringify({ type: "text", content }));
  }

  ping(): void {
    if (!this.ready) return;
    this.ws?.send(JSON.stringify({ type: "ping" }));
  }

  /** 请求后端新建会话（清空上下文，旧会话记录保留在数据库） */
  newSession(): void {
    if (!this.ready) return;
    this.ws?.send(JSON.stringify({ type: "new_session" }));
  }

  close(): void {
    this.intentionallyClosed = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.stopHeartbeat();
    this.ws?.close();
    this.ws = null;
  }

  get closedIntentionally(): boolean {
    return this.intentionallyClosed;
  }
}
