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
  onClose?: (code: number) => void;
  onEvent: (event: ChatEvent) => void;
}

export class ChatClient {
  private ws: WebSocket | null = null;
  private handlers: ChatClientHandlers;
  // 防止 close() 后触发自动重连
  private intentionallyClosed = false;

  constructor(handlers: ChatClientHandlers) {
    this.handlers = handlers;
  }

  connect(token: string): void {
    this.intentionallyClosed = false;
    const scheme = window.location.protocol === "https:" ? "wss" : "ws";
    this.ws = new WebSocket(
      `${scheme}://${window.location.host}/ws/chat?token=${encodeURIComponent(token)}`,
    );
    this.ws.onopen = () => this.handlers.onOpen?.();
    this.ws.onclose = (e) => this.handlers.onClose?.(e.code);
    this.ws.onmessage = (e) => {
      try {
        this.handlers.onEvent(JSON.parse(e.data) as ChatEvent);
      } catch {
        // 忽略非 JSON 消息
      }
    };
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
    this.ws?.close();
    this.ws = null;
  }

  get closedIntentionally(): boolean {
    return this.intentionallyClosed;
  }
}
