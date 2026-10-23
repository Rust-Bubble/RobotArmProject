# 全局上下文管理 (Memory) 与联调一：语音流转后端稳定性测试

本文档说明两部分工作的设计与自测方法：

1. 系统运行全局上下文管理 (Memory) 与持久化存储
2. 联调一：前端语音流转至后端处理的稳定性测试

---

## 一、全局上下文管理 (Memory)

### 1.1 设计

```
前端 (WebSocket)
   │  {"type":"text","content":"帮我拿水杯"}
   ▼
src/api/chat.py  (/ws/chat)
   │  连接建立时: memory_manager.get_session(user.id)  ← 恢复或新建会话
   │  orchestrator.history = memory.history            ← 与调度器共享同一份历史
   ▼
src/qiming/memory/manager.py   MemoryManager（进程级全局单例）
   │  内存: 会话上下文缓存（dict[session_id → SessionMemory]）
   │  落库: 每轮对话在发送 done 前写入数据库
   ▼
SQLite / PostgreSQL
   conversation_sessions     会话表（属于哪个用户、何时更新）
   conversation_messages     消息表（每轮的用户输入 + 助手回复）
```

核心类（[src/qiming/memory/manager.py](../src/qiming/memory/manager.py)）：

- `MemoryManager`：全局上下文管理器，整个后端共享一个实例。
  - `get_session(user_id, session_id=None)`：恢复或新建会话。查找顺序：进程内缓存 → 数据库（指定 id）→ 该用户最近更新的会话（跨重启恢复）→ 新建。
  - `reset_session(user_id)`：新建会话（旧会话记录保留在数据库中可追溯）。
  - `record_turn(session_id, user_text, reply, agent)`：把一轮对话落库。
- `SessionMemory`：单个会话的运行时上下文，`history` 列表与 Orchestrator
  的 `self.history` 是同一份引用，调度器照常往里 append，无需感知持久化。

### 1.2 关键时序保证

**先落库、后发 `done`**：`chat.py` 在把 `done` 事件发给前端之前先执行
`record_turn`。这保证前端收到"本轮结束"时上下文一定已持久化——即使它随即
断开（如刷新页面），最后一轮也不会丢。测试 `test_reconnect_restores_history`
专门覆盖了这一点。

**落库失败不影响对话**：`record_turn` 异常只记日志，本轮对话照常返回。

### 1.3 协议变更（前后端约定）

连接建立后后端先推两条初始化消息：

```
{"type": "robot_status", ...}                                  ← 机械臂状态快照（原有）
{"type": "history", "session_id": "...", "messages": [...]}    ← 会话历史（新增）
```

新增前端 → 后端消息：

```
{"type": "new_session"}    ← 新建会话，后端清空上下文并回推一条空 history
```

前端对应改动（[frontend/src/App.tsx](../frontend/src/App.tsx)）：
收到 `history` 事件恢复对话界面；"新建会话"按钮改为通知后端；麦克风按钮
接入浏览器 Web Speech API（语音 → 文本 → 复用同一条 WebSocket 链路）。

---

## 二、联调一：稳定性测试

"前端语音流转至后端"链路 = 浏览器语音识别产出文本 → WebSocket 发送 →
后端意图识别 → Agent 处理 → 工具执行 → 事件流回传。稳定性测试覆盖三层：

### 2.1 pytest 自动化测试（无外部依赖，mock 链路）

| 文件 | 覆盖内容 |
| --- | --- |
| [tests/test_memory.py](../tests/test_memory.py) | 会话新建/恢复、持久化、模拟服务重启后恢复、多用户隔离、新建会话清空 |
| [tests/test_stability.py](../tests/test_stability.py) | 连续 24 轮混合指令 + 心跳穿插、单轮延迟、断线重连恢复历史、新建会话、脏消息容错 |

```powershell
./robot-arm-env/Scripts/python.exe -m pytest tests/ -v
```

### 2.2 端到端联调脚本（对真实运行的后端压测）

[scripts/stability_test.py](../scripts/stability_test.py) 模拟前端完整链路：
REST 注册/登录 → WebSocket 连接 → 连续多轮指令（混合三种意图）→ 断线重连
校验历史恢复 → 新建会话校验上下文清空。全部通过退出码为 0。

```powershell
# 终端 1：启动后端
./robot-arm-env/Scripts/python.exe -m uvicorn src.api.app:app --port 8000

# 终端 2：运行联调脚本
./robot-arm-env/Scripts/python.exe scripts/stability_test.py --rounds 20
```

### 2.3 手动联调（前端页面）

```powershell
# 终端 1：后端
./robot-arm-env/Scripts/python.exe -m uvicorn src.api.app:app --port 8000

# 终端 2：前端
cd frontend; pnpm dev
```

浏览器打开 http://127.0.0.1:5173 ，注册/登录后：

1. 点麦克风按钮（Chrome/Edge），说"帮我拿一下水杯" → 语音转文本自动发送，
   右侧控制台应出现完整事件流（意图 → agent → 工具调用 → 完成）。
2. 说几句话后**刷新页面** → 对话历史应自动恢复（Memory 持久化）。
3. 点"新建会话" → 对话清空，从零开始。
4. **重启后端**再刷新页面 → 历史依然恢复（说明落库而非仅内存）。

### 2.4 已解决的问题（稳定性测试发现）

- **SQLite `database is locked`**：对话历史每轮写库后暴露写锁竞争。
  解决：连接启用 WAL 模式 + `busy_timeout=30000`（[database/connection.py](../src/qiming/database/connection.py)）。
- **最后一轮丢失**：原时序是先发 `done` 再落库，前端收到后立刻断开会取消
  落库。解决：先落库后发 `done`。
