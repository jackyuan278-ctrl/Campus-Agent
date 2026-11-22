# API 协议说明

> 本文件描述 Campus-Agent 暴露的两套对话接口及其协议差异。两个接口内部复用同一套 LangGraph 状态图（`graph.py`），但协议层完全独立、互不影响。

---

## 1. 接口总览

| | POST /assistant/chat | POST /v1/responses |
|---|---|---|
| **定位** | CampusHub 前端专用接口 | OpenAI Responses API 兼容层（Codex CLI / Cursor 用） |
| **前端协议** | 私有 SSE 格式 | Responses API 协议（`response.*` 事件） |
| **鉴权** | 必须带 JWT（`Authorization: Bearer`） | 无（可选 thread_id） |
| **多租户** | 严格隔离（`ensure_owner` 校验 thread_id 归属） | 不校验（thread_id 为 UUID，仅用于会话续接） |
| **附件支持** | 是（ChatRequest.attachments 数组） | 否（attachments 留空） |
| **用户配额** | 是（token + 上传次数） | 否（user_key 留空，跳过配额写入） |
| **历史查询** | 有（`GET /assistant/chat/messages`） | 无（Responses API 客户端自管历史） |
| **流式默认** | true（可 `?stream=false`） | true（由 `stream` 字段控制） |

---

## 2. POST /assistant/chat（私有 SSE 格式）

### 请求体

```json
{
  "message": string,        // 必填，0~4000 字符，仅发附件时可为空
  "thread_id": string,      // 必填，1~64 字符
  "mode": string,           // 可选，默认 "chat"，可选：chat / market / activity / doc / casual
  "attachments": [          // 可选，最多 5 个
    {
      "name": string,       // 文件名
      "type": string,       // MIME type，如 "image/png" / "application/pdf"
      "size": integer,      // 字节数
      "url": string         // OSS 公网 URL（必须 http/https，禁止内网地址）
    }
  ]
}
```

### 请求头

| Header | 必填 | 说明 |
|--------|------|------|
| `Authorization: Bearer <jwt>` | 是 | JWT payload 需含 `userId` 字段，服务端验签 |
| `Content-Type: application/json` | 是 | — |

### Query 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `stream` | true | false 时一次性返回 JSON，不走 SSE |

### SSE 事件帧（流式模式）

Content-Type: `text/event-stream`

**连接注释帧**（首个帧，让浏览器立刻看到首字节）：
```
: connected

```

**文本增量帧**：
```
data: {"delta": "这是", "mode": null}

data: {"delta": "一段回答", "mode": null}
```

**错误帧**（业务异常 / 未知异常）：
```
data: {"delta": "\n\n（今日额度已用完）", "error": "quota_exceeded"}

data: {"delta": "\n\n（服务异常，请稍后重试）", "error": "internal_error"}
```

**结束帧**（每轮对话必发）：
```
data: [DONE]

```

### 非流式响应（`?stream=false`）

```json
{
  "data": {
    "content": "完整回答文本",
    "thread_id": "abc123",
    "intent": "chat"
  }
}
```

### SSE 事件类型汇总表

| 帧 | 格式 | 触发时机 |
|----|------|---------|
| `: connected` | SSE 注释（以冒号开头） | 连接建立后立即发送，在拿到 thread_lock 之前 |
| `data: {"delta": "..."}` | JSON 文本增量 | agent 节点模型 stream 每个 chunk |
| `data: {"delta": "...", "error": "..."}` | JSON 错误 | AppError 捕获（如 LLMError / QuotaExceededError） |
| `data: [DONE]` | 纯文本 | 每次对话结束必发，无论成功失败 |

**重要**：SSE 事件**不包含** tool_calls 事件。`/assistant/chat` 只暴露最终文本回答，工具调用细节在服务端完成后才返回。前端永远看不到中间的 tool_calls。

---

## 3. POST /v1/responses（Responses API 兼容层）

### 请求体

```json
{
  "model": string,              // 占位字段，服务端忽略——走 config.GLM_CHAT_MODEL
  "instructions": string | null,// 可选，覆盖 system prompt（当前未实现透传，占位）
  "input": string,              // 必填，用户输入，0~4000 字符
  "stream": boolean,            // 默认 true
  "thread_id": string | null,   // 不传则服务端生成 UUID
  "mode": string                // 默认 "chat"
}
```

### 鉴权

无。端点不校验 JWT，thread_id 为 UUID（仅用于 LangGraph checkpoint 续接）。

### SSE 事件帧（流式模式）

Content-Type: `text/event-stream`

**事件类型对齐 OpenAI Responses API**：

| 事件 | 格式 | 说明 |
|------|------|------|
| `response.created` | `{"type": "response.created", "item_id": "msg_xxx"}` | 对话开始第一个帧 |
| `response.output_text.delta` | `{"type": "response.output_text.delta", "delta": "...", "item_id": "msg_xxx"}` | 逐 token 增量 |
| `response.function_call` | `{"type": "response.function_call", ...}` | **当前未实现**（占位事件） |
| `response.completed` | `{"type": "response.completed", "item_id": "msg_xxx"}` | 对话正常结束 |
| `data: [DONE]` | 纯文本 | 兼容层结束标记 |

> ⚠️ 错误时文本会包含在 `response.output_text.delta` 中（delta 字段带括号文案，如 `"（服务异常，请稍后重试）"`），不会单独发 error 事件。

### 非流式响应（stream=false）

```json
{
  "thread_id": "a1b2c3d4e5f6",
  "output_text": "完整回答文本",
  "mode": "chat"
}
```

### tool_calls 透传说明

**当前不支持。** 两个端点都做了统一过滤：

```python
# chat.py 和 responses.py 都有这段逻辑
async for event in graph.astream_events(inputs, config, version="v2"):
    if event["event"] != "on_chat_model_stream":
        continue                              # 跳过 tool_start / tool_end 等
    if (event.get("metadata") or {}).get("langgraph_node") != "agent":
        continue                              # 跳过 summarize 节点的模型调用
    text = getattr(event["data"].get("chunk"), "content", "") or ""
    if text:
        yield _sse({"delta": text})
```

LangGraph 在执行工具循环时会产生 `on_tool_start` / `on_tool_end` 事件，但两个端点的 stream 都只转发 `on_chat_model_stream` 且限定 `langgraph_node == "agent"`——**所有工具调用细节对客户端透明**。

如需对外暴露 tool_calls（例如让 Codex CLI 展示工具调用过程），需要在 `responses.py` 的 `event_stream()` 中额外转发 `on_tool_start` 事件并映射为 `response.function_call` 格式。

---

## 4. 两套 SSE 格式对比表

| 维度 | /assistant/chat | /v1/responses |
|------|-----------------|---------------|
| **首帧** | `: connected`（注释） | `response.created`（JSON） |
| **文本增量** | `{"delta": "文本"}` | `{"type": "response.output_text.delta", "delta": "文本", "item_id": "..."}` |
| **错误传递** | 独立 `{"delta": "...", "error": "xxx"}` 帧 | 把错误文案塞进 delta 字段，不单独发 error 事件 |
| **结束帧** | 只有 `[DONE]` | `response.completed` + `[DONE]` |
| **事件类型字段** | 无 type 字段，靠帧结构区分 | 每帧都有 `type` 字段（response.* 前缀） |
| **item_id** | 无 | 有，UUID 格式，供客户端关联多个 output 块 |
| **tool_calls 事件** | 不发 | 预留 `response.function_call` 类型但未实现 |

---

## 5. 内部复用机制

两个端点**完全解耦**（各自的 request schema、各自的 SSE 格式化逻辑），但复用同一个 LangGraph 状态图实例：

```
/assistant/chat                 /v1/responses
      │                              │
      ├── _get_graph()              ├── _get_graph()
      │   （从 request.app.state    │   （同一个函数、同一个 graph 对象）
      │     .agent_runtime 取图）   │
      │                              │
      ├── graph.astream_events()    ├── graph.astream_events()
      │   （同一套 inputs/config）  │   （同套 graph，inputs 略有不同：
      │                              │    user_key 留空、attachments 空数组）
      └── SSE 格式化（私有格式）    └── SSE 格式化（Responses API 格式）
```

graph 是进程内单例（AgentRuntime 在 lifespan 时构建一次，存到 `request.app.state.agent_runtime`）。两个端点都只做「协议层」的转换，核心逻辑全在 graph 里。

### thread_lock 串行化

两个端点的流式和非流式路径都用 `async with thread_lock(thread_id)` 包住图执行。同一个 thread_id 的请求**严格串行**——因为 LangGraph 的 checkpointer 是 last-write-wins，并发写会静默覆盖历史。
