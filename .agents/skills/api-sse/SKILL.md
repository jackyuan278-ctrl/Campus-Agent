---
name: api-sse
description: 改 SSE 推送格式、加端点、对齐 Responses API、改 schema 时触发。触发词：SSE、Responses API、v1/responses、endpoint、schema、stream。排除：日常代码改动、索引重建。
---

# api-sse

两个端点共用同一套 LangGraph 状态图（`graph.ainvoke` / `graph.astream_events`），协议层**完全解耦**。

## 端点对比

| | `POST /assistant/chat` | `POST /v1/responses` |
|---|---|---|
| 客户端 | 前端 campus-web | Codex CLI / Cursor 等外部 |
| 鉴权 | `require_user_id` + `ensure_owner` + 配额预检 | 无（`user_key=""`，跳过配额） |
| 非流式返回 | `{"data": {"content": ..., "thread_id": ..., "intent": ...}}` | `{"thread_id": ..., "output_text": ..., "mode": ...}` |
| 代码位置 | `app/api/v1/chat.py` | `app/api/v1/responses.py` |

## SSE 事件帧格式

### /assistant/chat

| 帧 | 示例 `data:` 内容 | 说明 |
|---|---|---|
| 首字节（注释帧） | `: connected` | 前端立刻看到连接，然后排队等 `thread_lock` |
| token 增量 | `{"delta": "你好"}` | 仅转发 `langgraph_node == "agent"` 的 `on_chat_model_stream`（过滤掉 summarize 节点的模型输出） |
| 异常兜底 | `{"delta": "\n\n（服务异常，请稍后重试）", "error": "internal_error"}` | **未知异常原文绝对不能推给前端**，里面可能带数据库地址 |
| 结束 | `data: [DONE]` | finally 块里 yield |

### /v1/responses（对齐 OpenAI Responses API）

| 帧 | 示例 `data:` 内容 |
|---|---|
| 创建 | `{"type": "response.created", "item_id": "msg_xxx"}` |
| token 增量 | `{"type": "response.output_text.delta", "delta": "...", "item_id": "msg_xxx"}` |
| 结束 | `{"type": "response.completed", "item_id": "msg_xxx"}` 后再 `data: [DONE]` |

## 关键实现细节

- 两个端点都过 `thread_lock(thread_id)` 串行化同一会话（checkpointer 是 last-write-wins，并发会静默覆盖历史）
- 两个端点都过滤 `langgraph_node != "agent"` 的事件——summarize 节点里也会调模型，不过滤的话摘要正文会漏进聊天框
- Nginx 关缓冲：`X-Accel-Buffering: no`，否则 token 会被攒着发
- 执行时断连（`InterfaceError` / `OperationalError`）自动 `runtime.recover()` 重建后重试一次

## 加新端点的 checklist

1. 从 `chat.py` 或 `responses.py` 复制骨架（`_get_graph` / `_sse` / `thread_lock` / 断连重试）
2. 只改协议层，**不要动 graph.ainvoke 的调用方式**
3. 如果需要鉴权 + 配额，参考 `/assistant/chat` 的 `require_user_id` + `get_quota`
4. 响应 schema 放 `app/schemas/`
