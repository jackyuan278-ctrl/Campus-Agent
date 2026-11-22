"""Responses API 兼容层端点（/v1/responses）。

给 Codex CLI / Cursor 等外部客户端用——协议对齐 OpenAI Responses API：
  - SSE 事件帧：data: {"type":"response.output_text.delta","delta":"..."}
  - 结束帧：data: {"type":"response.completed"} 后再 data: [DONE]

内部复用同一套 LangGraph 状态图（graph.ainvoke / graph.astream_events），
但和 /assistant/chat 完全解耦——那边的 request/response/SSE 一个字节不动。
"""

import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.agent.locks import thread_lock
from app.common.exceptions import AppError
from app.schemas.responses import ResponsesFinalResponse, ResponsesRequest

router = APIRouter()
logger = logging.getLogger(__name__)


async def _get_graph(request: Request):
    """取运行时里的状态图（内部会 ping 一次，连接死了自动重建）。"""
    runtime = getattr(request.app.state, "agent_runtime", None)
    if runtime is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent 运行时未初始化，请检查启动日志",
        )
    return await runtime.get_graph()


def _is_conn_error(exc: Exception) -> bool:
    """判断是不是「连接断了」这类可重试异常。"""
    return type(exc).__name__ in ("InterfaceError", "OperationalError") or "Not connected" in str(
        exc
    )


def _sse(payload: dict[str, Any] | str) -> str:
    """把 dict / 字符串打包成 SSE 帧。"""
    data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"data: {data}\n\n"


def _next_item_id() -> str:
    """生成一个 response.output_text 的 item_id（Codex CLI / Cursor 会用到）。"""
    return f"msg_{uuid.uuid4().hex[:24]}"


@router.post(
    "/responses",
    summary="Responses API 兼容层（SSE / 一次性 JSON）",
    response_model=None,
)
async def responses(req: ResponsesRequest, http_request: Request):
    """Responses API 端点。内部走同一套 graph.ainvoke，协议层独立。"""
    thread_id = req.thread_id or uuid.uuid4().hex
    item_id = _next_item_id()

    graph = await _get_graph(http_request)
    inputs = {
        "query": req.input,
        "thread_id": thread_id,
        "mode": req.mode,
        # responses 端点不做用户隔离，user_key 留空让 persist 跳过配额写入。
        "user_key": "",
        "attachments": [],
    }
    config = {"configurable": {"thread_id": thread_id}}

    if not req.stream:
        async with thread_lock(thread_id):
            try:
                result = await graph.ainvoke(inputs, config)
            except Exception as exc:
                if not _is_conn_error(exc):
                    raise
                logger.warning("responses 非流式：连接异常，重建后重试一次：%s", exc)
                graph = await http_request.app.state.agent_runtime.recover()
                result = await graph.ainvoke(inputs, config)
        return ResponsesFinalResponse(
            thread_id=thread_id,
            output_text=result.get("answer", ""),
            mode=req.mode,
        ).model_dump()

    # --- 流式 SSE ---
    async def event_stream():
        yield _sse({"type": "response.created", "item_id": item_id})

        async with thread_lock(thread_id):
            try:
                async for event in graph.astream_events(inputs, config, version="v2"):
                    evt = event.get("event")
                    node = (event.get("metadata") or {}).get("langgraph_node")

                    # --- 工具调用事件（ToolNode 执行时发出）---
                    if evt == "on_tool_start" and node == "tools":
                        yield _sse({
                            "type": "response.function_call",
                            "name": event.get("name", "unknown"),
                            "arguments": event.get("data", {}).get("input", {}),
                        })
                    elif evt == "on_tool_end" and node == "tools":
                        yield _sse({
                            "type": "response.function_call_output",
                            "name": event.get("name", "unknown"),
                            "output": str(event.get("data", {}).get("output", ""))[:2000],
                        })
                    # --- 模型文本增量（agent 节点）---
                    elif evt == "on_chat_model_stream":
                        if node != "agent":
                            continue
                        text = getattr(event["data"].get("chunk"), "content", "") or ""
                        if text:
                            yield _sse(
                                {
                                    "type": "response.output_text.delta",
                                    "delta": text,
                                    "item_id": item_id,
                                }
                            )
            except AppError as exc:
                logger.warning("responses 流式失败（业务异常 %s）：%s", exc.code, exc.message)
                yield _sse(
                    {
                        "type": "response.output_text.delta",
                        "delta": f"\n\n（{exc.message}）",
                        "item_id": item_id,
                    }
                )
            except Exception:
                logger.exception("responses 流式输出失败")
                yield _sse(
                    {
                        "type": "response.output_text.delta",
                        "delta": "\n\n（服务异常，请稍后重试）",
                        "item_id": item_id,
                    }
                )
            finally:
                yield _sse({"type": "response.completed", "item_id": item_id})
                yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
