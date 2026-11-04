"""对话相关接口：SSE 流式对话 + 历史消息管理。

两种返回形态（前端 campus-web 都支持）：
- 默认 SSE：Content-Type: text/event-stream，逐 token 推 `data: {"delta": "..."}`
- `?stream=false`：一次性返回 JSON，结构为 {"data": {"content": ..., "thread_id": ..., "intent": ...}}
"""
import json
import logging
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from app.common.exceptions import QuotaExceededError
from app.common.security import require_user_id
from app.memory.ownership import ensure_owner
from app.memory.quota import get_quota
from app.memory.store import MemoryStore
from app.schemas.chat import ChatRequest

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
    return (
        type(exc).__name__ in ("InterfaceError", "OperationalError")
        or "Not connected" in str(exc)
    )


def _sse(payload: dict[str, Any] | str) -> str:
    data = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return f"data: {data}\n\n"


@router.post("/chat", summary="对话（SSE 流式，?stream=false 返回 JSON）")
async def chat(
    request: ChatRequest,
    http_request: Request,
    stream: Annotated[bool, Query(description="false 时一次性返回 JSON")] = True,
):
    """执行一次对话。附件需先经 /assistant/upload 上传，这里只收 OSS URL。"""
    # 鉴权 + 会话归属：登录了才能聊，且这个 thread_id 必须是自己的
    user_id = require_user_id(http_request)
    user_key = f"user:{user_id}"
    await ensure_owner(request.thread_id, user_key)

    # 配额预检：超限直接挡掉，一次模型都不调
    quota = await get_quota(user_key)
    if quota.exceeded:
        logger.warning("配额用尽：%s 已用 %d / %d", user_key, quota.used, quota.limit)
        raise QuotaExceededError(
            f"今日额度已用完（已用 {quota.used:,} / 上限 {quota.limit:,} tokens），明天再来"
        )

    graph = await _get_graph(http_request)
    inputs = {
        "query": request.message,
        "thread_id": request.thread_id,
        "mode": request.mode,
        "user_key": user_key,
        "attachments": [item.model_dump() for item in request.attachments],
    }
    config = {"configurable": {"thread_id": request.thread_id}}

    # 非流式：直接跑完整张图，一次性返回（结构与旧版一致）
    if not stream:
        try:
            result = await graph.ainvoke(inputs, config)
        except Exception as exc:
            if not _is_conn_error(exc):
                raise
            # ping 通过但真正执行时断开：强制重建图再重试一次
            logger.warning("执行时连接异常，重建后重试一次：%s", exc)
            graph = await http_request.app.state.agent_runtime.recover()
            result = await graph.ainvoke(inputs, config)
        return {
            "data": {
                "content": result.get("answer", ""),
                "thread_id": request.thread_id,
                "intent": result.get("intent", ""),
            }
        }

    async def event_stream():
        # SSE 注释帧：浏览器立刻收到首字节，连接状态可见（前端解析器会跳过非 data: 行）
        yield ": connected\n\n"
        try:
            async for event in graph.astream_events(inputs, config, version="v2"):
                if event["event"] != "on_chat_model_stream":
                    continue
                # 只转发 agent 节点的模型输出。摘要在 summarize 节点里也会调模型，
                # 不过滤的话摘要正文会漏进用户的聊天框。
                if (event.get("metadata") or {}).get("langgraph_node") != "agent":
                    continue
                text = getattr(event["data"].get("chunk"), "content", "") or ""
                if text:
                    yield _sse({"delta": text})
        except Exception as exc:  # 上游失败也要让前端看得见，否则只能看到"转圈然后没反应"
            logger.exception("对话流式输出失败")
            # 前端只认 delta 字段，所以把错误也放进 delta，保证能显示在聊天框里
            yield _sse({"delta": f"\n\n（服务异常，请稍后重试）{exc}", "error": str(exc)})
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 关掉 Nginx 缓冲，否则 token 会被攒着发
        },
    )


@router.get("/chat/messages", summary="获取历史消息")
async def get_chat_messages(
    thread_id: Annotated[str, Query(min_length=1, max_length=64)],
    http_request: Request,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
):
    """按时间正序返回指定会话的消息（默认最近 100 条）。只能看自己的会话。"""
    user_key = f"user:{require_user_id(http_request)}"
    await ensure_owner(thread_id, user_key)
    messages = await MemoryStore().get_history(thread_id, limit=limit)
    return {"data": messages}


@router.delete("/chat/messages", summary="清空历史消息")
async def clear_chat_messages(
    thread_id: Annotated[str, Query(min_length=1, max_length=64)],
    http_request: Request,
):
    """清空指定会话：业务记录 + 图的 checkpoint 一起删，避免下次接话接上旧上下文。"""
    user_key = f"user:{require_user_id(http_request)}"
    await ensure_owner(thread_id, user_key)
    await MemoryStore().clear(thread_id)

    saver = None
    runtime = getattr(http_request.app.state, "agent_runtime", None)
    if runtime is not None:
        saver = runtime.saver
    if saver is not None:
        try:
            await saver.adelete_thread(thread_id)
        except Exception:  # checkpoint 清理失败不影响业务删除，记日志即可
            logger.warning("清理 checkpoint 失败：thread_id=%s", thread_id, exc_info=True)

    return {"data": {"thread_id": thread_id, "cleared": True}}
