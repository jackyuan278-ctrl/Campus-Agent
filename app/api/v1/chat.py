"""对话相关接口：流式对话 + 历史消息管理。

现状：脚手架阶段，三个接口一律返回 501。
这样「未实现」是显式契约，不会像返回 null 那样被前端当成成功。
"""
from fastapi import APIRouter, HTTPException, status

from app.schemas.chat import ChatRequest

router = APIRouter()

_NOT_READY = "接口尚未实现（脚手架阶段）"

# 把 501 声明进 OpenAPI：装饰器上的 status_code 决定默认响应码，
# responses 补一句说明，前端看文档就不会按 200 去解析
_NOT_READY_RESPONSES = {
    status.HTTP_501_NOT_IMPLEMENTED: {"description": _NOT_READY},
}


@router.post(
    "/chat",
    summary="对话（SSE 流式返回，待实现）",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    responses=_NOT_READY_RESPONSES,
)
async def chat(request: ChatRequest):
    """对话（SSE 流式返回）"""
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=_NOT_READY)


@router.get(
    "/chat/messages",
    summary="获取历史消息（待实现）",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    responses=_NOT_READY_RESPONSES,
)
async def get_chat_messages(thread_id: str):
    """获取历史消息"""
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=_NOT_READY)


@router.delete(
    "/chat/messages",
    summary="清空历史消息（待实现）",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    responses=_NOT_READY_RESPONSES,
)
async def clear_chat_messages(thread_id: str):
    """清空历史消息"""
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=_NOT_READY)
