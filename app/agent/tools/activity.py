"""活动 / 社区检索工具：把 RAG 检索包成模型可调用的 function calling 工具。

底层仍是 `app.rag.retriever.retrieve`（numpy 余弦 + 去重），这里只是加一层
「工具说明书」，让模型自己决定什么时候搜。
"""
from typing import Any

from langchain_core.tools import tool

from app.rag.retriever import retrieve


@tool
async def search_campus_activities(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """检索校园活动与社区帖子（向量检索）。用户问有什么活动、讲座、比赛、社团招新、帖子讨论时使用。"""
    return await retrieve(query, top_k=top_k)


CAMPUS_SEARCH_TOOLS = [search_campus_activities]
