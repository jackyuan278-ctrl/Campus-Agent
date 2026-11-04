"""联网搜索工具（Tavily）。

为什么需要它：校园库（二手/活动/社区）答不了「今天有什么新闻」「XX 怎么办」这类问题，
本地检索再好也覆盖不到外部世界。Tavily 是面向 LLM 的搜索 API，直接返回正文摘要，
比抓网页再解析省事得多。

使用边界：只在「随便聊聊」档位（mode=casual）下调用，别的档位不搜——
校园问题有本地库，联网搜反而会引入噪音和延迟。
"""
import logging
from typing import Any

from langchain_core.tools import tool
from tavily import AsyncTavilyClient

from app.config import get_settings

logger = logging.getLogger(__name__)

# 懒加载：import 时不建客户端，也不读 .env
_client: AsyncTavilyClient | None = None


def _get_client() -> AsyncTavilyClient:
    global _client
    if _client is None:
        _client = AsyncTavilyClient(api_key=get_settings().tavily_api_key)
    return _client


async def search_web(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """联网搜索，返回 [{title, url, content, score}]。

    失败不抛异常：联网属于「有更好」的能力，搜不到就让模型按自身知识回答，
    不该因此让整轮对话报错。
    """
    settings = get_settings()
    if not settings.tavily_api_key:
        logger.warning("未配置 TAVILY_API_KEY，跳过联网搜索")
        return []

    try:
        resp = await _get_client().search(
            query=query,
            max_results=max_results,
            search_depth="basic",  # basic 足够；advanced 更贵更慢
            include_answer=False,  # 让我们的模型自己组织语言，不要 Tavily 的现成答案
        )
    except Exception:
        logger.exception("Tavily 搜索失败：%s", query[:30])
        return []

    results = resp.get("results") or []
    logger.info("Tavily 命中 %d 条：%s", len(results), query[:30])
    return [
        {
            "title": item.get("title") or "",
            "url": item.get("url") or "",
            "content": item.get("content") or "",
            "score": round(float(item.get("score") or 0.0), 4),
            "source": "web",
        }
        for item in results
    ]


# ===== LangChain 工具封装 =====
# docstring 会作为工具说明送给模型，所以写清「什么时候该用」。


@tool
async def web_search_tool(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """联网搜索外部信息（新闻、攻略、实时数据、你不确定的事实）。校园本地库查不到时使用。"""
    return await search_web(query, max_results=max_results)


WEB_TOOLS = [web_search_tool]
