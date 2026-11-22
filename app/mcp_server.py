"""Campus-Agent MCP Server — 用标准 MCP 协议把工具暴露给 Codex CLI / 任意 MCP 客户端。

启动方式：
    uv run python -m app.mcp_server

Codex CLI 配置（~/.codex/config.toml）：
    [mcp_servers.campus-agent]
    command = "uv"
    args = ["run", "--project", "/path/to/campus-agent", "python", "-m", "app.mcp_server"]

MCP 传输协议：stdio（进程间通信，Codex CLI 拉起子进程用）
"""

from __future__ import annotations

import asyncio
import logging
import sys

# 让 uv run python -m app.mcp_server 时能找到 app 包
import os

os.environ.setdefault("APP_LOG_LEVEL", "INFO")

from mcp.server.fastmcp import FastMCP

from app.agent.tools.secondhand import search_secondhand_items
from app.agent.tools.activity import search_campus_activities, campus_search
from app.agent.tools.web import web_search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mcp")

mcp = FastMCP(
    "campus-agent",
    description="CampusHub 校园助手的 MCP 服务端 — 提供二手查询、校园活动检索、联网搜索、旅游规划等工具",
)


# ── 二手查询 ──────────────────────────────────────────────────────


@mcp.tool()
def search_secondhand(keyword: str, page: int = 1, page_size: int = 10) -> str:
    """查询校园二手市场。按关键词模糊匹配商品名称/描述/分类。只返回在售商品。"""
    result = asyncio.run(
        search_secondhand_items(keyword=keyword, page=page, page_size=page_size)
    )
    return _to_text(result)


# ── 校园活动检索 ──────────────────────────────────────────────────


@mcp.tool()
def search_activities(time_range: str = "最近一周", location: str = "") -> str:
    """检索校园活动（讲座、比赛、展览、社团招新等）。只返回未结束的活动。

    Args:
        time_range: 时间窗，如「今天」「本周」「本月」「最近三天」。
        location: 可选地点关键词，如「图书馆」「操场」。
    """
    result = asyncio.run(
        search_campus_activities(time_range=time_range, location=location or None)
    )
    return _to_text(result)


@mcp.tool()
def campus_search(query: str) -> str:
    """校园知识库搜索：覆盖二手、活动、社区帖子、常见问题。"""
    result = asyncio.run(campus_search(query=query))
    return _to_text(result)


# ── 联网搜索 ──────────────────────────────────────────────────────


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> str:
    """联网搜索（Tavily）。适合查「XX 怎么办」「最新 XX 资讯」这类校园库覆盖不到的问题。"""
    result = asyncio.run(web_search(query=query, max_results=max_results))
    return _to_text(result)


# ── 辅助 ──────────────────────────────────────────────────────────


def _to_text(result: object) -> str:
    """把工具返回值（通常是 list[dict]）格式化成模型好读的纯文本。"""
    if not result:
        return "（没有找到结果）"
    if isinstance(result, str):
        return result
    import json

    try:
        return json.dumps(result, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return str(result)


# ── 启动 ──────────────────────────────────────────────────────────


def main() -> None:
    logger.info("Campus-Agent MCP Server starting (stdio transport)...")
    try:
        mcp.run(transport="stdio")
    except Exception as exc:
        logger.exception("MCP Server crashed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
