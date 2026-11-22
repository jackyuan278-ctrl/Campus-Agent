"""literature skill 的 LangGraph tools：论文搜索、详情、引用图。

真实数据源：Semantic Scholar API（https://api.semanticscholar.org）。
免费额度：匿名 1 req/s，有 key 1 req/s（key 能提高稳定性）。
key 从 mcp_cfg.SEMANTIC_SCHOLAR_API_KEY 读取。
"""

from __future__ import annotations

import json
import logging
import urllib.parse
from typing import Any

import httpx
from langchain_core.tools import tool

from app.agent.skills import SkillRegistry
from mcp import config as mcp_cfg

logger = logging.getLogger(__name__)

_BASE_URL = "https://api.semanticscholar.org/graph/v1"


# ── 通用 helper ────────────────────────────────────────────────────


def _s2_headers() -> dict[str, str]:
    """Semantic Scholar API key 可选：有就带上，没有也能用（限流更严）。"""
    if mcp_cfg.SEMANTIC_SCHOLAR_API_KEY:
        return {"x-api-key": mcp_cfg.SEMANTIC_SCHOLAR_API_KEY}
    return {}


async def _s2_get(path: str, params: dict[str, Any]) -> dict[str, Any] | None:
    """封装 Semantic Scholar GET 请求，失败返回 None（让上层决定是否降级 mock）。"""
    url = f"{_BASE_URL}{path}"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, params=params, headers=_s2_headers())
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.warning("Semantic Scholar 请求失败 path=%s params=%s: %s", path, params, exc)
        return None


# ── 1. 论文搜索 ─────────────────────────────────────────────────────


@tool
async def search_papers(query: str, limit: int = 5) -> str:
    """按关键词搜索学术论文。数据来自 Semantic Scholar（免费学术搜索引擎）。

    Args:
        query: 搜索关键词，如 "attention is all you need" "transformer agent"。
        limit: 返回条数，1~20。默认 5。
    """
    limit = max(1, min(limit, 20))

    # 先尝试真实 API
    if not mcp_cfg.is_mock():
        fields = "title,authors,year,venue,citationCount,abstract,externalIds"
        data = await _s2_get(
            "/paper/search",
            {"query": query, "limit": limit, "fields": fields},
        )
        if data and (data.get("data") or []):
            papers = []
            for p in data["data"]:
                papers.append(
                    {
                        "paper_id": p.get("paperId"),
                        "title": p.get("title"),
                        "authors": [a.get("name") for a in (p.get("authors") or [])],
                        "year": p.get("year"),
                        "venue": p.get("venue"),
                        "citation_count": p.get("citationCount"),
                        "abstract": (p.get("abstract") or "")[:300],
                    }
                )
            return json.dumps(
                {"query": query, "data_source": "semantic_scholar", "papers": papers},
                ensure_ascii=False,
            )
        logger.warning("Semantic Scholar 搜索无结果，降级 mock")

    # mock 兜底
    return json.dumps(
        {
            "query": query,
            "data_source": "mock",
            "papers": mcp_cfg.MOCK_PAPERS[:limit],
            "note": "⚠️ 当前为 mock 数据。真实搜索需要 Semantic Scholar API 可用。",
        },
        ensure_ascii=False,
    )


# ── 2. 论文详情 ─────────────────────────────────────────────────────


@tool
async def get_paper_detail(paper_id: str) -> str:
    """按 paper_id 查询论文详情（含摘要、引用数、外部 ID）。

    Args:
        paper_id: Semantic Scholar paper ID（如 "649def34f8be52c8b66281af98ae884c09aef38b"）。
    """
    # 先尝试真实 API
    if not mcp_cfg.is_mock():
        fields = (
            "title,authors,year,venue,citationCount,abstract,externalIds,"
            "url,publicationDate,fieldsOfStudy"
        )
        data = await _s2_get(f"/paper/{paper_id}", {"fields": fields})
        if data and data.get("title"):
            return json.dumps(
                {
                    "paper_id": paper_id,
                    "data_source": "semantic_scholar",
                    "title": data.get("title"),
                    "authors": [a.get("name") for a in (data.get("authors") or [])],
                    "year": data.get("year"),
                    "venue": data.get("venue"),
                    "citation_count": data.get("citationCount"),
                    "abstract": data.get("abstract"),
                    "url": data.get("url"),
                    "external_ids": data.get("externalIds"),
                    "fields_of_study": data.get("fieldsOfStudy"),
                },
                ensure_ascii=False,
            )
        logger.warning("Semantic Scholar 论文详情无结果 paper_id=%s，降级 mock", paper_id)

    # mock 兜底
    mock_paper = next(
        (p for p in mcp_cfg.MOCK_PAPERS if p.get("paper_id") == paper_id),
        mcp_cfg.MOCK_PAPERS[0] if mcp_cfg.MOCK_PAPERS else {},
    )
    return json.dumps(
        {
            "paper_id": paper_id,
            "data_source": "mock",
            "note": "⚠️ 当前为 mock 数据。",
            **mock_paper,
        },
        ensure_ascii=False,
    )


# ── 3. 引用关系图 ───────────────────────────────────────────────────


@tool
async def get_citation_graph(paper_id: str, depth: int = 1) -> str:
    """查询论文的引用关系（被引用 / 引用了谁）。

    Args:
        paper_id: Semantic Scholar paper ID。
        depth: 引用深度。1 表示直接引用；仅支持 1。
    """
    depth = max(1, min(depth, 2))  # Semantic Scholar 深层引用较重，限制到 2

    if not mcp_cfg.is_mock():
        fields = "title,authors,year,citationCount"
        # 被谁引用（citations）和引用了谁（references）
        citations = await _s2_get(f"/paper/{paper_id}/citations", {"fields": fields, "limit": 10})
        references = await _s2_get(f"/paper/{paper_id}/references", {"fields": fields, "limit": 10})

        if citations or references:

            def _fmt(items: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
                out = []
                for item in items or []:
                    p = item.get("citedPaper") or item.get("citingPaper") or {}
                    out.append(
                        {
                            "paper_id": p.get("paperId"),
                            "title": p.get("title"),
                            "authors": [a.get("name") for a in (p.get("authors") or [])],
                            "year": p.get("year"),
                        }
                    )
                return out

            return json.dumps(
                {
                    "paper_id": paper_id,
                    "data_source": "semantic_scholar",
                    "cited_by_count": len((citations or {}).get("data") or []),
                    "references_count": len((references or {}).get("data") or []),
                    "cited_by": _fmt((citations or {}).get("data")),
                    "references": _fmt((references or {}).get("data")),
                },
                ensure_ascii=False,
            )
        logger.warning("Semantic Scholar 引用图无结果 paper_id=%s，降级 mock", paper_id)

    # mock 兜底
    return json.dumps(
        {
            "paper_id": paper_id,
            "data_source": "mock",
            "note": "⚠️ 当前为 mock 数据。",
            "cited_by": [
                {
                    "paper_id": "mock_cite_001",
                    "title": f"Mock 引用了 {paper_id}",
                    "authors": ["Mock Author"],
                    "year": 2024,
                },
            ],
            "references": [
                {
                    "paper_id": "mock_ref_001",
                    "title": "Mock 被引用论文",
                    "authors": ["Mock Author"],
                    "year": 2020,
                },
            ],
        },
        ensure_ascii=False,
    )


# ── SkillRegistry 注册 ────────────────────────────────────────────

LITERATURE_TOOLS = [search_papers, get_paper_detail, get_citation_graph]

LITERATURE_PROMPT = """你是学术文献查询助手，擅长帮用户搜索论文、查详情、看引用关系。

可用工具：
1. search_papers — 关键词搜索论文
2. get_paper_detail — 查某篇论文的完整信息
3. get_citation_graph — 看引用关系图

使用建议：
- 用户问「有没有关于 XX 的论文」→ 直接 search_papers
- 用户给了具体论文 ID → get_paper_detail
- 用户想了解某领域的脉络 → 先用 search_papers 找代表作，再 get_citation_graph
"""


def register(r: SkillRegistry) -> None:
    """把文献查询工具包注册进 SkillRegistry。

    - research（学术调研）档：主档位，给论文工具 + 学术人设 prompt
    - chat（智能问答）档：兜底，也附带论文检索能力
    """
    r.register_tools("research", LITERATURE_TOOLS)
    r.register_prompt("research", LITERATURE_PROMPT)
    r.register_tools("chat", LITERATURE_TOOLS)
