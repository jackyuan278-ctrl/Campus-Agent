"""local_knowledge skill 的 LangGraph tool：校园知识库问答。

复用 `app.rag.retriever.retrieve` — 项目已有的向量检索（稠密 + BM25 + RRF 融合）。
这其实是对现有校园检索能力的一层「面向 LLM 对话」的包装：
- activity.py 里的 search_campus_activities 有时间窗过滤和「活动没结束」硬过滤
- 本工具是更宽松的通用检索：任何 query 都查 top_k 条，不限时间、不限来源
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool

from app.agent.skills import SkillRegistry
from app.rag.retriever import retrieve

logger = logging.getLogger(__name__)


# 摘要截断：和 activity.py 里一致，避免返回太长占 prompt
_SUMMARY_CHARS = 300


def _trim(doc: dict[str, Any]) -> dict[str, Any]:
    """只保留模型对话真正需要的字段。"""
    return {
        "source": doc.get("source"),  # activity / post / resource
        "id": doc.get("id"),
        "title": doc.get("title"),
        "summary": str(doc.get("content") or "")[:_SUMMARY_CHARS],
        "extra": doc.get("extra"),
        "score": doc.get("score"),
    }


@tool
async def local_knowledge_query(query: str, top_k: int = 5) -> str:
    """从校园知识库检索相关内容（活动、帖子、学习资料等）。

    当用户问校园相关问题，如「XX 课怎么选」「图书馆在哪」「有没有编程社团」时使用。

    Args:
        query: 自然语言问题或关键词，如「高数怎么学」「考研自习室」。
        top_k: 返回候选条数，默认 5，最多 10。
    """
    top_k = max(1, min(top_k, 10))

    try:
        hits = await retrieve(query, top_k=top_k)
        results = [_trim(doc) for doc in hits]

        logger.info(
            "local_knowledge_query：query=%s 命中 %d 条（最高分 %.4f）",
            query[:30],
            len(results),
            results[0]["score"] if results else 0,
        )

        return json.dumps(
            {
                "query": query,
                "hits_count": len(results),
                "results": results,
                "tip": (
                    "检索结果按相关性排序，分数越高越匹配。"
                    "如果所有结果 score 都低于 0.3，可能是知识库中没有相关内容。"
                ),
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        logger.exception("local_knowledge_query 检索异常 query=%s", query[:30])
        return json.dumps(
            {"error": f"知识库检索失败：{exc}", "query": query},
            ensure_ascii=False,
        )


# ── SkillRegistry 注册 ────────────────────────────────────────────

LOCAL_KNOWLEDGE_TOOLS = [local_knowledge_query]

LOCAL_KNOWLEDGE_PROMPT = """你是校园知识库助手，擅长帮用户从校园知识库里找内容。

可用工具：
1. local_knowledge_query — 从校园知识库（活动/帖子/学习资料）检索相关内容

使用建议：
- 用户问题模糊时，先调用一次看看知识库有没有相关内容
- 如果返回所有结果 score 都很低（< 0.3），礼貌告知「知识库中暂未收录相关内容」，不要编造
- 可以根据返回的 source 字段判断结果类型（activity 活动 / post 帖子 / resource 学习资料）
"""


def register(r: SkillRegistry) -> None:
    """把校园知识库工具注册进 SkillRegistry。

    - campus（校园问题）档：主档位，给知识库工具 + 校园人设 prompt
    - chat（智能问答）档：兜底，也附带校园知识库能力
    """
    r.register_tools("campus", LOCAL_KNOWLEDGE_TOOLS)
    r.register_prompt("campus", LOCAL_KNOWLEDGE_PROMPT)
    r.register_tools("chat", LOCAL_KNOWLEDGE_TOOLS)
