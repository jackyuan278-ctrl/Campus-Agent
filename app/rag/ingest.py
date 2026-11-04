"""RAG 入库：从 campus-hub 的活动/问答表取文本 → 向量化 → 建索引。

只读！不写任何业务表。用法：
    uv run python -m app.rag.ingest                 # 每张表取最近 200 条
    uv run python -m app.rag.ingest --limit 500     # 调大
    uv run python -m app.rag.ingest --limit 0       # 全量（各 1000 条）
"""
import argparse
import asyncio
import logging

from sqlalchemy import text

from app.database import dispose_all, get_activity_engine, get_community_engine
from app.rag.embedding import aclose as close_embedding
from app.rag.embedding import embed_texts
from app.rag.index import VectorIndex

logger = logging.getLogger(__name__)


async def _load_activities(limit: int) -> list[dict]:
    """取活动，拼成可向量化的文本。"""
    sql = (
        "SELECT id, title, description, location, start_time "
        "FROM tb_activity WHERE status = 1 ORDER BY id DESC"
    )
    if limit > 0:
        sql += " LIMIT :limit"

    engine = get_activity_engine()
    async with engine.connect() as conn:
        result = await conn.execute(text(sql), {"limit": limit} if limit > 0 else {})
        rows = result.mappings().all()

    docs = []
    for row in rows:
        description = (row["description"] or "")[:500]
        extra = f"地点：{row['location'] or '未填写'}｜开始时间：{row['start_time']}"
        docs.append({
            "source": "activity",
            "id": row["id"],
            "title": row["title"] or "",
            "content": description,
            "extra": extra,
            "text": f"活动：{row['title']}。地点：{row['location'] or ''}。{description}",
        })
    logger.info("读取活动 %d 条", len(docs))
    return docs


async def _load_questions(limit: int) -> list[dict]:
    """取社区问答。"""
    sql = (
        "SELECT id, title, content, answer_count, create_time "
        "FROM tb_question WHERE status = 1 ORDER BY id DESC"
    )
    if limit > 0:
        sql += " LIMIT :limit"

    engine = get_community_engine()
    async with engine.connect() as conn:
        result = await conn.execute(text(sql), {"limit": limit} if limit > 0 else {})
        rows = result.mappings().all()

    docs = []
    for row in rows:
        content = (row["content"] or "")[:500]
        extra = f"回答数：{row['answer_count']}｜发帖时间：{row['create_time']}"
        docs.append({
            "source": "question",
            "id": row["id"],
            "title": row["title"] or "",
            "content": content,
            "extra": extra,
            "text": f"问答：{row['title']}。{content}",
        })
    logger.info("读取问答 %d 条", len(docs))
    return docs


async def build_index(limit_per_source: int = 200) -> VectorIndex:
    """全量重建索引（向量化 → 归一化 → 落盘）。"""
    docs = await _load_activities(limit_per_source) + await _load_questions(limit_per_source)
    if not docs:
        raise RuntimeError("没有取到任何数据，检查表是否为空")

    logger.info("开始向量化 %d 条文档…", len(docs))
    vectors = await embed_texts([doc["text"] for doc in docs])
    index = VectorIndex.build(vectors, docs)
    index.save()
    return index


async def _main(limit: int) -> None:
    try:
        index = await build_index(limit)
        print(f"✅ 索引构建完成：{len(index.docs)} 条，维度 {index.dim}")
    finally:
        await close_embedding()
        await dispose_all()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="构建 RAG 向量索引")
    parser.add_argument("--limit", type=int, default=200, help="每张表取多少条，0 表示全量")
    args = parser.parse_args()
    asyncio.run(_main(args.limit))
