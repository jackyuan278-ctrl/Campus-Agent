"""RAG 入库：从 campus-hub 的活动/问答表取文本 → 向量化 → 建索引。

只读！不写任何业务表。用法：
    uv run python -m app.rag.ingest                 # 每张表取最近 200 条
    uv run python -m app.rag.ingest --limit 500     # 调大
    uv run python -m app.rag.ingest --limit 0       # 全量（各表全取）
    uv run python -m app.rag.ingest --limit 0 --full  # 全量且不复用旧向量

两个关键设计：
1. **按文本指纹去重**。campus-hub 的造数脚本（database/13_seed_community.sql）
   用 ELT() 拼标题正文，1000 条问答实际只有 34 种文本。全量入索引会让这些
   近似重复的向量占满检索候选池（top_k*3 取 15 条，去重后剩 1 条）。
2. **指纹相同就复用旧向量**。内容没变就不该重复花 embedding 调用。
"""
import argparse
import asyncio
import hashlib
import logging
from typing import Any

from sqlalchemy import text

from app.database import dispose_all, get_activity_engine, get_community_engine
from app.rag.embedding import aclose as close_embedding
from app.rag.embedding import embed_texts
from app.rag.index import VectorIndex
from app.rag.write_lock import index_write_lock

logger = logging.getLogger(__name__)


def fingerprint(text: str) -> str:
    """文本指纹：同时用于「去重」和「内容没变就复用向量」。

    取 sha1 前 16 位足够——这是 2^64 空间，5000 条文档撞车概率可以忽略。
    """
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


# 拼 text 的逻辑必须只有一份：全量 ingest 和单条增量（indexer）如果各拼各的，
# 同一条记录会算出不同的向量，索引里就会出现「内容一样但向量不同」的鬼影。
#
# 收录口径：status=1 只代表「已发布」（报名中 / 未开始 / 报名已截止），活动真正
# 开始后定时任务会把它翻成 status=2「进行中」。只收 1 的话，用户在活动页看着一个
# 正在进行中的活动，问助手却答「没找到」——展示层和知识库口径就分叉了。
# 已结束（3）、已取消（4）仍排除：推荐一个已经办完的活动没有意义。
_ACTIVITY_SQL = (
    "SELECT id, title, description, location, start_time, end_time, signup_end_time "
    "FROM tb_activity WHERE status IN (1, 2)"
)
_QUESTION_SQL = (
    "SELECT id, title, content, answer_count, create_time "
    "FROM tb_question WHERE status = 1"
)


def _iso(value: Any) -> str:
    """datetime → 'YYYY-MM-DD HH:MM:SS'（不带时区，语义是北京时间）。

    `str(datetime)` 在带微秒时会给 '...:00.000000'，fromisoformat 两种都能解析，
    但统一成 isoformat 让人读索引文件时不用猜。
    """
    return value.isoformat(sep=" ") if hasattr(value, "isoformat") else str(value or "")


def activity_doc(row: Any) -> dict:
    """一行活动 → 可向量化文档。

    时间字段**单独存三个**，不拼进 `text`：
        start_time / end_time / signup_end_time 是「过滤和排序」用的结构化数据，
        由检索时的谓词直接比较，不需要模型去读、也不需要向量去理解。
        拼进 text 会让 fingerprint 变化 → 356 条活动全部重算向量（一次白花钱），
        而且时间对语义相似度没有贡献。

    `text` 保持原样是刻意的：它是「被向量化的内容」，指纹也就该只盯着它。
    """
    description = (row["description"] or "")[:500]
    return {
        "source": "activity",
        "id": row["id"],
        "title": row["title"] or "",
        "content": description,
        "extra": f"地点：{row['location'] or '未填写'}｜开始时间：{row['start_time']}",
        # ↓ 结构化时间：检索时按区间过滤、按新鲜度排序，都读这三个字段
        "start_time": _iso(row["start_time"]),
        "end_time": _iso(row["end_time"]),
        "signup_end_time": _iso(row["signup_end_time"]),
        "text": f"活动：{row['title']}。地点：{row['location'] or ''}。{description}",
    }


def question_doc(row: Any) -> dict:
    """一行问答 → 可向量化文档。"""
    content = (row["content"] or "")[:500]
    return {
        "source": "question",
        "id": row["id"],
        "title": row["title"] or "",
        "content": content,
        "extra": f"回答数：{row['answer_count']}｜发帖时间：{row['create_time']}",
        "text": f"问答：{row['title']}。{content}",
    }


async def load_activity(activity_id: int) -> dict | None:
    """按 id 读一条活动。已结束（3）/已取消（4）/不存在返回 None —— 调用方据此删向量。"""
    engine = get_activity_engine()
    async with engine.connect() as conn:
        result = await conn.execute(
            text(_ACTIVITY_SQL + " AND id = :id"), {"id": activity_id}
        )
        row = result.mappings().first()
    return None if row is None else activity_doc(row)


async def load_question(question_id: int) -> dict | None:
    """按 id 读一条问答。隐藏（status≠1）或已删返回 None。"""
    engine = get_community_engine()
    async with engine.connect() as conn:
        result = await conn.execute(
            text(_QUESTION_SQL + " AND id = :id"), {"id": question_id}
        )
        row = result.mappings().first()
    return None if row is None else question_doc(row)


async def _load_activities(limit: int) -> list[dict]:
    """取活动（status=1 已发布 + status=2 进行中，口径与 _ACTIVITY_SQL 一致）。"""
    sql = _ACTIVITY_SQL + " ORDER BY id DESC"
    if limit > 0:
        sql += " LIMIT :limit"

    engine = get_activity_engine()
    async with engine.connect() as conn:
        result = await conn.execute(text(sql), {"limit": limit} if limit > 0 else {})
        rows = result.mappings().all()

    docs = [activity_doc(row) for row in rows]
    logger.info("读取活动 %d 条", len(docs))
    return docs


async def _load_questions(limit: int) -> list[dict]:
    """取社区问答（只取正常的 status=1）。"""
    sql = _QUESTION_SQL + " ORDER BY id DESC"
    if limit > 0:
        sql += " LIMIT :limit"

    engine = get_community_engine()
    async with engine.connect() as conn:
        result = await conn.execute(text(sql), {"limit": limit} if limit > 0 else {})
        rows = result.mappings().all()

    docs = [question_doc(row) for row in rows]
    logger.info("读取问答 %d 条", len(docs))
    return docs


async def build_index(limit_per_source: int = 200, incremental: bool = True) -> VectorIndex:
    """重建索引（去重 → 向量化 → 归一化 → 落盘）。

    incremental=True 时，文本指纹没变的文档直接复用旧向量，只为新增/变更的
    文档花 embedding 调用。整体重写文件，所以「下架 / 删除」不用特殊处理
    ——不在这次结果集里的文档，自然不会被写进去。
    """
    raw = await _load_activities(limit_per_source) + await _load_questions(limit_per_source)
    if not raw:
        raise RuntimeError("没有取到任何数据，检查表是否为空")

    # 按文本指纹去重：同一条内容只保留第一次出现的那份（id 最小的先被读到）
    docs: list[dict] = []
    seen: set[str] = set()
    for doc in raw:
        fp = fingerprint(doc["text"])
        if fp in seen:
            continue
        seen.add(fp)
        doc["fingerprint"] = fp
        docs.append(doc)
    if len(docs) != len(raw):
        logger.info("去重：%d 条 → %d 条（丢弃 %d 条同内容重复）", len(raw), len(docs), len(raw) - len(docs))

    # 旧索引按指纹建表：老数据没有 fingerprint 字段，用 text 现算（不花钱）
    old_by_fp: dict[str, int] = {}
    # 从读旧索引到落盘整段加锁：两个写入方（消费者 / 手动 ingest）同时跑时，
    # 各自基于同一份旧索引计算，后写的会盖掉前一个的改动。
    async with index_write_lock():
        old = VectorIndex.load() if incremental else None
        if old is not None:
            for row, doc in enumerate(old.docs):
                old_fp = doc.get("fingerprint") or fingerprint(doc.get("text", ""))
                old_by_fp.setdefault(old_fp, row)  # 同指纹只认第一行

        vectors: list[list[float] | None] = []
        pending: list[tuple[int, str]] = []
        for doc in docs:
            row = old_by_fp.get(doc["fingerprint"])
            if row is not None:
                vectors.append(old.vectors[row].tolist())  # 复用，零成本
            else:
                vectors.append(None)
                pending.append((len(vectors) - 1, doc["text"]))

        reused = len(docs) - len(pending)
        if pending:
            logger.info("需要向量化 %d 条（复用旧向量 %d 条）", len(pending), reused)
            fresh = await embed_texts([t for _, t in pending])
            for (pos, _), vector in zip(pending, fresh):
                vectors[pos] = vector
        else:
            logger.info("全部复用旧向量（%d 条），本次 0 次 embedding 调用", reused)

        index = VectorIndex.build(vectors, docs)
        index.save()
    return index


async def _main(limit: int, full: bool = False) -> None:
    try:
        index = await build_index(limit, incremental=not full)
        print(f"✅ 索引构建完成：{len(index.docs)} 条，维度 {index.dim}")
    finally:
        await close_embedding()
        await dispose_all()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="构建 RAG 向量索引")
    parser.add_argument("--limit", type=int, default=200, help="每张表取多少条，0 表示全量")
    parser.add_argument(
        "--full",
        action="store_true",
        help="不复用旧向量，全部重新向量化（索引疑似损坏时用）",
    )
    args = parser.parse_args()
    asyncio.run(_main(args.limit, full=args.full))
