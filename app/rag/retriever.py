"""RAG 检索：查询 → 向量化 → 余弦 top-k。

索引在首次使用时懒加载并常驻内存；重建索引后调 reload_index() 生效。
"""
import logging

from app.rag.embedding import embed_query
from app.rag.index import VectorIndex

logger = logging.getLogger(__name__)

_index: VectorIndex | None = None
_loaded = False


def get_index() -> VectorIndex | None:
    """懒加载索引（进程内只读一次盘）。"""
    global _index, _loaded
    if not _loaded:
        _index = VectorIndex.load()
        _loaded = True
    return _index


def reload_index() -> None:
    """丢弃缓存，下次检索重新读盘（重新 ingest 之后调用）。"""
    global _index, _loaded
    _index = None
    _loaded = False
    logger.info("索引缓存已失效，下次检索重新加载")


async def retrieve(query: str, top_k: int = 5, min_score: float = 0.0) -> list[dict]:
    """检索 top_k 条最相关文档。

    Args:
        min_score: 相似度下限，默认 0（不过滤）。等看过真实分数分布再调。
    """
    index = get_index()
    if index is None or not index.docs:
        logger.warning("向量索引为空，跳过检索（先跑 python -m app.rag.ingest）")
        return []

    vector = await embed_query(query)
    hits = index.search(vector, top_k=top_k * 3)  # 多取一些，去重后再截断

    results: list[dict] = []
    seen: set[str] = set()
    for doc, score in hits:
        # 种子数据里同名文档会出现多条，按「来源+标题」去重，避免 prompt 里全是重复内容
        key = f"{doc.get('source')}:{doc.get('title')}"
        if key in seen:
            continue
        seen.add(key)
        if score < min_score:
            continue
        results.append({**doc, "score": round(score, 4)})
        if len(results) >= top_k:
            break

    if results:
        logger.info(
            "检索命中 %d 条，最高分 %.4f（%s）",
            len(results), results[0]["score"], results[0].get("title", "")[:20],
        )
    return results
