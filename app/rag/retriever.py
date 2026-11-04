"""RAG 检索：查询 → 向量化 → 余弦 top-k。

索引在首次使用时懒加载并常驻内存。因为增量索引可能是**另一个进程**在改文件
（`python -m app.rag.ingest`、MQ 消费者），所以每次取索引都 stat 一下文件
mtime：变了就重新读盘。stat 是微秒级开销，比「改完必须重启服务」划算。
"""
import logging

from app.rag.embedding import embed_query
from app.rag.index import INDEX_DIR, VECTORS_FILE, VectorIndex

logger = logging.getLogger(__name__)

_index: VectorIndex | None = None
_loaded = False
_loaded_mtime: float = 0.0


def _index_mtime() -> float:
    """索引文件的修改时间；文件不存在时返回 0。"""
    try:
        return (INDEX_DIR / VECTORS_FILE).stat().st_mtime
    except OSError:
        return 0.0


def get_index() -> VectorIndex | None:
    """取索引；文件被别人改过就自动重新加载。"""
    global _index, _loaded, _loaded_mtime
    mtime = _index_mtime()
    if not _loaded or mtime != _loaded_mtime:
        _index = VectorIndex.load()
        _loaded = True
        _loaded_mtime = mtime
    return _index


def reload_index() -> None:
    """丢弃缓存，下次检索重新读盘（同进程重建索引后调用）。"""
    global _index, _loaded, _loaded_mtime
    _index = None
    _loaded = False
    _loaded_mtime = 0.0
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
