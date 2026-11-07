"""RAG 检索：查询 → 向量化 → 余弦 top-k。

索引在首次使用时懒加载并常驻内存。因为增量索引可能是**另一个进程**在改文件
（`python -m app.rag.ingest`、MQ 消费者），所以每次取索引都 stat 一下文件
mtime：变了就重新读盘。stat 是微秒级开销，比「改完必须重启服务」划算。
"""
import logging
from collections.abc import Callable

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


async def retrieve(
    query: str,
    top_k: int = 5,
    min_score: float = 0.0,
    where: Callable[[dict], bool] | None = None,
) -> list[dict]:
    """检索 top_k 条最相关文档。

    Args:
        min_score: 相似度下限，默认 0（不过滤）。等看过真实分数分布再调。
        where: 候选过滤谓词（结构化条件，如「活动没结束」「本周内」）。
            **在排序截断之前生效**，这一点很关键：先取 top 15 再过滤的话，
            「这周有什么活动」很可能一条不剩——不是真的没有，是名额被别的活动占满了。
            给了谓词就退化成对全量打分（579 条，一次矩阵乘），代价可以忽略。
    """
    index = get_index()
    if index is None or not index.docs:
        logger.warning("向量索引为空，跳过检索（先跑 python -m app.rag.ingest）")
        return []

    vector = await embed_query(query)
    # 有谓词：全量打分后过滤，避免"过滤完没货"；无谓词：老路径，多取一些去重后截断
    limit = len(index.docs) if where is not None else top_k * 3
    hits = index.search(vector, top_k=limit)

    results: list[dict] = []
    seen: set[str] = set()
    for doc, score in hits:
        if where is not None and not where(doc):
            continue
        # 用 id 而不是 title：标题可能撞车（造数数据里大量重复标题），
        # 按 title 去重会把不同的活动当成同一条吃掉
        key = f"{doc.get('source')}:{doc.get('id')}"
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
    elif where is not None:
        logger.info("检索有谓词但零命中（候选 %d 条全部被过滤）", len(index.docs))
    return results
