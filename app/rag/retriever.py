"""RAG 检索：稠密 + BM25 两路召回 → RRF 融合 → 轻量重排 → 截断。

为什么两路：
    稠密擅长「意思相近」，BM25 擅长「字面对得上」（专名、编号、缩写）。
    两者失败模式不重叠，融合才有意义。详见 app/rag/bm25.py 与 fuse.py。

为什么重排不用模型：
    首字延迟已经 2.5~3s（模型强制思考），再加一次 rerank 调用（+150~300ms 起）
    是拿用户体感换 MRR 的小数点。而「标题命中加权 + 新鲜度」这类确定性规则
    能修掉大部分排序问题，零延迟零成本。真需要模型重排时再换，接口不变。

索引在首次使用时懒加载并常驻内存。因为增量索引可能是**另一个进程**在改文件
（`python -m app.rag.ingest`、MQ 消费者），所以每次取索引都 stat 一下文件
mtime：变了就重新读盘。stat 是微秒级开销，比「改完必须重启服务」划算。
"""
import logging
from collections.abc import Callable
from typing import Any

from app.common import clock
from app.rag.bm25 import BM25Index, tokenize
from app.rag.embedding import embed_query
from app.rag.fuse import doc_key, rrf_fuse
from app.rag.index import INDEX_DIR, VECTORS_FILE, VectorIndex

logger = logging.getLogger(__name__)

# 每一路召回多少条候选。30 是因为 579 条的池子里，30 条之后的文档
# 基本不可能进最终 top5；取大只会让融合变慢、让噪声进来
_RECALL_K = 30

# 重排权重。RRF 分数已归一到 0~1，再和启发式分加权：
# 0.7 保召回排序的主体，0.3 留给「标题精确命中」和「新鲜度」
_RRF_WEIGHT = 0.7
_HEUR_WEIGHT = 0.3
_TITLE_WEIGHT = 0.6      # 启发式分内部：标题命中的占比
_FRESH_WEIGHT = 0.4      # 启发式分内部：新鲜度的占比
_FRESH_DAYS = 7          # 7 天内开始的活动给新鲜度加分

_index: VectorIndex | None = None
_loaded = False
_loaded_mtime: float = 0.0

# BM25 索引跟着 VectorIndex 的生命周期走：索引重载时一起重建，
# 避免出现「向量是新文档、BM25 是旧文档」这种对不上的状态
_bm25: BM25Index | None = None


def _index_mtime() -> float:
    """索引文件的修改时间；文件不存在时返回 0。"""
    try:
        return (INDEX_DIR / VECTORS_FILE).stat().st_mtime
    except OSError:
        return 0.0


def get_index() -> VectorIndex | None:
    """取索引；文件被别人改过就自动重新加载。"""
    global _index, _loaded, _loaded_mtime, _bm25
    mtime = _index_mtime()
    if not _loaded or mtime != _loaded_mtime:
        _index = VectorIndex.load()
        _loaded = True
        _loaded_mtime = mtime
        _bm25 = None            # 文档变了，关键词索引必须跟着重建
    return _index


def reload_index() -> None:
    """丢弃缓存，下次检索重新读盘（同进程重建索引后调用）。"""
    global _index, _loaded, _loaded_mtime, _bm25
    _index = None
    _bm25 = None
    _loaded = False
    _loaded_mtime = 0.0
    logger.info("索引缓存已失效，下次检索重新加载")


def _get_bm25(index: VectorIndex) -> BM25Index:
    """惰性构建关键词索引（579 条约 10ms，只在索引变更后建一次）。"""
    global _bm25
    if _bm25 is None:
        _bm25 = BM25Index(index.docs)
    return _bm25


def _dense_recall(
    index: VectorIndex,
    vector: list[float],
    where: Callable[[dict], bool] | None,
) -> list[tuple[dict[str, Any], float]]:
    """稠密召回。有谓词时全量打分再过滤，否则取 3 倍候选。"""
    limit = len(index.docs) if where is not None else _RECALL_K * 3
    hits = index.search(vector, top_k=limit)

    out: list[tuple[dict[str, Any], float]] = []
    seen: set[str] = set()
    for doc, score in hits:
        if where is not None and not where(doc):
            continue
        key = doc_key(doc)
        if key in seen:
            continue
        seen.add(key)
        out.append((doc, score))
        if len(out) >= _RECALL_K:
            break
    return out


def _rerank(
    query: str,
    fused: list[tuple[dict[str, Any], float]],
) -> list[tuple[dict[str, Any], float]]:
    """确定性重排：RRF 归一化后，叠加「标题命中」和「新鲜度」。

    两个加分项都表达成 0~1，再按 _HEUR_WEIGHT 混进 RRF 分，
    这样不会出现「一个启发式加分把整路排序推翻」的情况。
    """
    if not fused:
        return []

    peak = fused[0][1] or 1.0
    query_terms = set(tokenize(query))
    now = clock.now_cn()

    rescored: list[tuple[dict[str, Any], float]] = []
    for doc, rrf_score in fused:
        title_terms = set(tokenize(str(doc.get("title") or "")))
        title_hit = 1.0 if query_terms & title_terms else 0.0

        fresh = 0.0
        start = clock.parse_dt(doc.get("start_time"))
        if start is not None:
            days = (start - now).total_seconds() / 86400
            if 0 <= days <= _FRESH_DAYS:
                fresh = 1 - days / _FRESH_DAYS

        heuristic = _TITLE_WEIGHT * title_hit + _FRESH_WEIGHT * fresh
        final = _RRF_WEIGHT * (rrf_score / peak) + _HEUR_WEIGHT * heuristic
        rescored.append((doc, final))

    rescored.sort(key=lambda pair: pair[1], reverse=True)
    return rescored


async def retrieve(
    query: str,
    top_k: int = 5,
    min_score: float = 0.0,
    where: Callable[[dict], bool] | None = None,
) -> list[dict]:
    """检索 top_k 条最相关文档。

    Args:
        min_score: 最终得分下限（0~1，RRF 归一化后再加权的混合分），默认 0 不过滤。
        where: 候选过滤谓词（如「活动没结束」「本周内」）。**在排序截断之前生效**——
            先取 top 15 再过滤的话，「这周有什么活动」很可能一条不剩：
            不是真的没有，是名额被别的活动占满了。

    Returns:
        每条带 score（0~1 的混合分：RRF 归一化后按 0.7/0.3 加权启发式分），
        量级和旧版余弦相似度不同，别拿去和旧日志对比。
    """
    index = get_index()
    if index is None or not index.docs:
        logger.warning("向量索引为空，跳过检索（先跑 python -m app.rag.ingest）")
        return []

    vector = await embed_query(query)
    dense = _dense_recall(index, vector, where)
    sparse = [
        (doc, score)
        for doc, score in _get_bm25(index).search(query, top_k=_RECALL_K)
        if where is None or where(doc)
    ]

    fused = rrf_fuse([dense, sparse])
    ranked = _rerank(query, fused)

    results: list[dict] = []
    seen: set[str] = set()
    for doc, score in ranked:
        key = doc_key(doc)
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
            "检索 %d 条（稠密 %d / 关键词 %d / 融合 %d）：最高分 %.4f（%s）",
            len(results), len(dense), len(sparse), len(fused),
            results[0]["score"], results[0].get("title", "")[:20],
        )
    elif where is not None:
        logger.info("检索有谓词但零命中（候选 %d 条全部被过滤）", len(index.docs))
    return results
