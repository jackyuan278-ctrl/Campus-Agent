"""多路融合：RRF（Reciprocal Rank Fusion）。

为什么用名次融合，而不是「加权分数相加」：
    稠密是余弦（0~1），BM25 是任意正数（同一查询下 top 可能是 12.7，
    换个查询是 3.1）。两个量纲不可加。做 min-max / z-score 归一化能凑合，
    但归一化会随候选集变化而抖动：同一个文档，候选池换一批，分数就变了。

    RRF 只用名次，天生免疫量纲：
        score(d) = Σ_route 1 / (k + rank_route(d))
    k=60 是原论文经验值，作用是压低头部差距（第 1 名和第 2 名不至于差一倍）。

它真正买到的东西：
    不是「哪一路更强」，而是「**两路都认可的文档更可信**」。
    一路排第 3、另一路排第 5 的文档，会排在「一路第 1、另一路完全没召回」的前面。
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)

# RRF 的 k 参数。取 60 是论文默认值；改小会更看重头部名次
RRF_K = 60


def doc_key(doc: dict[str, Any]) -> str:
    """文档唯一键。用 source+id，不用 title——标题可能撞车。"""
    return f"{doc.get('source')}:{doc.get('id')}"


def rrf_fuse(
    ranked_lists: list[list[tuple[dict[str, Any], float]]],
    k: int = RRF_K,
) -> list[tuple[dict[str, Any], float]]:
    """把多路排序结果融成一路。

    Args:
        ranked_lists: 每一路是 [(doc, 该路自己的分数)]，按该路排序好。
                      这里的分数只用于日志，融合只认名次。

    Returns:
        [(doc, rrf_score)]，按 rrf_score 降序。rrf_score 量级约 1/(k+rank)，
        所以两路都在前 5 时约 0.032，一路命中约 0.016。
    """
    scores: dict[str, float] = {}
    picked: dict[str, dict[str, Any]] = {}
    routes: dict[str, int] = {}

    for ranked in ranked_lists:
        for rank, (doc, _raw_score) in enumerate(ranked, start=1):
            key = doc_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            routes[key] = routes.get(key, 0) + 1
            # 同一文档被多路召回时，保留第一次见到的副本即可（内容相同）
            picked.setdefault(key, doc)

    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    multi = sum(1 for n in routes.values() if n > 1)
    logger.debug(
        "RRF 融合：%d 路 → %d 条候选，其中 %d 条被两路同时召回",
        len(ranked_lists), len(fused), multi,
    )
    return [(picked[key], score) for key, score in fused]
