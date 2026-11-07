"""BM25 关键词召回：补稠密向量的短板。

为什么需要第二路：
    稠密向量擅长「意思相近」，不擅长「字面对得上」。
    「计算机二级」「1202 教室」「CS 408」这类专名、编号、缩写，
    换个说法向量还能对上，但字面完全一样时它反而不保证排第一。
    反过来，稠密对「有啥好玩的」这种口语很强，BM25 在这种查询上基本没用。

    两路的失败模式不重叠，这才是融合有意义的前提——如果两路一样烂，
    RRF 只是把同一个错误排两遍。

为什么分词用字符 bigram：
    「数学建模」→ 数学 / 学建 / 建模。零依赖，不怕新词，
    也不会像词典分词那样在「软件工程」这种词上切错。
    代价是有少量无意义二元组（学建），BM25 的 IDF 会把它们的权重压下去。

为什么索引不落文件：
    579 条文档、平均 100 个 token，现建倒排约 10ms。
    落文件就要和 vectors.npy / docs.json 保持同步——那正是之前踩过的
    「向量 579 行、文档 578 条」的坑。宁可花 10ms，不要第三个需要对齐的文件。
"""
import logging
import math
import re
from collections import Counter, defaultdict
from typing import Any

logger = logging.getLogger(__name__)

# 切词：先按「非字母数字非汉字」切开，再对纯汉字段做 bigram
_SPLIT = re.compile(r"[^\w\u4e00-\u9fff]+")
_CJK = re.compile(r"[\u4e00-\u9fff]+")

# BM25 标准参数。k1 控制词频饱和速度，b 控制文档长度归一化强度
_K1 = 1.2
_B = 0.75

# 标题重复几次（等价于标题字段加权）。标题命中比正文命中重要得多
_TITLE_WEIGHT = 3


def tokenize(text: str) -> list[str]:
    """中英混排文本 → token 列表。"""
    tokens: list[str] = []
    for seg in _SPLIT.split((text or "").lower()):
        if not seg:
            continue
        if _CJK.fullmatch(seg):
            if len(seg) == 1:
                tokens.append(seg)          # 单字保留（"书"这种查询）
            else:
                tokens.extend(seg[i:i + 2] for i in range(len(seg) - 1))
        else:
            tokens.append(seg)              # 英文/数字整段保留
    return tokens


class BM25Index:
    """内存倒排索引 + BM25 打分。构建一次，查询很多次。"""

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self.docs = docs
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._lengths: list[int] = []

        for row, doc in enumerate(docs):
            title = str(doc.get("title") or "")
            # 标题加权：重复计入词频，比给每个 token 乘系数更好调
            tokens = tokenize(" ".join([title] * _TITLE_WEIGHT + [str(doc.get("content") or "")]))
            counts = Counter(tokens)
            self._lengths.append(len(tokens) or 1)
            for term, freq in counts.items():
                self._postings[term].append((row, freq))

        self._avg_len = sum(self._lengths) / len(self._lengths) if self._lengths else 1.0
        logger.info("BM25 索引就绪：%d 条 / %d 个词元", len(docs), len(self._postings))

    def search(self, query: str, top_k: int) -> list[tuple[dict[str, Any], float]]:
        """返回 [(doc, score)]，按分数降序，只包含命中至少一个词的文档。"""
        terms = set(tokenize(query))
        if not terms:
            return []

        total = len(self.docs)
        scores: dict[int, float] = defaultdict(float)
        for term in terms:
            postings = self._postings.get(term)
            if not postings:
                continue
            df = len(postings)
            # 标准 BM25 IDF，加 1 保证永远非负（避免高频词倒扣分）
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            for row, freq in postings:
                norm = _K1 * (1 - _B + _B * self._lengths[row] / self._avg_len)
                scores[row] += idf * freq * (_K1 + 1) / (freq + norm)

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(self.docs[row], score) for row, score in ranked]
