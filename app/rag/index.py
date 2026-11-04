"""内存向量索引：numpy 矩阵 + 余弦相似度。

为什么不用 Milvus：活动/问答各 1000 条，总量级 10^3。
向量矩阵只有 2000 x 1024 个 float32（约 8MB），一次矩阵乘法几十毫秒，
比维护一个向量库简单得多。数据量上到 10^5 再换 Milvus。
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# 索引落地目录（相对项目根）
INDEX_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "rag"
VECTORS_FILE = "vectors.npy"
DOCS_FILE = "docs.json"


@dataclass
class VectorIndex:
    """向量 + 文档元数据。

    vectors 已做 L2 归一化，所以余弦相似度退化成点积（一次矩阵乘）。
    """

    vectors: np.ndarray          # shape: (N, dim), float32, 已归一化
    docs: list[dict[str, Any]]   # 与 vectors 行一一对应

    @classmethod
    def build(cls, vectors: list[list[float]], docs: list[dict[str, Any]]) -> "VectorIndex":
        if len(vectors) != len(docs):
            raise ValueError(f"向量数({len(vectors)})与文档数({len(docs)})不一致")
        if not vectors:
            # 空索引：给个形状合法（0, 0）的矩阵，后续 search 会因为 docs 为空直接返回 []
            return cls(vectors=np.zeros((0, 0), dtype=np.float32), docs=[])
        matrix = np.asarray(vectors, dtype=np.float32)
        return cls(vectors=cls._l2_normalize(matrix), docs=docs)

    @staticmethod
    def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # 防止除零
        return matrix / norms

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.size else 0

    def search(self, query_vector: list[float], top_k: int = 5) -> list[tuple[dict[str, Any], float]]:
        """返回 [(doc, score)]，按相似度降序。"""
        if not self.docs:
            return []
        q = np.asarray(query_vector, dtype=np.float32)
        if q.shape[0] != self.dim:
            # 维度不一致（换了 embedding 模型）直接报错，别让它静默返回垃圾结果
            raise ValueError(f"查询向量维度 {q.shape[0]} 与索引 {self.dim} 不匹配，请重建索引")
        q = q / (np.linalg.norm(q) or 1.0)
        scores = self.vectors @ q
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.docs[i], float(scores[i])) for i in top_idx]

    # ---------- 落盘 / 加载 ----------

    def save(self, directory: Path | None = None) -> Path:
        target = directory or INDEX_DIR
        target.mkdir(parents=True, exist_ok=True)
        np.save(target / VECTORS_FILE, self.vectors)
        (target / DOCS_FILE).write_text(
            json.dumps(self.docs, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("索引已保存：%s（%d 条，维度 %d）", target, len(self.docs), self.dim)
        return target

    @classmethod
    def load(cls, directory: Path | None = None) -> "VectorIndex | None":
        target = directory or INDEX_DIR
        vectors_path = target / VECTORS_FILE
        docs_path = target / DOCS_FILE
        if not vectors_path.exists() or not docs_path.exists():
            logger.warning("索引文件不存在：%s", target)
            return None
        vectors = np.load(vectors_path)
        docs = json.loads(docs_path.read_text(encoding="utf-8"))
        logger.info("索引已加载：%d 条，维度 %d", len(docs), vectors.shape[1])
        return cls(vectors=vectors, docs=docs)
