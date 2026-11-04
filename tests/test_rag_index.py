"""向量索引单元测试（纯 numpy，不调 embedding，零 API 费用）。"""
import pytest

from app.rag.index import VectorIndex


def _build() -> VectorIndex:
    docs = [{"title": "a"}, {"title": "b"}, {"title": "c"}]
    # a 和 q 同向，b 正交，c 接近 a
    return VectorIndex.build([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], docs)


def test_search_orders_by_similarity():
    """最相似的排第一，分数递减。"""
    hits = _build().search([1.0, 0.0], top_k=3)
    assert hits[0][0]["title"] == "a"
    assert hits[0][1] > hits[1][1]
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)


def test_search_top_k_truncates():
    assert len(_build().search([1.0, 0.0], top_k=2)) == 2


def test_dimension_mismatch_raises():
    """查询维度与索引不一致时必须报错，不能静默返回垃圾结果。"""
    with pytest.raises(ValueError, match="维度"):
        _build().search([1.0, 0.0, 0.0], top_k=1)


def test_empty_index_returns_empty():
    assert VectorIndex.build([], []).search([1.0, 0.0]) == []
