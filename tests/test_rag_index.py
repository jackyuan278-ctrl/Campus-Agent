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


def test_load_rejects_mismatched_rows(tmp_path):
    """向量行数和文档条数对不上时必须报错，不能静默返回错位结果。

    vectors.npy 本身不知道每行对应哪个文档，全靠「第 i 行 ↔ docs[i]」这个约定。
    一旦错位，search 会把分数安到错误的文档上，而且完全不报错——比直接崩更糟。
    """
    import json

    idx = VectorIndex.build([[1.0, 0.0], [0.0, 1.0]], [{"title": "a"}, {"title": "b"}])
    idx.save(tmp_path)

    docs = json.loads((tmp_path / "docs.json").read_text(encoding="utf-8"))
    (tmp_path / "docs.json").write_text(json.dumps(docs[:1]), encoding="utf-8")

    with pytest.raises(ValueError, match="索引文件损坏"):
        VectorIndex.load(tmp_path)


def test_save_is_atomic(tmp_path):
    """保存后不能残留 .tmp 文件，且两个文件都能正常读回。"""
    idx = VectorIndex.build([[1.0, 0.0]], [{"title": "a"}])
    idx.save(tmp_path)

    assert not list(tmp_path.glob("*.tmp")), "原子写不应留下临时文件"
    loaded = VectorIndex.load(tmp_path)
    assert loaded is not None
    assert len(loaded.docs) == 1
