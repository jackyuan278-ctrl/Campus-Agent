"""文档切块与类型判断测试。

切块质量直接决定检索能不能命中，所以这部分单独锁住：
块太大浪费 token、块太小丢上下文、重叠没了答案会被切断。
"""
from app.rag.doc_ingest import is_document, split_markdown

SIZE = 200
OVERLAP = 40


def _paragraph(tag: str, times: int = 10) -> str:
    return f"{tag}：" + "这是一段用来测试切块的正文。" * times


def test_each_chunk_within_size_limit():
    text = "\n\n".join(_paragraph(f"第{i}段") for i in range(6))
    chunks = split_markdown(text, size=SIZE, overlap=OVERLAP)

    assert len(chunks) > 1, "六段长文本应该被切成多块"
    assert all(len(c) <= SIZE for c in chunks), "任何一块都不能超过上限"


def test_adjacent_chunks_overlap():
    """上一块结尾的内容要出现在下一块开头，否则答案正好被切断时会丢。"""
    text = "\n\n".join(_paragraph(f"第{i}段") for i in range(6))
    chunks = split_markdown(text, size=SIZE, overlap=OVERLAP)

    overlap_text = chunks[0][-OVERLAP:]
    assert overlap_text in chunks[1], "相邻块之间必须有重叠"


def test_no_content_is_lost():
    """切块不能丢内容：最后一段的标记必须出现在最后一块里。"""
    tail_marker = "最后一段的独特标记"
    text = "\n\n".join([_paragraph("开头", 5), f"{tail_marker}结束"])
    chunks = split_markdown(text, size=SIZE, overlap=OVERLAP)

    assert tail_marker in chunks[-1]


def test_empty_text_returns_no_chunks():
    assert split_markdown("   \n\n  ") == []


def test_short_text_stays_single_chunk():
    assert split_markdown("只有一句话。", size=SIZE, overlap=OVERLAP) == ["只有一句话。"]


def test_oversized_paragraph_is_hard_split():
    """代码块、长表格这类没有段落分隔的整块文本，必须硬切而不是原样返回。"""
    text = "啊" * (SIZE * 3)
    chunks = split_markdown(text, size=SIZE, overlap=OVERLAP)

    assert len(chunks) >= 2
    assert all(len(c) <= SIZE for c in chunks)


def test_is_document_by_content_type_and_suffix():
    assert is_document("application/pdf", "a.pdf")
    assert is_document("text/markdown", "笔记")
    # content-type 缺失时看后缀
    assert is_document("", "讲义.docx")
    # 图片不算文档，走视觉模型
    assert not is_document("image/png", "题目.png")
    assert not is_document("", "题目.jpg")
