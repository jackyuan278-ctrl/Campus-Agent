"""文档入库：MinerU 解析 → 切块 → 向量化 → 写 Milvus Lite。

为什么不整篇进 prompt：
    一份 50 页讲义约 3 万字，全塞进去是 3 万 token，而日配额只有 10 万——
    三轮就爆；而且每轮都要重发一遍历史。切块后每轮只带最相关的 4 块
    （约 1600 字 ≈ 1000 token），差 30 倍。

为什么切块要重叠：
    答案经常正好落在切分点上。相邻块重叠 80 字，被切断的半句话在下一块里还在。
"""
import logging
import re

from app.common.constants import DOC_CHUNK_OVERLAP, DOC_CHUNK_SIZE
from app.llm.mineru import MineruError, parse_document
from app.rag.doc_store import upsert_chunks
from app.rag.embedding import embed_texts

logger = logging.getLogger(__name__)

# 能走 MinerU 的文档类型。图片走视觉模型，不在这里。
DOC_SUFFIXES = (".pdf", ".doc", ".docx", ".txt", ".md")
DOC_CONTENT_TYPES = (
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/markdown",
)


def is_document(content_type: str, filename: str) -> bool:
    """判断附件要不要走文档解析。content-type 缺失时退回看后缀。"""
    if (content_type or "").split(";")[0].strip().lower() in DOC_CONTENT_TYPES:
        return True
    return (filename or "").lower().endswith(DOC_SUFFIXES)


def split_markdown(
    text: str,
    size: int = DOC_CHUNK_SIZE,
    overlap: int = DOC_CHUNK_OVERLAP,
) -> list[str]:
    """按段落聚合成块，保留 overlap 字重叠。

    优先在段落边界切（不打散一句话），单个段落本身超长时才硬切。
    """
    text = (text or "").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    buf = ""

    for para in paragraphs:
        if not buf:
            buf = para
        elif len(buf) + len(para) + 1 <= size:
            buf = f"{buf}\n{para}"
            continue
        else:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap else ""
            buf = f"{tail}\n{para}" if tail else para

        # 单段就超过块大小 → 硬切（代码块、表格行常见）
        while len(buf) > size:
            chunks.append(buf[:size])
            buf = buf[size - overlap:] if overlap else buf[size:]

    if buf.strip():
        chunks.append(buf.strip())
    return chunks


async def index_document(file_url: str, file_name: str) -> dict:
    """解析并索引一份文档，返回 {chunks, chars, images}。失败抛 MineruError。"""
    markdown, images = await parse_document(file_url)
    chunks = split_markdown(markdown)
    if not chunks:
        raise MineruError("没有从文档里提取到文字（可能是纯扫描件或空白页）")

    vectors = await embed_texts(chunks)
    count = await upsert_chunks(file_url, file_name, chunks, vectors)
    logger.info("文档入索引：%s（%d 块 / %d 字 / %d 张图未解析）",
                file_name, count, len(markdown), images)
    return {"chunks": count, "chars": len(markdown), "images": images}
