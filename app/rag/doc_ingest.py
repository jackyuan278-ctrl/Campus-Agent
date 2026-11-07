"""文档入库：分流解析 → 归一化 → 切块 → 向量化 → 写 Milvus Lite。

为什么不整篇进 prompt：
    一份 50 页讲义约 3 万字，全塞进去是 3 万 token，而日配额只有 10 万——
    三轮就爆；而且每轮都要重发一遍历史。切块后每轮只带最相关的 4 块
    （约 1600 字 ≈ 1000 token），差 30 倍。

为什么切块要重叠：
    答案经常正好落在切分点上。相邻块重叠 80 字，被切断的半句话在下一块里还在。

为什么先分流再解析（实测驱动）：
    同一份 1 页成绩单，pypdfium2 直抽 0.01 秒得到 2454 字干净文本；
    MinerU 花 60 秒得到 20149 字，其中 86% 是 <td rowspan=1 colspan=1> 标签，
    切成 63 块、烧 63 次 embedding——而真实内容只有 2000 多字。
    文本层是作者写进去的，不是识别出来的，能直抽就绝不走 OCR。
"""
import logging
import re
from pathlib import Path

from app.common.constants import MAX_IMAGE_PAGES
from app.llm.mineru import MineruError, parse_document
from app.llm.pdf_text import (
    ROUTE_MIXED,
    ROUTE_SCAN,
    ROUTE_TABLE,
    ROUTE_TEXT,
    build_text,
    classify,
    inspect_pdf,
    low_density_pages,
)
from app.rag.chunker import chunk_profile, split_with_pages
from app.rag.chunker import split_markdown as split_markdown  # noqa: PLC0414  兼容旧的 import 路径
from app.rag.doc_store import upsert_chunks
from app.rag.embedding import embed_texts

logger = logging.getLogger(__name__)

# 能走解析的文档类型。图片走视觉模型，不在这里。
DOC_SUFFIXES = (".pdf", ".doc", ".docx", ".txt", ".md")
DOC_CONTENT_TYPES = (
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/markdown",
)

# 纯文本类型：连 MinerU 都不用，直接解码
_PLAIN_SUFFIXES = (".txt", ".md")

# ---------- HTML 表格归一化 ----------
# MinerU 对「有合并单元格的表格」会输出内联 HTML（Markdown 的 | a | b | 语法
# 不支持 rowspan/colspan）。实测一份成绩单 86% 的字符是标签，54% 是 rowspan 属性。
# 不归一化的话：切块切在标签中间、向量里全是属性噪声、检索基本失效。
_TABLE = re.compile(r"<table[^>]*>(.*?)</table>", re.S | re.I)
_ROW = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_CELL = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S | re.I)
# 只剥「已知的」标签，不用 <[^>]+>：后者会把正文里的 "a<b" 一起吃掉
_KNOWN_TAG = re.compile(
    r"</?(?:table|thead|tbody|tfoot|tr|td|th|div|span|p|br|img|hr"
    r"|sup|sub|b|strong|i|em|u|ul|ol|li|h[1-6])\b[^>]*>",
    re.I,
)


def _table_to_rows(html: str) -> str:
    """一张 HTML 表格 → 若干行 `| 单元格 | 单元格 |`，行间空行。

    行间用空行（而不是单个换行）是关键：split_markdown 按 `\\n{2,}` 分段，
    这样每一行表格都是独立段落，400 字缓冲会在行边界累积，不会再切在标签中间。
    全空的行（实测里出现过整块 `></td><td>`）直接丢掉。
    """
    rows: list[str] = []
    for tr in _ROW.findall(html):
        cells = [re.sub(r"\s+", " ", _KNOWN_TAG.sub("", c)).strip() for c in _CELL.findall(tr)]
        if any(cells):
            rows.append("| " + " | ".join(cells) + " |")
    return "\n\n".join(rows)


# markdown pipe 表的一行：| a | b |；以及分隔行 | --- | --- |
_PIPE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_PIPE_SEP = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


def _explode_pipe_tables(text: str) -> str:
    """把连续的 `| a | b |` 行拆成「一行一段」，让两种表格形态统一。

    为什么要拆：MinerU 对简单表格输出 markdown pipe 表，行与行之间只有单个换行。
    split_markdown 按 `\\n{2,}` 分段，会把整张表当成一个巨大段落，超过 400 字就硬切——
    实测切出「块0 末行 = | 课」「块1 首行 = 程14 | 2.0 | 80 |」，一行表格被拦腰砍断。

    拆法：表头 + 分隔行捆成一段，每个数据行各成一段。这样切点必然落在行边界。
    """
    out: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if not buf:
            return
        head = [row for i, row in enumerate(buf) if i == 0 or _PIPE_SEP.match(row)]
        body = [row for i, row in enumerate(buf) if not (i == 0 or _PIPE_SEP.match(row))]
        out.extend(head)
        if body:
            out.append("")
            for row in body:
                out.extend([row, ""])
        buf.clear()

    for line in text.split("\n"):
        if _PIPE_ROW.match(line):
            buf.append(line.strip())
        else:
            flush()
            out.append(line)
    flush()
    return "\n".join(out)


def normalize_markdown(text: str) -> str:
    """把解析结果洗成「适合切块和向量化」的纯文本。幂等。

    两种表格形态都要处理，因为 MinerU 按复杂度给不同输出：
        复杂表格（有合并单元格）→ 内联 HTML <table>
        简单表格                → markdown pipe 表 `| a | b |`
    只处理前者的话，后者会被 400 字硬切，切出「| 课」+「程14 | 2.0 | 80 |」——
    实测过一次，块末行就是半个表格行。
    """
    if not text:
        return ""
    # 1) HTML 表格转成 pipe 行（此时标签还没剥，_CELL 才能认出单元格）
    text = _TABLE.sub(lambda m: _table_to_rows(m.group(1)), text)
    # 2) 剥掉残留的已知标签（嵌套表格、div 包裹之类）
    text = _KNOWN_TAG.sub("", text)
    # 3) 让 pipe 表格「一行一段」，和 HTML 表格统一成同一种形态
    text = _explode_pipe_tables(text)
    # 4) 统一换行、压掉多余空行（PDF 抽出来常带 \r\n）
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_document(content_type: str, filename: str) -> bool:
    """判断附件要不要走文档解析。content-type 缺失时退回看后缀。"""
    if (content_type or "").split(";")[0].strip().lower() in DOC_CONTENT_TYPES:
        return True
    return (filename or "").lower().endswith(DOC_SUFFIXES)


# split_markdown 已移到 app/rag/chunker.py（切分逻辑独立成模块），
# 这里保留同名导入，旧的 `from app.rag.doc_ingest import split_markdown` 不会断。


async def _extract(
    file_url: str,
    file_name: str,
    data: bytes | None,
) -> tuple[str, str, int]:
    """分流解析。返回 (正文, 路由, 图片页数)。

    路由取值：
        plain   纯文本（.txt/.md），解码即用
        text    PDF 有完整文本层，直抽，零 OCR
        mixed   PDF 部分页是图，直抽 + 标注低密度页
        scan    PDF 几乎无文本层，整份交 MinerU
        mineru  doc/docx 或扫描件兜底
    """
    suffix = Path(file_name or "").suffix.lower()

    if suffix in _PLAIN_SUFFIXES and data:
        return data.decode("utf-8", errors="replace"), "plain", 0

    if suffix == ".pdf" and data:
        try:
            pages = inspect_pdf(data)
        except Exception:  # noqa: BLE001 - 加密/损坏的 PDF 交给 MinerU 兜底
            logger.warning("PDF 文本层读取失败，降级 MinerU：%s", file_name, exc_info=True)
        else:
            route = classify(pages)
            if route in (ROUTE_TEXT, ROUTE_MIXED):
                skipped = low_density_pages(pages, MAX_IMAGE_PAGES)
                logger.info(
                    "PDF 直抽文本层：%s（%d 页，路由=%s，低密度页 %d 页）",
                    file_name, len(pages), route, len(skipped),
                )
                return build_text(pages, skipped), route, len(skipped)
            if route == ROUTE_TABLE:
                logger.info("PDF 表格密集，交 MinerU 还原版面：%s", file_name)
            else:  # ROUTE_SCAN
                logger.info("PDF 无文本层（扫描件），交 MinerU：%s", file_name)

    markdown, images = await parse_document(file_url)
    return markdown, "mineru", images


async def index_document(
    file_url: str,
    file_name: str,
    data: bytes | None = None,
    content_type: str = "",
) -> dict:
    """解析并索引一份文档，返回 {chunks, chars, images, route}。失败抛 MineruError。

    data 是可选的原始字节：给了才能走「文本层直抽」，不给就只能花 60 秒走 MinerU。
    上传接口手里有这份字节（刚读完），顺手传进来最省事。
    """
    raw, route, images = await _extract(file_url, file_name, data)
    markdown = normalize_markdown(raw)
    profile = chunk_profile(route, Path(file_name or "").suffix.lower())
    chunks = split_with_pages(
        markdown, profile.size, profile.overlap, profile.inject_heading
    )
    if not chunks:
        raise MineruError("没有从文档里提取到文字（可能是纯扫描件或空白页）")

    # 只有 text 进向量；page / kind 是元数据，跟着块走但不参与相似度计算
    vectors = await embed_texts([chunk.text for chunk in chunks])
    count = await upsert_chunks(file_url, file_name, chunks, vectors)
    logger.info(
        "文档入索引：%s（%s 路由 / %d 块 / %d 字 / 切分 %d+%d / %d 张图未解析）",
        file_name, route, count, len(markdown), profile.size, profile.overlap, images,
    )
    return {"chunks": count, "chars": len(markdown), "images": images, "route": route}
