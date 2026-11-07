"""PDF 文本层抽取：先看有没有文字，再决定要不要花 60 秒走 OCR。

为什么需要这一层（实测，2026-11-07 一份 1 页成绩单）：

    pypdfium2 直抽   0.01 秒  2454 字     干净、按行
    MinerU           60 秒    20149 字    86% 是 <td rowspan=1 colspan=1> 标签

PDF 的文本层是作者写进去的，不是识别出来的——能直抽就绝不该走 OCR。

判据是「每页文字量」，不是「图片清不清晰」：
    > 200 字/页   富文本页，直抽
    50~200 字/页  图文混排页，直抽，之后可选让视觉模型补图
    ≤ 50 字/页    扫描页 / 纯图片页，交 MinerU（它内部跑 OCR）

为什么不用「文档里有没有图片对象」来判：两个真实反例会判反。
    整页扫描的成绩单——图片对象只有 1 个，但全是文字、没有文本层，必须 OCR；
    带页眉 logo 的讲义——图片对象 20 个，但正文全是文本层，直抽最好，logo 是噪声。
"""
import logging
import re
from dataclasses import dataclass

import pypdfium2 as pdfium

from app.common.constants import (
    PAGE_TEXT_LOW,
    PAGE_TEXT_RICH,
    TABLE_NUMERIC_RATIO,
    TABLE_PAGE_RATIO,
)

logger = logging.getLogger(__name__)

# 文档级路由结果
ROUTE_TEXT = "text"     # 全页都有文本层、且不是表格：完全不用 OCR
ROUTE_MIXED = "mixed"   # 部分页是图：直抽 + 对低密度页做视觉补全
ROUTE_TABLE = "table"   # 表格密集：直抽语序不可靠，交 MinerU 还原版面
ROUTE_SCAN = "scan"     # 几乎没有文本层：整份交 MinerU

# 数字 token：识别表格行用（"计算机概论 必修 48(学时) 2.5 64" 有 4 个）
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def numeric_line_ratio(text: str) -> float:
    """「含 ≥2 个数字的行」占比。表格行普遍如此，散文段落几乎没有。"""
    lines = [line for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return 0.0
    hits = sum(1 for line in lines if len(_NUMBER.findall(line)) >= 2)
    return hits / len(lines)


@dataclass(frozen=True)
class PageText:
    """一页的文字。index 从 1 开始，和人类说的「第几页」一致。"""

    index: int
    text: str

    @property
    def chars(self) -> int:
        return len(self.text.strip())

    @property
    def kind(self) -> str:
        if self.chars > PAGE_TEXT_RICH:
            return "rich"
        if self.chars > PAGE_TEXT_LOW:
            return "mixed"
        return "low"


def inspect_pdf(data: bytes) -> list[PageText]:
    """按页抽文本层。加密/损坏时抛异常，由调用方降级到 MinerU。"""
    doc = pdfium.PdfDocument(data)
    try:
        pages: list[PageText] = []
        for i in range(len(doc)):
            try:
                text = doc[i].get_textpage().get_text_range()
            except Exception:  # noqa: BLE001 - 单页抽不出来不该拖垮整份文档
                logger.warning("第 %d 页文本层抽取失败，按空页处理", i + 1, exc_info=True)
                text = ""
            pages.append(PageText(index=i + 1, text=text))
        return pages
    finally:
        doc.close()


def classify(pages: list[PageText]) -> str:
    """整份文档走哪条路。

    看「低密度页占多少」而不是平均值：一份 100 页讲义里有 5 页是图，
    平均字符数照样很高，但平均值会把那 5 页的缺失完全掩盖掉。

    表格页单独判一次，因为实测发现：表格密集的 PDF（成绩单、课表）文字层里
    语序本身就是乱的，直抽越快越错，只有版面解析能还原。
    """
    if not pages:
        return ROUTE_SCAN
    low = sum(1 for p in pages if p.kind == "low")
    if low == len(pages):
        return ROUTE_SCAN

    text_pages = [p for p in pages if p.kind != "low"]
    if text_pages:
        tables = sum(1 for p in text_pages if is_table_page(p))
        if tables / len(text_pages) > TABLE_PAGE_RATIO:
            logger.info("表格页占比 %d/%d，语序不可靠，改走 MinerU", tables, len(text_pages))
            return ROUTE_TABLE

    if low:
        return ROUTE_MIXED
    return ROUTE_TEXT


def is_table_page(page: PageText) -> bool:
    """这一页是否表格密集（数字密度判据，见 constants 里的实测依据）。"""
    return numeric_line_ratio(page.text) > TABLE_NUMERIC_RATIO


def low_density_pages(pages: list[PageText], limit: int) -> list[int]:
    """需要视觉补全的页号（已按上限截断）。"""
    hits = [p.index for p in pages if p.kind == "low"]
    return hits[:limit]


def build_text(pages: list[PageText], skipped: list[int] | None = None) -> str:
    """把各页文本拼成一段，低密度页写一句占位说明。

    占位说明是必要的：不写的话模型看到的是一段连续文字，会以为这就是全文，
    然后对「第 5 页那张图讲了什么」凭空编造。写明了它才知道边界在哪。
    """
    skipped = skipped or []
    blocks: list[str] = []
    for page in pages:
        body = page.text.strip()
        if page.kind == "low":
            blocks.append(
                f"（第 {page.index} 页是图片或扫描页，"
                f"{'已用视觉模型补充内容' if page.index in skipped else '其中的图本轮未解析'}）"
            )
        elif body:
            # 行与行之间插空行：split_markdown 按 `\n{2,}` 分段，这样每行都是独立段落，
            # 400 字的块会在行边界累积。PDF 抽出来的文本本来就是一行一条记录
            # （成绩单、课表、书目），按行切才不会把一条记录劈成两半。
            lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
            blocks.append(f"第 {page.index} 页\n" + "\n\n".join(lines))
    return "\n\n".join(blocks)
