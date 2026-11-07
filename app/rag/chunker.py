"""切分策略：按「解析路由 + 文件格式」取参数，并按标题层级给子块补上下文。

为什么单独一个模块：
    切分是纯字符串逻辑，和「从 PDF/Word 里取文本」是两件事。放在一起会让
    doc_ingest 既管解析又管切分，改一处要读两百行。

为什么参数按 profile 取，而不是散落的常量：
    不同来源的文档，最优块大小本来就不同（OCR 出来的文本错字多、断句碎，
    块要更小；原生文本层出来的连续性好，块可以大一点）。
    但——**目前只有 PDF 这两条跑过真实样本**，其余格式先复用默认值并标注
    「待样本校准」。宁可表格里写 TODO，也不要拍脑袋填一个别人的经验值。

为什么标题要拼进块里，而不是只做元数据：
    模型最终看到的只有 text。标题只存字段不拼进 text，检索和生成都用不上，
    等于没做。代价是每块多 10~20 字，可以接受——它换来的是「这块属于哪一章」。
"""
import logging
import re
from dataclasses import dataclass

from app.common.constants import DOC_CHUNK_OVERLAP, DOC_CHUNK_SIZE

logger = logging.getLogger(__name__)

# markdown 标题：# ~ ######
_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")

# 标题路径最多带几层。带太深会把前缀撑得很长（"第3章 > 3.2 > 3.2.1 > ..."），
# 而检索真正需要的是「最近的上下文」，两层足够。
_MAX_HEADING_DEPTH = 2


@dataclass(frozen=True)
class ChunkProfile:
    """一种来源的切分参数。"""

    size: int
    overlap: int
    inject_heading: bool = True


@dataclass(frozen=True)
class Chunk:
    """一个块 + 两个标签。

    text 才是进向量的东西；page / kind 是**随块携带的元数据**，不参与向量计算。
    这个区分很重要：如果把页码拼进 text，同一份文档重传时页码一变，
    指纹就变，所有向量白算一遍（时间字段踩过同样的坑）。
    """

    text: str
    page: int | None = None
    kind: str = "text"      # text / table


# 「第 N 页」标记行：由 pdf_text.build_text 插入，这里取出来变成字段。
_PAGE_MARK = re.compile(r"^第\s*(\d+)\s*页$")


# 默认值来自配额倒推：top_k=4 × 400 字 ≈ 1000 token，日配额 10 万 → 约 100 轮。
# 改成 800 会让单轮翻倍，必须先用评测集验证 Recall@5 是否真的变好。
_DEFAULT = ChunkProfile(size=DOC_CHUNK_SIZE, overlap=DOC_CHUNK_OVERLAP)

# 按解析路由覆盖。没列出的路由/格式一律走 _DEFAULT（= 待样本校准）。
_PROFILES: dict[str, ChunkProfile] = {
    # 原生文本层：连续性好，按段落切即可（实测样本：无，等讲义）
    "text": _DEFAULT,
    # 表格密集：MinerU 输出，行是原子单位，切点天然落在行边界
    "table": _DEFAULT,
    # 扫描件：OCR 文本断句碎、错字多，理论块应更小——但**没有样本**，先不动
    "scan": _DEFAULT,  # TODO 待扫描件样本校准
}


def chunk_profile(route: str, suffix: str = "") -> ChunkProfile:
    """取切分参数。未知路由回落默认值，不报错也不拦。"""
    profile = _PROFILES.get(route)
    if profile is None:
        logger.debug("未登记的解析路由 %r（后缀 %r），使用默认切分参数", route, suffix)
        return _DEFAULT
    return profile


def split_markdown(
    text: str,
    size: int = DOC_CHUNK_SIZE,
    overlap: int = DOC_CHUNK_OVERLAP,
) -> list[str]:
    """按段落聚合成块，保留 overlap 字重叠。

    优先在段落边界切（不打散一句话），单个段落本身超长时才硬切。
    这是**唯一的切块算法**：markdown 段落、HTML 表格行、pipe 表格行
    在进入这里之前都已经被归约成「段落序列」，所以不需要分支。

    唯一的例外是表格行：`| a | b |` 是原子单位，硬切会切出
    「… | 创新创」+「修 | 36(学时) …」这种半个单元格，检索和阅读都废。
    所以表格行不参与硬切，超长就让它独占一块（上限 size×2，再长才认输）。
    """
    text = (text or "").strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    hard_limit = size * 2

    for para in paragraphs:
        # 表格行：整行是原子单位，宁可超长也不切断
        if para.startswith("|") and len(para) > size:
            if buf:
                chunks.append(buf)
                buf = ""
            if len(para) <= hard_limit:
                chunks.append(para)
            else:
                chunks.extend(para[i:i + hard_limit] for i in range(0, len(para), hard_limit))
            continue

        if not buf:
            buf = para
        elif len(buf) + len(para) + 1 <= size:
            buf = f"{buf}\n{para}"
            continue
        else:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap else ""
            buf = f"{tail}\n{para}" if tail else para

        # 单段就超过块大小 → 硬切（代码块、超长句子常见）
        while len(buf) > size:
            chunks.append(buf[:size])
            buf = buf[size - overlap:] if overlap else buf[size:]

    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def _heading_path(stack: list[tuple[int, str]]) -> str:
    """把标题栈拼成一行前缀，只取最近几层。"""
    recent = [text for _, text in stack[-_MAX_HEADING_DEPTH:]]
    return "｜".join(recent)


def split_by_heading(
    text: str,
    size: int = DOC_CHUNK_SIZE,
    overlap: int = DOC_CHUNK_OVERLAP,
    inject_heading: bool = True,
) -> list[str]:
    """按标题分段，每段内部再按段落切块，每块前置所属标题。

    为什么要先按标题切：
        讲义里「3.2 时间片轮转」的正文被切成三块后，中间那块完全看不出属于哪一章。
        检索命中它时，模型拿到的是「每个进程被分配一个时间片…」，不知道上下文。
        按标题分段后，每块都带着「第3章｜3.2 时间片轮转」。

    没有标题的文档（原生 PDF 直抽、OCR 结果）会得到一个空栈，
    退化成普通段落切分，行为和 split_markdown 一致。
    """
    text = (text or "").strip()
    if not text:
        return []

    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []

    def flush() -> None:
        body = "\n\n".join(buf).strip()
        if body:
            sections.append((_heading_path(stack), body))
        buf.clear()

    for line in text.split("\n"):
        match = _HEADING.match(line.strip())
        if not match:
            buf.append(line)
            continue
        flush()
        level = len(match.group(1))
        # 同级或更浅的标题要先把更深的弹掉，否则路径会串味
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, match.group(2).strip()))
    flush()

    if not sections:
        return split_markdown(text, size, overlap)

    chunks: list[str] = []
    for path, body in sections:
        for piece in split_markdown(body, size, overlap):
            chunks.append(f"{path}\n{piece}" if (inject_heading and path) else piece)
    return chunks


def _kind_of(text: str) -> str:
    """粗判块类型：多数行以 | 开头就当表格块。"""
    lines = [line for line in text.split("\n") if line.strip()]
    if not lines:
        return "text"
    pipes = sum(1 for line in lines if line.strip().startswith("|"))
    return "table" if pipes / len(lines) >= 0.6 else "text"


def split_with_pages(
    text: str,
    size: int = DOC_CHUNK_SIZE,
    overlap: int = DOC_CHUNK_OVERLAP,
    inject_heading: bool = True,
) -> list[Chunk]:
    """一次遍历：同时处理页标记和标题，产出带页码/类型的块。

    三个决定：
    1. **遇到页标记就冲一段**。这样任何一段都不会跨页，页码是确定的；
       代价是跨页的长段落会被页边界截断——讲义里页末通常正好是段落结束，
       这种情况少见，而且 80 字重叠会把断处补上一半。
    2. **标题栈跨页保留**。不能先按页切开再各自处理标题，否则第 2 页的
       「3.2 时间片轮转」会丢掉第 1 页的「第3章 进程管理」这个父级。
    3. **页标记本身不进正文**。「第 3 页」四个字对语义相似度没有贡献，
       只是噪声；页码改由 Chunk.page 携带，检索时再补回成「（第 3 页）」。
    """
    sections: list[tuple[str, int | None, str]] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    page: int | None = None
    buf_page: int | None = None

    def flush() -> None:
        nonlocal buf, buf_page
        body = "\n".join(buf).strip()
        if body:
            sections.append((_heading_path(stack), buf_page, body))
        buf, buf_page = [], None

    for line in (text or "").split("\n"):
        mark = _PAGE_MARK.match(line.strip())
        if mark:
            flush()
            page = int(mark.group(1))
            continue

        heading = _HEADING.match(line.strip())
        if heading:
            flush()
            level = len(heading.group(1))
            # 同级或更浅的标题要先把更深的弹掉，否则路径会串味
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading.group(2).strip()))
            continue

        buf.append(line)
        if buf_page is None:
            buf_page = page
    flush()

    chunks: list[Chunk] = []
    for path, chunk_page, body in sections:
        for piece in split_markdown(body, size, overlap):
            text_piece = f"{path}\n{piece}" if (inject_heading and path) else piece
            chunks.append(
                Chunk(text=text_piece, page=chunk_page, kind=_kind_of(text_piece))
            )
    return chunks
