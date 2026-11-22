"""Pre-Processing 层：在进入 agent loop 之前，同步处理附件（图片视觉识别 + 文档全文注入/向量检索）。

graph.py 的流程已简化为  START → pre → agent ...（不再单独跑 image_agent 节点）。
本模块把旧 image_agent 的**全部**逻辑复制过来——图片识别 + summarization 进 history +
短文档全文注入 + 长文档向量检索 + doc 档追问复用。

与 LangGraph 解耦：不依赖 graph 的节点调度，只吃 AgentState、吐 AgentState。
依赖通过可选 kwargs 注入（方便测试 monkeypatch 到 graph 模块上生效）。
"""

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage

from app.agent.state import AgentState
from app.common.constants import DOC_TOP_K, MAX_IMAGES

logger = logging.getLogger(__name__)

# ---- 常量（复制自 graph.py，一字不差） ----

# 附件摘要提示词
_ATTACHMENT_SUMMARY_PROMPT = """下面是一张图片的文字识别结果或一段短文档正文。
用 1-2 句话说清楚"这是什么"，保留关键内容，去掉冗余格式标记。
直接输出摘要正文，不要加前缀。50 字以内。

<content>
{content}
</content>"""

# 附件摘要写入 history 时的前缀标记
_ATTACHMENT_SUMMARY_PREFIX = "[附件摘要] "

# 图片正文在 summarization 后保留的下限（字符数）
_ATTACHMENT_SUMMARY_MIN_CHARS = 120

# 压缩后写入 history 的上限（字符数）
_ATTACHMENT_SUMMARY_MAX_CHARS = 120

# PDF/Word 短文档阈值（字符数）
_DOC_SHORT_CHARS = 2000

# 微信 / Windows 传来的图片经常没有 MIME，只能靠扩展名兜底
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")


def _looks_like_image(item: dict[str, Any]) -> bool:
    """判断附件是不是图片：先看 content-type，缺失时退回看文件名后缀。"""
    if str(item.get("type", "")).startswith("image/"):
        return True
    return str(item.get("name", "")).lower().endswith(_IMAGE_SUFFIXES)


async def _summarize_attachment(content: str, get_chat_model: Callable) -> str | None:
    """给附件正文做一次轻量压缩（1-2 句），结果进 history 供追问用。"""
    if len(content.strip()) < _ATTACHMENT_SUMMARY_MIN_CHARS:
        return None
    try:
        m = get_chat_model(temperature=0.0, max_tokens=300)
        resp = await m.ainvoke(
            [HumanMessage(content=_ATTACHMENT_SUMMARY_PROMPT.format(content=content[:4000]))]
        )
        text = (resp.content or "").strip()
        if text:
            return text[:_ATTACHMENT_SUMMARY_MAX_CHARS]
    except Exception:
        logger.exception("附件摘要失败（长度=%d），跳过", len(content))
    return None


async def _document_blocks(
    documents: list[dict[str, Any]],
    query: str,
    embed_query: Callable,
    search_chunks: Callable,
) -> list[str]:
    """把和这次提问最相关的几块拼进上下文。复制自 graph._document_blocks。"""
    try:
        vector = await embed_query(query or "这份文档的主要内容")
    except Exception:
        logger.exception("文档检索：查询向量化失败")
        names = "、".join(str(d.get("name") or "文档") for d in documents)
        return [f"- {names}：正文检索失败，这次读不到内容，请如实告知用户"]

    blocks: list[str] = []
    for item in documents:
        name = str(item.get("name") or "文档")
        try:
            chunks = await search_chunks(str(item["url"]), vector, DOC_TOP_K)
        except Exception:
            logger.exception("文档检索失败：%s", name)
            chunks = []

        if not chunks:
            blocks.append(
                f"- {name}：没有检索到与这次提问相关的内容"
                "（可能解析失败，也可能文档里确实没有），请如实说明，不要猜测"
            )
            continue
        blocks.append(f"- {name} 的相关段落：\n" + "\n---\n".join(chunks))
    return blocks


async def preprocess(
    state: AgentState,
    *,
    describe_image: Callable | None = None,
    get_all_chunks: Callable | None = None,
    is_document: Callable | None = None,
    embed_query: Callable | None = None,
    search_chunks: Callable | None = None,
    get_chat_model: Callable | None = None,
) -> AgentState:
    """同步预处理层：附件识别 + summarization + 长文档向量检索 + doc 档追问。

    Args:
        state: LangGraph AgentState。
        describe_image: 覆盖默认视觉模型（测试 monkeypatch graph_module.describe_image 时用）。
        get_all_chunks: 覆盖默认文档 chunk 获取函数。
        is_document: 覆盖默认文档类型判断函数。
        embed_query: 覆盖默认查询向量化函数。
        search_chunks: 覆盖默认 chunk 检索函数。
        get_chat_model: 覆盖默认 LLM 工厂（附件摘要用）。
    """
    # ---- 默认依赖 ----
    if describe_image is None:
        from app.llm.vision import describe_image as _describe_image

        describe_image = _describe_image
    if get_all_chunks is None:
        from app.rag.doc_store import get_all_chunks as _get_all_chunks

        get_all_chunks = _get_all_chunks
    if is_document is None:
        from app.rag.doc_ingest import is_document as _is_document

        is_document = _is_document
    if embed_query is None:
        from app.rag.embedding import embed_query as _embed_query

        embed_query = _embed_query
    if search_chunks is None:
        from app.rag.doc_store import search_chunks as _search_chunks

        search_chunks = _search_chunks
    if get_chat_model is None:
        from app.llm.gateway import get_chat_model as _get_chat_model

        get_chat_model = _get_chat_model

    attachments = state.get("attachments") or []

    images: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    for item in attachments:
        if _looks_like_image(item) and item.get("url"):
            images.append(item)
        elif item.get("url") and is_document(
            str(item.get("type") or ""), str(item.get("name") or "")
        ):
            documents.append(item)
        else:
            unsupported.append(item)

    blocks: list[str] = []
    new_doc_urls: list[dict[str, Any]] = []
    doc_summaries: list[str] = []

    # ---- 文档处理 ----
    # 本轮带了文档 → 记住地址（供后续追问重新检索）
    if documents:
        new_doc_urls = [
            {"name": str(d.get("name") or "文档"), "url": str(d.get("url"))} for d in documents
        ]
    elif not attachments and state.get("mode") == "doc":
        # 本轮没带附件，但会话里之前传过文档 → 用这次的问题重新检索一遍（追问）
        stored = state.get("doc_urls") or []
        if stored:
            documents = stored
            logger.info("preprocess：doc 档追问：复用会话内 %d 份文档重新检索", len(stored))

    # 区分「本轮新上传的文档」和「追问复用的旧文档」
    prev_urls = {str(d.get("url")) for d in (state.get("doc_urls") or [])}
    new_urls_set = {str(d.get("url")) for d in new_doc_urls if str(d.get("url")) not in prev_urls}

    for doc in documents:
        url = str(doc.get("url"))
        name = str(doc.get("name") or "文档")
        try:
            chunks = await get_all_chunks(url)
        except Exception:
            logger.exception("preprocess：短文档检查失败：%s", name)
            chunks = []

        if chunks:
            full_text = "\n\n".join(chunks)
            if len(full_text) < _DOC_SHORT_CHARS:
                # 短文档：全文当轮注入（比向量检索 4 块不丢上下文）
                blocks.append(f"- {name} 全文：\n{full_text}")
                # 新上传的做 summarization 进 history（追问复用的已做过）
                if url in new_urls_set:
                    summary = await _summarize_attachment(full_text, get_chat_model)
                    if summary:
                        doc_summaries.append(f"{_ATTACHMENT_SUMMARY_PREFIX}{summary}")
                    elif len(full_text.strip()) >= _ATTACHMENT_SUMMARY_MIN_CHARS:
                        doc_summaries.append(
                            f"{_ATTACHMENT_SUMMARY_PREFIX}{full_text[:_ATTACHMENT_SUMMARY_MAX_CHARS]}"
                        )
                    else:
                        # 太短跳过压缩但也要进 history（否则追问失忆）
                        doc_summaries.append(
                            f"{_ATTACHMENT_SUMMARY_PREFIX}{full_text[:_ATTACHMENT_SUMMARY_MAX_CHARS]}"
                        )
            else:
                # 长文档：向量检索 top 4 块
                blocks.extend(
                    await _document_blocks(
                        [doc], state.get("query") or "", embed_query, search_chunks
                    )
                )
        else:
            # Milvus 里没这份文档的 chunks —— 退回到向量检索让它自己报"没检到"
            blocks.extend(
                await _document_blocks([doc], state.get("query") or "", embed_query, search_chunks)
            )

    # ---- 不支持的附件 ----
    if unsupported:
        names = "、".join(str(s.get("name") or "未知文件") for s in unsupported)
        blocks.append(f"（附件 {names} 暂不支持解析，如被问到请如实说明）")
        if any(not s.get("url") for s in unsupported):
            logger.warning(
                "preprocess：附件缺 url（前端未走 /assistant/upload？）：%s",
                [{"name": s.get("name"), "type": s.get("type")} for s in unsupported],
            )

    # ---- 图片：视觉模型并发识别 ----
    vision_tokens = 0
    image_summaries: list[str] = []

    if images:
        # 超出上限的图片明确说明，而不是静默丢掉
        overflow = images[MAX_IMAGES:]
        images = images[:MAX_IMAGES]
        if overflow:
            names = "、".join(str(o.get("name") or "图片") for o in overflow)
            blocks.append(f"（图片 {names} 超出单轮 {MAX_IMAGES} 张上限，本轮未读取）")

        results = await asyncio.gather(
            *(describe_image(item["url"]) for item in images), return_exceptions=True
        )

        for item, result in zip(images, results, strict=True):
            name = item.get("name") or "图片"
            if isinstance(result, BaseException):
                logger.warning("preprocess：图片识别失败：%s（%s）", name, result)
                blocks.append(f"- {name}：识别失败，未获取到内容")
                continue
            text, tokens = result
            vision_tokens += tokens
            if not text:
                blocks.append(f"- {name}：没有识别到可用内容")
            else:
                blocks.append(f"- {name}：\n{text}")
                summary = await _summarize_attachment(text, get_chat_model)
                if summary:
                    image_summaries.append(f"{_ATTACHMENT_SUMMARY_PREFIX}{summary}")
                elif len(text) < _ATTACHMENT_SUMMARY_MIN_CHARS and text.strip():
                    # 太短跳过压缩但也要进 history（否则追问失忆）
                    image_summaries.append(
                        f"{_ATTACHMENT_SUMMARY_PREFIX}{text[:_ATTACHMENT_SUMMARY_MAX_CHARS]}"
                    )

    # ---- 组装更新 ----
    # 显式设置 attachment_text / vision_tokens（覆盖 checkpointer 可能残留的上一轮值）
    attachment_text = "\n".join(blocks) if blocks else ""

    new_state: AgentState = dict(state)
    new_state["attachment_text"] = attachment_text
    new_state["vision_tokens"] = vision_tokens

    # 摘要进 messages —— 让后续轮次自然带着（checkpointer 持久化）
    all_summaries = [*image_summaries, *doc_summaries]
    if all_summaries:
        existing_messages = list(state.get("messages") or [])
        existing_messages.append(HumanMessage(content="\n".join(all_summaries)))
        new_state["messages"] = existing_messages
        logger.info(
            "preprocess：附件摘要已进 history — 图片 %d 张 + 文档 %d 份",
            len(image_summaries),
            len(doc_summaries),
        )

    # 新上传的文档地址存入 doc_urls（供后续追问重新检索）
    if new_doc_urls:
        existing_urls = list(state.get("doc_urls") or [])
        seen = {str(d.get("url")) for d in existing_urls}
        for item in new_doc_urls:
            if item["url"] not in seen:
                existing_urls.append(item)
                seen.add(item["url"])
        new_state["doc_urls"] = existing_urls

    logger.info(
        "preprocess 完成：图片 %d + 文档 %d + 不支持 %d，vision_tokens=%d",
        len(images),
        len(documents),
        len(unsupported),
        vision_tokens,
    )
    return new_state
