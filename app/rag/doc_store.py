"""附件切块向量库（Milvus Lite，进程内、落本地文件）。

为什么附件单独一个库，不和校园库（numpy 索引）混：
    两者的生命周期完全不同。校园库是 579 条固定数据、只读、全量重建；
    附件块是随上传增长、按文件增删、**只能被上传者自己搜到**。
    混在一起的话，问「图书馆几点开门」会检索到某份讲义里的句子。

为什么用 Milvus Lite 而不是纯 numpy：
    附件块要按 file_url 做标量过滤 + 按文件整批删除，这正是向量库擅长、
    手写 numpy 需要「全量扫 + 重建矩阵文件」才能做到的。而 Milvus Lite
    是进程内的，不需要起任何服务，也不需要那台 192.168.100.128 的虚拟机。

MilvusClient 是同步 API，全部包在 asyncio.to_thread 里——一次插入几百条
向量要几十毫秒，直接在事件循环里跑会卡住同进程的其它请求。
"""
import asyncio
import logging
from pathlib import Path

from pymilvus import MilvusClient

from app.config import get_settings
from app.rag.chunker import Chunk

logger = logging.getLogger(__name__)

COLLECTION = "doc_chunks"

_client: MilvusClient | None = None
# Milvus Lite 是单文件库，写入串行化，避免两条上传同时改同一个文件。
# 按 loop 惰性创建：模块级 asyncio.Lock 换 loop 会永久挂起（同 graph/vision 的教训）。
_write_lock: asyncio.Lock | None = None
_write_lock_loop: asyncio.AbstractEventLoop | None = None


def _write_guard() -> asyncio.Lock:
    global _write_lock, _write_lock_loop
    loop = asyncio.get_running_loop()
    if _write_lock is None or _write_lock_loop is not loop:
        _write_lock = asyncio.Lock()
        _write_lock_loop = loop
    return _write_lock


def _get_client() -> MilvusClient:
    """懒加载：首次用到才打开本地文件（服务启动不碰它）。"""
    global _client
    if _client is None:
        path = get_settings().milvus_lite_path
        # Milvus Lite 不会自动建目录，父目录不存在时直接报
        # "Open local milvus failed, dir ... not exists"——而且要到第一次传文档才暴露。
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        _client = MilvusClient(path)
        logger.info("Milvus Lite 已打开：%s", path)
    return _client


def _ensure_collection(client: MilvusClient, dim: int) -> None:
    if client.has_collection(COLLECTION):
        # 已存在时确认维度一致：换了 embedding 模型必须重建，否则检索全是垃圾。
        # 不按下标取字段——schema 里 id 和 vector 的先后顺序是 pymilvus 决定的，
        # 按「哪个字段带 dim」来找才稳。
        fields = client.describe_collection(COLLECTION).get("fields", [])
        existing = next(
            (int(f["params"]["dim"]) for f in fields if "dim" in (f.get("params") or {})),
            0,
        )
        if existing and existing != dim:
            raise ValueError(
                f"附件索引维度不一致（库内 {existing}，当前 {dim}），换了 embedding 模型需要重建"
            )
        return
    client.create_collection(
        collection_name=COLLECTION,
        dimension=dim,
        metric_type="COSINE",   # 与校园库的余弦口径保持一致
        auto_id=True,
        enable_dynamic_field=True,  # file_url / text 等放在动态字段里
    )
    logger.info("已创建 Milvus 集合 %s（dim=%d）", COLLECTION, dim)


def _ensure_loaded(client: MilvusClient) -> None:
    """确保集合已 load 进内存，否则 search 会报 collection not loaded。

    get_load_state 在 pymilvus 3.x 返回的是 dict（{'state': LoadState.Loaded}），
    老版本是带 .state 的对象——两种都兼容，避免升级客户端就崩。
    """
    raw = client.get_load_state(collection_name=COLLECTION)
    state = raw.get("state") if isinstance(raw, dict) else getattr(raw, "state", raw)
    if str(getattr(state, "name", state)).upper().endswith("LOADED"):
        return
    client.load_collection(COLLECTION)


def _escape(value: str) -> str:
    """过滤表达式里的字符串转义，避免 URL 里的引号把表达式拼坏。"""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _as_chunk(item) -> Chunk:
    """兼容两种入参：Chunk 对象，或纯字符串（老调用方）。"""
    return item if isinstance(item, Chunk) else Chunk(text=str(item))


async def upsert_chunks(
    file_url: str,
    file_name: str,
    chunks: list[Chunk] | list[str],
    vectors: list[list[float]],
    image_texts: dict[int, str] | None = None,
) -> int:
    """写入一份文件的全部块。返回写入条数。

    先按 file_url 删掉旧块再插入：同一份文件重复上传（或重试）时不会残留
    两套向量，语义上和「整份文件覆盖」一致。

    image_texts: {块序号: 视觉模型对该块所属图片的描述}。**单独一个字段，不拼进 text**——
        图片描述不该参与向量计算（否则改一次 OCR 就得重算全部向量），
        但检索命中后要一起带给模型，所以存下来、在 search_chunks 里拼。

    page / kind 同理：是随块携带的标签，不参与向量。page=0 表示「这份文档没有页码概念」
    （docx、扫描件），不是第 0 页。
    """
    if not chunks:
        return 0
    if len(chunks) != len(vectors):
        raise ValueError(f"块数({len(chunks)})与向量数({len(vectors)})不一致")

    image_texts = image_texts or {}
    items = [_as_chunk(c) for c in chunks]
    dim = len(vectors[0])

    def _work() -> int:
        client = _get_client()
        _ensure_collection(client, dim)
        client.delete(COLLECTION, filter=f'file_url == "{_escape(file_url)}"')
        rows = [
            {
                "vector": vectors[i],
                "file_url": file_url,
                "file_name": file_name,
                "chunk_index": i,
                "text": item.text,
                "page": item.page or 0,
                "kind": item.kind,
                "image_text": image_texts.get(i, ""),
            }
            for i, item in enumerate(items)
        ]
        client.insert(COLLECTION, rows)
        client.flush(COLLECTION)
        return len(rows)

    async with _write_guard():
        count = await asyncio.to_thread(_work)
    logger.info("附件入索引：%s（%d 块）", file_name, count)
    return count


async def search_chunks(file_url: str, vector: list[float], top_k: int) -> list[str]:
    """在一份文件内部检索最相关的块。文件没索引过时返回空列表。

    拼进 prompt 的顺序：页码前缀 + 正文 + 图片描述。
        page 是字段而不是文本内容（文本里已经没有「第 N 页」了），
        所以这里补回成「（第 3 页）」，模型才能说「在你文档第 3 页」。
        image_text 同理：不参与向量，但属于同一块的上下文。
    """

    def _work() -> list[str]:
        client = _get_client()
        if not client.has_collection(COLLECTION):
            return []
        _ensure_loaded(client)
        hits = client.search(
            COLLECTION,
            data=[vector],
            limit=top_k,
            filter=f'file_url == "{_escape(file_url)}"',
            output_fields=["text", "chunk_index", "page", "kind", "image_text"],
        )
        if not hits:
            return []
        # 按块序还原，让拼进 prompt 的段落顺序和原文一致
        ordered = sorted(hits[0], key=lambda h: h["entity"]["chunk_index"])
        out: list[str] = []
        for hit in ordered:
            entity = hit["entity"]
            text = str(entity.get("text") or "")
            page = int(entity.get("page") or 0)
            if page:
                text = f"（第 {page} 页）{text}"
            image_text = str(entity.get("image_text") or "").strip()
            if image_text:
                text = f"{text}\n【该块包含的图片】{image_text}"
            out.append(text)
        return out

    return await asyncio.to_thread(_work)


async def delete_file(file_url: str) -> None:
    """删掉一份文件的全部块（会话删除 / 重新上传时用）。"""

    def _work() -> None:
        client = _get_client()
        if client.has_collection(COLLECTION):
            client.delete(COLLECTION, filter=f'file_url == "{_escape(file_url)}"')

    async with _write_guard():
        await asyncio.to_thread(_work)


def aclose() -> None:
    """关闭本地库（应用退出时调用）。"""
    global _client
    if _client is not None:
        _client.close()
        _client = None
