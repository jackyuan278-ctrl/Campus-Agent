"""单条索引更新：MQ 消费者调这里，也可以手动 `python -m app.rag.indexer` 调试。

以查库结果为准，所以天然幂等：

    收到 (source, id) → 查库
      ├─ 查到且在架上 → 删旧行 + 算新向量 + 追加
      └─ 查不到 / 已下架 → 只删旧行

这样每条消息只花 1 次 embedding（删除是 0 次），不需要重算整个索引。
"""
import argparse
import asyncio
import logging

from app.database import dispose_all
from app.rag.embedding import aclose as close_embedding
from app.rag.embedding import embed_texts
from app.rag.index import VectorIndex
from app.rag.ingest import fingerprint, load_activity, load_question

logger = logging.getLogger(__name__)

# 支持的数据源。campus-hub 发消息时 source 要传这两个值之一。
LOADERS = {
    "activity": load_activity,
    "question": load_question,
}


async def reindex_one(source: str, doc_id: int) -> str:
    """重新索引一条文档。

    Returns:
        "upserted" 新增/更新；"deleted" 已下架或不存在；"skipped" 无法处理（未知 source / 索引缺失）
    """
    loader = LOADERS.get(source)
    if loader is None:
        logger.warning("未知的 source=%r（只支持 %s），忽略", source, list(LOADERS))
        return "skipped"

    index = VectorIndex.load()
    if index is None:
        logger.warning("索引文件不存在，跳过单条更新（先跑一次全量 ingest 建基线）")
        return "skipped"

    doc = await loader(doc_id)
    trimmed = index.without(source, doc_id)

    if doc is None:
        trimmed.save()
        logger.info("索引删除 %s:%s（已下架或记录不存在）", source, doc_id)
        return "deleted"

    # 先按内容指纹删一遍：库里可能已有一条内容完全相同的旧文档，
    # 不删的话增量会重新制造出重复向量。
    fp = fingerprint(doc["text"])
    trimmed = trimmed.without_fingerprint(fp)

    doc["fingerprint"] = fp
    vector = (await embed_texts([doc["text"]]))[0]
    trimmed.append(doc, vector).save()
    logger.info("索引更新 %s:%s → %s", source, doc_id, (doc.get("title") or "")[:24])
    return "upserted"


async def _main(source: str, doc_id: int) -> None:
    try:
        action = await reindex_one(source, doc_id)
        print(f"✅ {source}:{doc_id} → {action}")
    finally:
        await close_embedding()
        await dispose_all()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="单条重建 RAG 向量（调试用）")
    parser.add_argument("source", choices=list(LOADERS), help="数据源")
    parser.add_argument("doc_id", type=int, help="业务 id")
    args = parser.parse_args()
    asyncio.run(_main(args.source, args.doc_id))
