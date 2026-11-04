"""Embedding 客户端：腾讯 TokenHub（OpenAI 兼容协议）。

注意两个坑：
1. openai SDK 新版默认 `encoding_format="base64"`，TokenHub 只接受 float，必须显式传
2. 单次请求有输入条数/长度上限，所以要分批
"""
import logging

from openai import APIError, AsyncOpenAI

from app.common.exceptions import AppError
from app.config import get_settings

logger = logging.getLogger(__name__)

# 每批送多少条文本（TokenHub 单请求上限内取个保守值）
BATCH_SIZE = 16


class EmbeddingError(AppError):
    """向量化失败。"""

    code = 2002
    status_code = 502
    message = "向量化服务暂时不可用"


_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    """懒建 + 复用连接池。"""
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncOpenAI(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            timeout=30,
            max_retries=2,
        )
        logger.info("创建 embedding 客户端：%s", settings.embedding_base_url)
    return _client


async def aclose() -> None:
    """关闭客户端（应用退出时调用）。"""
    global _client
    if _client is not None:
        await _client.close()
        _client = None


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """把一批文本转成向量，返回顺序与入参一致。"""
    if not texts:
        return []

    settings = get_settings()
    client = _get_client()
    vectors: list[list[float]] = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start:start + BATCH_SIZE]
        try:
            resp = await client.embeddings.create(
                model=settings.embedding_model,
                input=batch,
                encoding_format="float",  # TokenHub 只认 float，默认 base64 会 400
            )
        except APIError as exc:
            logger.error("embedding 调用失败：%s", exc)
            raise EmbeddingError("向量化失败") from exc
        vectors.extend([item.embedding for item in resp.data])

    logger.info("向量化完成：%d 条", len(vectors))
    return vectors


async def embed_query(text: str) -> list[float]:
    """向量化单条查询文本。"""
    vectors = await embed_texts([text])
    if not vectors:
        raise EmbeddingError("查询向量为空")
    return vectors[0]
