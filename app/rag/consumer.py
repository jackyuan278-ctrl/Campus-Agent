"""RabbitMQ 索引消费者：campus-hub 增删改后发一条消息，这里重算单条向量。

启动（独立进程，和 API 分开跑）：
    uv run python -m app.rag.consumer

消息格式（JSON，发到默认 exchange，routing key = 队列名）：
    {"source": "activity",  "id": 9100000000000002001}
    {"source": "question",  "id": 9200000000000002001}

Java 侧（campus-hub）发消息示例：
    rabbitTemplate.convertAndSend(
        "campus.index.rebuild",
        objectMapper.writeValueAsString(Map.of("source", "activity", "id", activityId)));

设计要点：
1. **一条消息只带 id，不带内容**。消费者自己去查库，以库里当前状态为准，
   所以增、删、改三条路径合成一条，天然幂等（消息重复、乱序都不怕）。
2. **prefetch_count=1，串行处理**。索引文件（vectors.npy + docs.json）不是并发
   安全的，两条消息同时改会互相覆盖。
3. **失败不无限重试**：第一次失败 nack 重新入队，第二次（redelivered）还失败就
   reject 丢弃——宁可丢一条消息（事后跑一次全量 ingest 就能补），也不要卡住队列。
"""
import asyncio
import json
import logging

import aio_pika

from app.config import get_settings
from app.database import dispose_all
from app.rag.embedding import aclose as close_embedding
from app.rag.indexer import reindex_one

logger = logging.getLogger(__name__)


async def _handle(body: bytes) -> str:
    """处理一条消息，返回动作描述。格式错误抛 ValueError（不重试）。"""
    try:
        data = json.loads(body)
        source = str(data["source"])
        doc_id = int(data["id"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"消息格式不对，需要 {{'source':..., 'id':...}}：{body[:120]!r}") from exc
    return await reindex_one(source, doc_id)


async def consume() -> None:
    settings = get_settings()
    if not settings.rabbitmq_configured:
        raise RuntimeError("未配置 RABBITMQ_URL，无法启动索引消费者")

    queue_name = settings.rabbitmq_index_queue
    logger.info("连接 RabbitMQ 并监听队列 %s", queue_name)
    connection = await aio_pika.connect_robust(settings.rabbitmq_url)

    async with connection:
        channel = await connection.channel()
        # 索引文件不是并发安全的，必须一条一条处理
        await channel.set_qos(prefetch_count=1)
        queue = await channel.declare_queue(queue_name, durable=True)
        logger.info("索引消费者已就绪，等待消息…")

        async for message in queue:
            # requeue=True：异常时重新入队；reject_on_redelivered=True：重投一次还失败就丢弃
            try:
                async with message.process(requeue=True, reject_on_redelivered=True):
                    action = await _handle(message.body)
                    logger.info("处理完成：%s", action)
            except ValueError as exc:
                # 格式错误重试也没用，直接吃掉这条消息
                logger.error("丢弃非法消息：%s", exc)
            except Exception:
                logger.exception("处理消息失败，已放回队列（重投仍失败则丢弃）")


async def _main() -> None:
    try:
        await consume()
    finally:
        await close_embedding()
        await dispose_all()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("消费者已停止")
