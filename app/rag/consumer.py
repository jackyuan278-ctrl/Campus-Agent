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
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import aio_pika
from sqlalchemy import text

from app.config import get_settings
from app.database import dispose_all, get_agent_engine
from app.rag.embedding import aclose as close_embedding
from app.rag.indexer import reindex_one

logger = logging.getLogger(__name__)

# MySQL 命名锁的名字。同一时刻只允许一个消费者持有。
_LOCK_NAME = "campus_agent_index_consumer"

# 保活间隔。MySQL 的 wait_timeout 默认 8 小时，会把长时间空闲的连接断掉；
# 连接一断，这条连接持有的 GET_LOCK 就随之释放，而进程还在正常消费——
# 单实例保护静默失效，重复启动也不会被拦住。所以定期确认锁还在自己手上。
_LOCK_KEEPALIVE_SECONDS = 30 * 60


async def _keep_lock_alive(conn) -> None:
    """定期确认锁仍由本连接持有；发现丢了就重新抢，抢不到只告警不退出。

    为什么不直接再调一次 GET_LOCK：MySQL 5.7+ 里同一会话重复 GET_LOCK 会
    累加持有计数，需要同样多次 RELEASE_LOCK 才真正释放。先用 IS_USED_LOCK
    对比 CONNECTION_ID，确认没丢就什么都不做。
    """
    while True:
        await asyncio.sleep(_LOCK_KEEPALIVE_SECONDS)
        try:
            row = (
                await conn.execute(
                    text("SELECT IS_USED_LOCK(:name), CONNECTION_ID()"), {"name": _LOCK_NAME}
                )
            ).first()
        except Exception:
            logger.exception("消费者锁保活检查失败，下次再试")
            continue

        holder, own = (row[0], row[1]) if row else (None, None)
        if holder is not None and own is not None and int(holder) == int(own):
            continue

        logger.warning("消费者锁已丢失（holder=%s, 本连接=%s），尝试重新获取", holder, own)
        try:
            got = (
                await conn.execute(
                    text("SELECT GET_LOCK(:name, 0)"), {"name": _LOCK_NAME}
                )
            ).scalar()
        except Exception:
            logger.exception("重新获取消费者锁失败")
            continue
        if got == 1:
            logger.info("已重新获得消费者锁：%s", _LOCK_NAME)
        else:
            # 另一个进程抢先了。不停摆：停了两边都没人消费更糟，这里只提醒排查。
            logger.error("消费者锁被其他进程持有（%s），本进程可能重复消费，请确认是否重复启动", _LOCK_NAME)


@asynccontextmanager
async def _exclusive_lock() -> AsyncIterator[None]:
    """保证同时只有一个消费者进程在跑。

    为什么要这个：uvicorn 开多 worker、或者手滑起了两个进程，同一条消息会被
    处理多次。虽然 reindex_one 是幂等的（结果一样），但每次都要花一次 embedding
    调用——白烧钱。

    用 MySQL 的 GET_LOCK 而不是文件锁：连接断开时 MySQL 会自动释放锁，
    所以进程被 kill -9 也不会留下需要手工清理的死锁。
    """
    engine = get_agent_engine()
    async with engine.connect() as conn:
        got = (
            await conn.execute(text("SELECT GET_LOCK(:name, 0)"), {"name": _LOCK_NAME})
        ).scalar()
        if got != 1:
            raise RuntimeError(
                f"已有消费者持有锁 {_LOCK_NAME}，本进程退出（确认是否重复启动）"
            )
        logger.info("已获得消费者锁：%s", _LOCK_NAME)
        keeper = asyncio.create_task(_keep_lock_alive(conn))
        try:
            yield
        finally:
            keeper.cancel()
            try:
                await conn.execute(text("SELECT RELEASE_LOCK(:name)"), {"name": _LOCK_NAME})
            except Exception:  # noqa: BLE001 - 释放失败不影响退出，连接断开也会自动释放
                logger.warning("释放消费者锁失败（连接断开时会自动释放）")


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

    async with connection, _exclusive_lock():
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
