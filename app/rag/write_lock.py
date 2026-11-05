"""索引写入互斥：跨进程保护「读索引 → 改 → 落盘」这一整段。

为什么需要：索引是「读出来 → 在内存里改 → 整体写回」的模型，两个写入方
（MQ 消费者、手动 `python -m app.rag.ingest`）同时跑时，各自读到同一份旧索引，
后写的那个会把前一个的改动整个盖掉。原子写（见 index.save）只能保证文件
不成半新半旧，救不了这种更新丢失。

为什么用 MySQL 的 GET_LOCK：项目里已经有这个依赖，不需要引新组件；连接断开
时 MySQL 自动释放锁，进程被 kill -9 也不会留下死锁。

注意加锁范围必须包含 embedding 调用——它夹在「读旧索引」和「写新索引」中间，
放到锁外面就又出现两个写入方各自基于旧索引计算的问题了。代价是一次 ingest
期间消费者会排队等待，但两边都是低频操作，可以接受。
"""
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text

from app.database import get_agent_engine

logger = logging.getLogger(__name__)

_LOCK_NAME = "campus_agent_index_write"
# 全量 ingest 要读全表 + 向量化，可能跑十几秒；给足等待时间，
# 让后来者排队而不是直接失败。
_WAIT_SECONDS = 120


@asynccontextmanager
async def index_write_lock(wait_seconds: int = _WAIT_SECONDS) -> AsyncIterator[None]:
    """独占索引写入权；拿不到就抛错，绝不带着「可能覆盖别人」的状态往下走。"""
    engine = get_agent_engine()
    async with engine.connect() as conn:
        got = (
            await conn.execute(
                text("SELECT GET_LOCK(:name, :wait)"),
                {"name": _LOCK_NAME, "wait": wait_seconds},
            )
        ).scalar()
        if got != 1:
            raise RuntimeError(
                f"{wait_seconds}s 内没拿到索引写锁 {_LOCK_NAME}"
                "（另一个 ingest 或消费者正在写索引，稍后重试）"
            )
        logger.debug("获得索引写锁：%s", _LOCK_NAME)
        try:
            yield
        finally:
            try:
                await conn.execute(text("SELECT RELEASE_LOCK(:name)"), {"name": _LOCK_NAME})
            except Exception:  # noqa: BLE001 - 释放失败不影响结果，连接断开也会自动释放
                logger.warning("释放索引写锁失败（连接断开时会自动释放）")
