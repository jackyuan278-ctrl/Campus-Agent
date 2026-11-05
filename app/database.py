"""数据库连接：只读连接 CampusHub 的三个业务库。

设计原则：
- Agent 只读，不写业务数据（活动/社区/商品都归 CampusHub 管）
- 每个库一个 AsyncEngine，按需创建、进程内缓存复用
- 全异步：驱动 aiomysql，与 FastAPI / Redis 链路保持一致
- 配置不在 import 时固化：每次取 get_settings()，避免测试/改配置后拿到旧值
- 连接池由 main.py 的 lifespan 在应用退出时统一 dispose（注意是 await）
"""
import logging
import threading

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.config import get_settings

logger = logging.getLogger(__name__)

# 库名 -> AsyncEngine（懒加载，避免启动即连库失败）
_engines: dict[str, AsyncEngine] = {}
# 双检锁：协程里的同步函数本身不会被打断，但 sync 端点跑在线程池里，
# 两个线程同时首次取同一个库会各建一个 engine，其中一个永远不 dispose（连接泄漏）。
_lock = threading.Lock()


def get_engine(db_name: str) -> AsyncEngine:
    """获取指定库的 AsyncEngine，首次调用时创建并缓存。"""
    engine = _engines.get(db_name)
    if engine is not None:
        return engine
    with _lock:
        if db_name not in _engines:  # 双检：可能别的线程刚建好
            settings = get_settings()
            logger.info(
                "创建数据库连接池：%s@%s/%s", settings.mysql_user, settings.mysql_host, db_name
            )
            _engines[db_name] = create_async_engine(
                settings.db_url(db_name),
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False,
            )
    return _engines[db_name]


def get_activity_engine() -> AsyncEngine:
    """活动库。"""
    return get_engine(get_settings().mysql_db_activity)


def get_community_engine() -> AsyncEngine:
    """社区库。"""
    return get_engine(get_settings().mysql_db_community)


def get_secondhand_engine() -> AsyncEngine:
    """二手交易库。"""
    return get_engine(get_settings().mysql_db_secondhand)


def get_agent_engine() -> AsyncEngine:
    """Agent 自己的库（聊天记录 / 上下文）。"""
    return get_engine(get_settings().mysql_db_agent)


async def dispose_all() -> None:
    """关闭并清理所有连接池（应用退出时调用，异步）。"""
    for db_name, engine in _engines.items():
        await engine.dispose()
        logger.info("已关闭数据库连接池：%s", db_name)
    _engines.clear()
