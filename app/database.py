"""数据库连接：只读连接 CampusHub 的三个业务库。

设计原则：
- Agent 只读，不写业务数据（活动/社区/商品都归 CampusHub 管）
- 每个库一个 Engine，按需创建、进程内缓存复用
- 配置不在 import 时固化：每次取 get_settings()，避免测试/改配置后拿到旧值
- 连接池由 main.py 的 lifespan 在应用退出时统一 dispose
"""
import logging

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from app.config import get_settings

logger = logging.getLogger(__name__)

# 库名 -> Engine（懒加载，避免启动即连库失败）
_engines: dict[str, Engine] = {}


def get_engine(db_name: str) -> Engine:
    """获取指定库的 Engine，首次调用时创建并缓存。"""
    if db_name not in _engines:
        settings = get_settings()
        logger.info("创建数据库连接池：%s@%s/%s", settings.mysql_user, settings.mysql_host, db_name)
        _engines[db_name] = create_engine(
            settings.db_url(db_name),
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False,
        )
    return _engines[db_name]


def get_activity_engine() -> Engine:
    """活动库。"""
    return get_engine(get_settings().mysql_db_activity)


def get_community_engine() -> Engine:
    """社区库。"""
    return get_engine(get_settings().mysql_db_community)


def get_secondhand_engine() -> Engine:
    """二手交易库。"""
    return get_engine(get_settings().mysql_db_secondhand)


def dispose_all() -> None:
    """关闭并清理所有连接池（应用退出时调用）。"""
    for db_name, engine in _engines.items():
        engine.dispose()
        logger.info("已关闭数据库连接池：%s", db_name)
    _engines.clear()
