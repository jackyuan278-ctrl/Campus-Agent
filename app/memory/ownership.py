"""会话归属：解决「知道 thread_id 就能读/删别人会话」的越权问题。

campus-hub 的数据天然挂在 userId 上，campus-agent 的 thread_id 是前端生成的，
必须自己记一份 owner，否则猜到一个 id 就能读、删别人的对话。

规则：第一次用到某个 thread_id 的人就是它的 owner，之后别的用户访问一律 403。
"""
import logging

from sqlalchemy import text

from app.common.exceptions import ForbiddenError
from app.database import get_agent_engine

logger = logging.getLogger(__name__)


async def ensure_owner(thread_id: str, user_key: str) -> None:
    """确认会话属于当前用户；没有记录就先占位，属于别人则抛 403。

    用 INSERT IGNORE 抢占再回读，两个请求并发首次访问同一个新会话也不会炸唯一键。
    """
    async with get_agent_engine().begin() as conn:
        await conn.execute(
            text("INSERT IGNORE INTO tb_thread_owner (thread_id, user_key) VALUES (:t, :k)"),
            {"t": thread_id, "k": user_key},
        )
        row = (await conn.execute(
            text("SELECT user_key FROM tb_thread_owner WHERE thread_id = :t"),
            {"t": thread_id},
        )).first()

    if row is not None and row[0] != user_key:
        logger.warning(
            "会话越权访问被拒绝：thread=%s owner=%s visitor=%s", thread_id, row[0], user_key
        )
        raise ForbiddenError()
