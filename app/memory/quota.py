"""每日 token 配额：按用户统计，超限直接拒绝。

设计取舍：
- **按「天」而不是按「月」**：日配额能及时止损，月配额到月底才发现就晚了。
- **预检 + 事后结算**：入口只能看到「已用多少」，本轮真实消耗要等模型返回才知道。
  所以单轮可能略微超出配额（最多超一轮的量），这是行业惯例，不追求精确拦截。
- **记在 MySQL 而不是内存**：多进程/重启后不清零；一张日聚合表，UPSERT 一条语句搞定。
"""
import logging
from dataclasses import dataclass

from sqlalchemy import text

from app.common.exceptions import QuotaExceededError
from app.config import get_settings
from app.database import get_agent_engine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuotaState:
    """某用户当天的用量快照。"""

    user_key: str
    used: int
    limit: int

    @property
    def exceeded(self) -> bool:
        return self.used >= self.limit

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)


async def get_quota(user_key: str) -> QuotaState:
    """查当天已用 token。没有记录就是 0。"""
    limit = get_settings().daily_token_quota
    sql = text(
        "SELECT total_tokens FROM tb_token_usage "
        "WHERE user_key = :key AND usage_date = CURDATE()"
    )
    async with get_agent_engine().connect() as conn:
        row = (await conn.execute(sql, {"key": user_key})).first()
    return QuotaState(user_key=user_key, used=int(row[0]) if row else 0, limit=limit)


async def add_usage(user_key: str, tokens: int) -> None:
    """累加当天用量。一条 UPSERT，靠唯一键 (user_key, usage_date) 合并。"""
    if tokens <= 0:
        return
    sql = text(
        "INSERT INTO tb_token_usage (user_key, usage_date, total_tokens, request_count) "
        "VALUES (:key, CURDATE(), :tokens, 1) "
        "ON DUPLICATE KEY UPDATE "
        "  total_tokens = total_tokens + VALUES(total_tokens), "
        "  request_count = request_count + 1"
    )
    async with get_agent_engine().begin() as conn:
        await conn.execute(sql, {"key": user_key, "tokens": tokens})
    logger.info("用量 +%d tokens（%s）", tokens, user_key)


async def consume_upload(user_key: str) -> None:
    """上传次数配额：每人每天最多 DAILY_UPLOAD_LIMIT 次。

    先查后加，两次并发可能少拦一次，但不会多拦——宁可放行也不误伤。
    """
    limit = get_settings().daily_upload_limit
    sql_count = text(
        "SELECT upload_count FROM tb_upload_usage "
        "WHERE user_key = :key AND usage_date = CURDATE()"
    )
    async with get_agent_engine().begin() as conn:
        row = (await conn.execute(sql_count, {"key": user_key})).first()
        used = int(row[0]) if row else 0
        if used >= limit:
            raise QuotaExceededError(f"每天最多上传 {limit} 个文件，明天再来")
        await conn.execute(
            text(
                "INSERT INTO tb_upload_usage (user_key, usage_date, upload_count) "
                "VALUES (:key, CURDATE(), 1) "
                "ON DUPLICATE KEY UPDATE upload_count = upload_count + 1"
            ),
            {"key": user_key},
        )
    logger.info("上传 +1（%s，今天第 %d/%d 次）", user_key, used + 1, limit)
