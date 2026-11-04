"""会话记忆层：MySQL 持久化多轮对话。

设计原则（重要）：
- **只追加，不删除**：裁剪是「读取策略」，不是「写入动作」。
  库里永远保留全量消息，控制上下文长度靠 SELECT ... LIMIT，绝不 DELETE。
- 一份数据两个视图：最近 N 条当上下文，全量当聊天记录。
- 全异步：复用 app.database 的 AsyncEngine（aiomysql）。
"""
import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.common.constants import CHAT_HISTORY_LIMIT
from app.database import get_agent_engine

logger = logging.getLogger(__name__)


class MemoryStore:
    """会话记忆：读写 campushub_agent.tb_chat_message。"""

    def __init__(self) -> None:
        self._engine: AsyncEngine = get_agent_engine()

    async def append(self, thread_id: str, role: str, content: str) -> None:
        """追加一条消息。只 INSERT，不删任何历史。"""
        if not thread_id:
            raise ValueError("thread_id 不能为空")

        sql = text(
            "INSERT INTO tb_chat_message (thread_id, role, content) "
            "VALUES (:thread_id, :role, :content)"
        )
        async with self._engine.begin() as conn:
            await conn.execute(
                sql,
                {"thread_id": thread_id, "role": role, "content": content},
            )

    async def get_history(
        self,
        thread_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """取最近 N 条消息，按时间正序返回（用于拼上下文）。

        先按 id 倒序取 N 条（走索引），再反转——避免 ORDER BY created_at ASC
        在大表上先扫全量再截断。
        """
        size = limit or CHAT_HISTORY_LIMIT
        sql = text(
            "SELECT role, content, created_at FROM tb_chat_message "
            "WHERE thread_id = :thread_id "
            "ORDER BY id DESC LIMIT :limit"
        )
        async with self._engine.connect() as conn:
            result = await conn.execute(sql, {"thread_id": thread_id, "limit": size})
            rows = result.mappings().all()

        return [
            {
                "role": row["role"],
                "content": row["content"],
                "created_at": str(row["created_at"]),
            }
            for row in reversed(rows)
        ]

    async def list_all(self, thread_id: str) -> list[dict[str, Any]]:
        """取该会话的全部消息（前端展示 / 导出用），按时间正序。

        和 get_history 的区别：这是给人看的全量，不受上下文条数限制。
        """
        sql = text(
            "SELECT id, role, content, created_at FROM tb_chat_message "
            "WHERE thread_id = :thread_id ORDER BY id ASC"
        )
        async with self._engine.connect() as conn:
            result = await conn.execute(sql, {"thread_id": thread_id})
            rows = result.mappings().all()

        return [
            {
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]

    async def clear(self, thread_id: str) -> None:
        """清空指定会话的记录。

        仅用于用户主动清空或测试；正常业务流程里永远不调用。
        """
        async with self._engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM tb_chat_message WHERE thread_id = :thread_id"),
                {"thread_id": thread_id},
            )
