"""会话记忆层：Redis 存多轮对话上下文（占位）。"""
from app.config import get_settings


class MemoryStore:
    """会话记忆。当前占位，接入 Redis 后实现 get/append/clear。"""

    def __init__(self) -> None:
        self._settings = get_settings()

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        """获取会话历史。"""
        # TODO: 从 Redis 读取 session_id 对应的历史消息

        return []

    def append(self, session_id: str, role: str, content: str) -> None:
        """追加一条消息。"""
        # TODO: 写入 Redis
        pass
