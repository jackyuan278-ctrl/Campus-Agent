"""会话记忆层：Redis 存多轮对话上下文。"""
import logging
import json
import redis.asyncio as redis_async

from app.common.constants import REDIS_HISTORY_TTL, REDIS_MAX_HISTORY, REDIS_KEY_PREFIX
from app.config import get_settings

logger = logging.getLogger(__name__)

redis_client: redis_async.Redis | None = None

def get_redis() -> redis_async.Redis:
    global redis_client
    if redis_client is None:
        redis_client = redis_async.Redis.from_url(
            get_settings().redis_url,
            decode_responses=True
        )
    return redis_client

class MemoryStore:
    """会话记忆：Redis List存储对话消息，异步版本"""
    def __init__(self) -> None:
        self._settings = get_settings()
        self._redis = get_redis()

    async def get_history(self, session_id: str) -> list[dict[str, str]]:
        """获取会话历史"""
        key = f"{REDIS_KEY_PREFIX}:{session_id}:messages"
        raw_list = await self._redis.lrange(key, 0, -1)
        return [json.loads(item) for item in raw_list]

    async def append(self, session_id: str, role: str, content: str) -> None:
        """追加一条消息，自动截断、设置过期"""
        key = f"{REDIS_KEY_PREFIX}:{session_id}:messages"
        data = {"role": role, "content": content}
        msg = json.dumps(data, ensure_ascii=False)
        async with self._redis.pipeline() as pipe:
            pipe.rpush(key, msg)
            pipe.ltrim(key, -REDIS_MAX_HISTORY, -1)
            pipe.expire(key, REDIS_HISTORY_TTL)
            await pipe.execute()

    async def clear(self, session_id: str) -> None:
        """清空指定会话的历史"""
        key = f"{REDIS_KEY_PREFIX}:{session_id}:messages"
        await self._redis.delete(key)
