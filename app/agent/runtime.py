"""Agent 运行时：管理 checkpointer 连接 + 编译后的状态图，支持连接自愈。

为什么需要这层：
    AIOMySQLSaver 内部是「一条长连接 + 不重连」。MySQL 的 wait_timeout（默认 8 小时）
    或任意一次网络抖动都会让这条连接失效，之后所有对话都会报
    `InterfaceError: (0, 'Not connected')`，只能重启服务。

做法：
    每次取图前 ping 一次；ping 失败就重建 saver 并重新编译图，再交给调用方。
    图是无状态对象（状态都在 MySQL 里），所以重建图不会丢会话。
"""
import asyncio
import logging
import time
from contextlib import AsyncExitStack
from typing import Any

from langgraph.checkpoint.mysql.aio import AIOMySQLSaver

from app.agent.graph import build_graph

logger = logging.getLogger(__name__)

# 探活结果缓存时长。每个请求都 ping 一次 MySQL 是白白多一个往返，
# 而掉线最坏晚 5 秒被发现——期间失败的请求会走 recover() 重建，代价可接受。
_PING_TTL = 5.0


class AgentRuntime:
    """持有 checkpointer 与编译后的图，负责建立 / 探活 / 重建 / 关闭。"""

    def __init__(self, conn_string: str) -> None:
        self._conn_string = conn_string
        self._stack: AsyncExitStack | None = None
        self._saver: AIOMySQLSaver | None = None
        self._graph: Any = None
        self._lock = asyncio.Lock()
        self._schema_ready = False  # setup() 只需跑一次，重建时跳过
        self._healthy_until = 0.0   # 探活结果的缓存截止时间（monotonic）

    # ---------- 对外只读访问 ----------

    @property
    def saver(self) -> AIOMySQLSaver | None:
        """当前 saver（清空会话时要用它删 checkpoint）。"""
        return self._saver

    @property
    def graph(self) -> Any:
        """当前编译好的图（可能已过期，正式取图请用 get_graph）。"""
        return self._graph

    # ---------- 生命周期 ----------

    async def start(self) -> None:
        async with self._lock:
            await self._rebuild()

    async def aclose(self) -> None:
        async with self._lock:
            await self._close()

    async def get_graph(self) -> Any:
        """取图。健康时走无锁快路径，只有需要重建时才抢锁。

        之前是「每个请求都先拿锁、再 ping」，等于把所有并发请求串行化在一把
        锁上，而它们其实只是想读一下 self._graph 这个引用。
        """
        if self._graph is not None and await self._is_healthy():
            return self._graph

        async with self._lock:
            # 双检：等锁期间别的请求可能已经重建好了
            if self._graph is None or not await self._is_healthy():
                logger.warning("checkpointer 连接不可用，正在重建…")
                await self._rebuild()
            return self._graph

    async def recover(self) -> Any:
        """强制重建（调用方遇到连接类异常时使用）。"""
        async with self._lock:
            logger.warning("检测到连接类异常，强制重建 checkpointer")
            await self._rebuild()
            return self._graph

    # ---------- 内部实现 ----------

    async def _is_healthy(self) -> bool:
        if self._saver is None:
            return False
        now = time.monotonic()
        if now < self._healthy_until:
            return True          # 5 秒内刚探过，不再打一次 MySQL
        try:
            # reconnect=False：只探测，不偷偷换连接（换连接会绕过我们的重建逻辑）
            await self._saver.conn.ping(reconnect=False)
            self._healthy_until = now + _PING_TTL
            return True
        except Exception as exc:  # noqa: BLE001 —— 探活失败的原因不重要，一律视为不健康
            logger.warning("checkpointer ping 失败：%s", exc)
            self._healthy_until = 0.0
            return False

    async def _rebuild(self) -> None:
        await self._close()
        stack = AsyncExitStack()
        try:
            saver = await stack.enter_async_context(
                AIOMySQLSaver.from_conn_string(self._conn_string)
            )
            if not self._schema_ready:
                await saver.setup()  # 建 4 张表（幂等）；重建连接时不需要再跑
                self._schema_ready = True
        except Exception:
            await stack.aclose()
            raise

        self._stack = stack
        self._saver = saver
        self._graph = build_graph(saver)
        logger.info("状态图已就绪（checkpointer=MySQL）")

    async def _close(self) -> None:
        stack, self._stack = self._stack, None
        self._saver = None
        self._graph = None
        self._healthy_until = 0.0
        if stack is not None:
            try:
                await stack.aclose()
            except Exception:
                logger.warning("关闭 checkpointer 连接失败", exc_info=True)
