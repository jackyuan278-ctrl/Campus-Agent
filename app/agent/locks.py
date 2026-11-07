"""同一会话的串行锁。

为什么需要：
    checkpointer 是「读状态 → 执行 → 写回」的 last-write-wins 模型，没有版本冲突
    检测。同一个 thread_id 并发发起两个请求时，两边都读到同一份旧状态，各自跑完
    再写回，后写的把先写的整段覆盖掉。

    实测（MySQL checkpointer，同一会话 4 并发）：4 个请求全部「成功」，但 checkpoint
    里最终只剩 2 条消息（理想 8 条）。消息不是丢失报错，是**静默消失**——用户发现
    「我刚才说的话它不记得了」，且日志里一点异常都没有。

    所以同一会话的请求必须在应用层排队。跨会话不受影响，仍然并行。

局限：
    这是**进程内**锁。uvicorn 开多 worker 时不跨进程，需要 Redis 锁。当前部署是
    单 worker，够用；将来要多实例，把这里换成分布式锁即可，调用方不用改。

    ⚠️ **不可重入**：同一个请求里不要重复进入 thread_lock(同一个 thread_id)，
    asyncio.Lock 不可重入，会直接死锁。
"""
import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# 排队超过这个时长才打日志：正常情况（无并发）不该刷屏
_QUEUE_LOG_THRESHOLD = 0.05

_locks: dict[str, asyncio.Lock] = {}
# 引用计数：会话结束后把锁丢掉，否则长跑服务的 _locks 会随会话数无限增长
_refs: dict[str, int] = {}
# 保护上面两个 dict 的锁。**必须按 loop 惰性创建**：模块级的 asyncio.Lock 绑定
# 首次使用它的事件循环，换一个 loop（pytest 每个用例一个 loop、进程重启、
# 某些框架的 fork）再申请会永久挂起且不报错。
_guard: asyncio.Lock | None = None
_guard_loop: asyncio.AbstractEventLoop | None = None


def _get_guard() -> asyncio.Lock:
    global _guard, _guard_loop
    loop = asyncio.get_running_loop()
    if _guard is None or _guard_loop is not loop:
        _guard = asyncio.Lock()
        _guard_loop = loop
    return _guard


@asynccontextmanager
async def thread_lock(thread_id: str) -> AsyncIterator[None]:
    """持有某个会话的独占锁；同一会话的其它请求在此排队。"""
    async with _get_guard():
        lock = _locks.get(thread_id)
        if lock is None:
            lock = asyncio.Lock()
            _locks[thread_id] = lock
        _refs[thread_id] = _refs.get(thread_id, 0) + 1

    try:
        started = time.perf_counter()
        async with lock:
            # 用实际耗时判断是否排过队。之前用 lock.locked() 预判，但那一刻
            # 前一个请求可能还没真正 acquire，会漏报（A1）。
            waited = time.perf_counter() - started
            if waited >= _QUEUE_LOG_THRESHOLD:
                logger.info("会话 %s 排队 %.0f ms 后才开始处理", thread_id, waited * 1000)
            yield
    finally:
        async with _get_guard():
            _refs[thread_id] -= 1
            if _refs[thread_id] <= 0:
                _locks.pop(thread_id, None)
                _refs.pop(thread_id, None)
