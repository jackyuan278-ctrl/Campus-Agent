"""Post-Processing 层：graph.py agent loop 结束后的同步副作用。

把 persist / cleanup 相关逻辑从 graph.py 节点里抽成独立函数，
方便后续在 chat.py 里脱离 LangGraph 直接调用。

graph.py 的 persist 节点还保留着（状态图闭环不能断），这里是它的镜像逻辑。
"""

import logging
from typing import Any

from app.agent.state import AgentState
from app.memory.quota import add_usage
from app.memory.store import MemoryStore

logger = logging.getLogger(__name__)


async def postprocess(state: AgentState, extras: dict | None = None, *, _store=None) -> None:
    """Agent loop 结束后的后处理：持久化、配额扣减、vision_tokens 落库。

    设计成副作用函数（返回 None）——调用方不需要拿返回值，失败了记日志即可，
    不能因为落库抖动让用户对话超时。

    Args:
        state: agent loop 的最终状态（LangGraph AgentState）。
        extras: 可选的额外参数，可能包含：
            - "vision_tokens": int — 视觉识别消耗的 token（若调用方想单独落库）
            - "user_id": str — 用户 ID（用于写 tb_upload_record 等按用户聚合的表）
            - "thread_id": str — 会话 ID（如果 state 里拿不到的话）
            - "attachments": list — 原始附件列表（落库时可能需要文件名 / URL）
        _store: 可选的 MemoryStore 实例（测试 monkeypatch graph_module.MemoryStore 时通过 graph.py 传入）。
    """
    extras = extras or {}
    thread_id = extras.get("thread_id") or state.get("thread_id")
    user_id = extras.get("user_id")
    attachments = extras.get("attachments") or state.get("attachments") or []

    # ---------- 1. vision_tokens 独立落库（tb_upload_record）----------
    # TODO: 当前 app 目录里没有任何地方写 tb_upload_record。
    #   graph.py 的 image_agent 算出 vision_tokens 后，route 节点把它并入
    #   turn_tokens，再由 persist 节点的 add_usage 一起累加到 tb_token_usage。
    #   AGENTS.md 硬性约束第 7 条说"vision_tokens 独立计费、写入 tb_upload_record.vision_tokens"，
    #   但 upload 接口（/assistant/upload）和 chat 接口都没实现这条。
    #   后续补上 INSERT tb_upload_record 的逻辑时，extras["vision_tokens"] 就是入口。
    vision_tokens = extras.get("vision_tokens", 0) or state.get("vision_tokens", 0)
    if vision_tokens > 0:
        logger.info(
            "vision_tokens 待落库（未实现）：tokens=%d user=%s thread=%s 附件数=%d",
            vision_tokens,
            user_id,
            thread_id,
            len(attachments),
        )

    # ---------- 2. session 业务记录持久化（tb_chat_message）----------
    # 从 graph.py persist 节点原样复制：同一会话写两条——user 和 assistant。
    store = _store if _store is not None else MemoryStore()
    if thread_id:
        try:
            await store.append(thread_id, "user", state.get("query", ""))
            await store.append(thread_id, "assistant", state.get("answer", ""))
        except Exception:
            # 对话已经完成了，落库失败不能让请求 500。记日志让运维排查。
            logger.exception(
                "业务记录落库失败（thread=%s）", thread_id
            )

    # ---------- 3. 配额扣减（tb_token_usage 累加）----------
    # 从 graph.py persist 节点原样复制：turn_tokens 已含 vision_tokens。
    user_key = state.get("user_key") or (f"user:{user_id}" if user_id else None)
    if user_key:
        try:
            await add_usage(user_key, state.get("turn_tokens", 0))
        except Exception:
            logger.exception("token 用量累加失败（%s）", user_key)
