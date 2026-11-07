"""Agent 状态定义（LangGraph StateGraph 的状态）+ 意图识别。

LangGraph 要求状态是 TypedDict（或 dataclass / pydantic），这里用 TypedDict：
- messages 用 add_messages reducer，节点返回的消息会自动**追加**而不是覆盖
- 其余字段是普通覆盖语义
"""
import logging
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

logger = logging.getLogger(__name__)

# 意图常量
INTENT_SECONDHAND = "secondhand"
INTENT_ACTIVITY = "activity"
INTENT_COMMUNITY = "community"
INTENT_OTHER = "other"
# 「随便聊聊」：由前端 mode=casual 直接指定，不靠关键词判断。
# 它和其余意图平级，区别是走联网搜索（Tavily）而不是本地库。
INTENT_CASUAL = "casual"

# 关键词规则：顺序有意义，先匹配到的胜出
_INTENT_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    (
        INTENT_SECONDHAND,
        ("二手", "多少钱", "成色", "几成新", "出售", "出个", "收购", "闲置",
         "教材", "自行车", "键盘", "台灯", "买", "卖"),
    ),
    (INTENT_ACTIVITY, ("活动", "报名", "讲座", "比赛", "社团招新", "签到", "名额")),
    (INTENT_COMMUNITY, ("帖子", "讨论", "问答", "社团", "评论", "点赞", "关注")),
]


class AgentState(TypedDict, total=False):
    """图的状态。checkpointer 会按 thread_id 把这份状态落盘。"""

    messages: Annotated[list[BaseMessage], add_messages]  # 多轮对话（自动追加）
    query: str          # 本轮用户输入
    thread_id: str      # 会话 ID（同时是 checkpointer 的 thread_id）
    mode: str           # 前端档位：chat / market / activity / doc / casual
    intent: str         # secondhand / activity / community / casual / other
    tool_rounds: int    # 已经跑了几轮工具调用（防死循环）
    user_key: str       # 配额统计标识：user:<userId> 或 anon:<threadId>
    turn_tokens: int    # 本轮累计 token（含工具轮），persist 时写入用量表
    attachments: list[dict[str, Any]]  # 附件元信息（含 OSS url）
    doc_urls: list[dict[str, Any]]     # 本会话传过的文档地址（name+url），供后续追问重新检索
    attachment_text: str  # 图片经视觉模型转成的文字（拼进本轮提问）
    vision_tokens: int    # 识图花掉的 token（并入本轮配额）
    answer: str         # 本轮最终回复


def infer_intent(query: str) -> str:
    """意图识别（规则版：零延迟、零成本、可解释）。

    ⚠️ 这里算出来的 intent **只用于日志和接口返回，不参与任何能力决策**。
    给哪些工具、用哪份 system prompt，都由前端传的 mode 决定（见 graph.py 的
    tools_for_mode / prompt_for_mode）。曾经用 intent 裁剪工具集，结果
    「最便宜的是哪个」被判成 other → 工具集为空 → 模型只能拿历史复述。

    为什么不用 LLM 做这一步：只是观测用途，一次模型调用的
    延迟（0.3~0.8s）和费用更不划算。
    """
    for intent, keywords in _INTENT_KEYWORDS:
        if any(word in query for word in keywords):
            logger.debug("意图命中：%s ← %s", intent, query)
            return intent
    return INTENT_OTHER
