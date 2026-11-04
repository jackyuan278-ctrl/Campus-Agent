"""Agent 编排（LangGraph 状态图）——function calling 版。

    START → summarize → image_agent → route → agent ─┬─(模型要工具)→ tools ─┐
                                                      └─(不要工具)──────────┴→ persist → END

各节点职责：
    summarize   上下文压缩（SummarizationMiddleware）：早期对话 → 一段摘要
    image_agent 图片识别：附件里的图片先交给视觉模型转成文字，**所有意图共用**
    route       判意图（「随便聊聊」由前端 mode 指定，其余走关键词规则），
                并把本轮用户消息（含图片文字）写进 messages
    agent       调模型。模型可以直接回答，也可以返回 tool_calls 要求调工具
    tools       ToolNode：执行模型点名的工具，把结果作为 ToolMessage 追加回 messages
    persist     写 tb_chat_message（业务记录） + 累加当日 token 用量

为什么让模型选工具（而不是代码 if/else 决定）：
    代码只能处理「一个意图对应一个工具」；模型可以组合（先查商品再查活动）、
    可以自己从「200 块以内」抽出 max_price=200。代价是每次用到工具要多一轮模型调用。

三个「记忆」相关组件，职责不要混：
    checkpointer   存状态快照（原样、无损、可回滚），管「跑得下去」
    summarize      压缩状态里的 messages（有损、保语义），管「这次发多少」
    tb_chat_message 可读的对话记录，管「人查得到」
"""
import asyncio
import logging
from typing import Any

from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.agent.state import (
    INTENT_ACTIVITY,
    INTENT_CASUAL,
    INTENT_COMMUNITY,
    INTENT_SECONDHAND,
    AgentState,
    infer_intent,
)
from app.agent.tools.activity import CAMPUS_SEARCH_TOOLS
from app.agent.tools.secondhand import SECONDHAND_TOOLS
from app.agent.tools.web import WEB_TOOLS
from app.common.constants import (
    CHAT_SUMMARY_KEEP,
    CHAT_SUMMARY_TRIGGER,
    MAX_IMAGES,
    MAX_TOOL_ROUNDS,
)
from app.llm.gateway import get_chat_model
from app.llm.vision import describe_image
from app.memory.quota import add_usage
from app.memory.store import MemoryStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是 CampusHub 校园助手，帮同学查二手商品、活动和社区讨论。

工具使用规则：
1. 需要商品 / 活动 / 帖子的真实数据时，调用对应工具，不要凭印象回答。
2. 工具返回空结果时，直接说没找到，不要编造。
3. 只依据工具返回的内容描述商品名、价格、成色、时间和地点。

回答风格：中文、口语化、不超过 3 句话；有多件商品时简要列出来。"""

# 「随便聊聊」档位：什么都能聊，联网是可选工具而不是唯一依据
CASUAL_SYSTEM_PROMPT = """你是 CampusHub 校园助手，当前是「随便聊聊」模式，可以聊任何话题。

规则：
1. 需要最新或外部信息时才调用联网搜索工具；日常闲聊直接回答，不要调用。
2. 引用检索结果时不要编造链接、数字和日期。
3. 中文、口语化、简洁，不超过 4 句话。"""

# 摘要提示词：必须保留 {messages} 占位（中间件用 .format(messages=...) 渲染）。
# 面向校园问答场景，不沿用默认那份针对编程 agent 的长提示词（很长、很贵）。
SUMMARY_PROMPT = """把下面这段校园助手与用户的对话压缩成一段摘要，供后续轮次继续对话使用。

必须保留：
- 用户的核心需求（想找什么商品 / 活动 / 信息）和已知条件（预算、品类、时间）
- 已经查到的关键结果（商品名与价格、活动名称与时间地点）
- 用户还没被满足的问题

不要保留寒暄、重复表述和格式说明。用中文，200 字以内，直接输出摘要正文。

<messages>
{messages}
</messages>"""

# 意图 → 可用工具集。只给该意图相关的工具，缩小模型选错的概率（也少送 token）。
_TOOLSETS: dict[str, list[BaseTool]] = {
    INTENT_SECONDHAND: SECONDHAND_TOOLS,
    INTENT_ACTIVITY: CAMPUS_SEARCH_TOOLS,
    INTENT_COMMUNITY: CAMPUS_SEARCH_TOOLS,
    INTENT_CASUAL: WEB_TOOLS,
}

_ALL_TOOLS: list[BaseTool] = SECONDHAND_TOOLS + CAMPUS_SEARCH_TOOLS + WEB_TOOLS


def tools_for_intent(intent: str | None) -> list[BaseTool]:
    """这个意图下允许模型调用的工具；没有匹配的返回空列表（模型只能直接回答）。"""
    return _TOOLSETS.get(intent or "", [])


def _localize_summary(patch: dict[str, Any]) -> None:
    """把中间件写死的英文前缀换成中文（摘要正文本身不动）。

    中间件 _build_new_messages 会产出 "Here is a summary of the conversation to date:"，
    对中文场景不自然，且会和中文摘要混在一起。这里就地替换前缀。
    """
    for msg in patch.get("messages") or []:
        if isinstance(msg, HumanMessage) and msg.additional_kwargs.get("lc_source") == "summarization":
            msg.content = str(msg.content).replace(
                "Here is a summary of the conversation to date:", "【此前对话摘要】"
            )


# 微信 / Windows 传来的图片经常没有 MIME，只能靠扩展名兜底
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")


def _looks_like_image(item: dict[str, Any]) -> bool:
    """判断附件是不是图片：先看 content-type，缺失时退回看文件名后缀。"""
    if str(item.get("type", "")).startswith("image/"):
        return True
    return str(item.get("name", "")).lower().endswith(_IMAGE_SUFFIXES)


def _tokens_of(resp: AIMessage, prompt_messages: list) -> int:
    """本次模型调用消耗的 token。

    优先用上游返回的真实 usage（ChatOpenAI 的 usage_metadata）；
    流式或上游没带 usage 时退化为估算（约 4 字符 / token，中文偏保守）。
    """
    usage = getattr(resp, "usage_metadata", None) or {}
    total = int(usage.get("total_tokens") or 0)
    if total:
        return total
    try:
        return count_tokens_approximately([*prompt_messages, resp])
    except Exception:  # noqa: BLE001 - 估算失败不该影响对话
        return 0


def build_graph(checkpointer=None):
    """构建并编译状态图。

    Args:
        checkpointer: LangGraph 的 saver（AIOMySQLSaver）。传 None 则状态不落盘，
                      仅用于单元测试。
    """
    store = MemoryStore()
    model = get_chat_model()
    # 摘要用独立实例：温度 0。只在触发压缩时才会真的调一次模型。
    # max_tokens 给足 1500：glm-5.3-flash 强制思考，预算太小会被推理吃光、
    # 摘要返回空串（之前测过 max_tokens=20 时正文就是空的）。
    summarizer = SummarizationMiddleware(
        model=get_chat_model(temperature=0.0, max_tokens=1500),
        trigger=("messages", CHAT_SUMMARY_TRIGGER),
        keep=("messages", CHAT_SUMMARY_KEEP),
        summary_prompt=SUMMARY_PROMPT,
        trim_tokens_to_summarize=4000,
    )

    async def summarize(state: AgentState) -> dict[str, Any]:
        """上下文压缩：消息超过阈值时，把早期对话交给模型压成一段摘要。

        交给 SummarizationMiddleware 做，不是按条截断——截断会直接丢掉
        「用户想找什么、已经查到什么」，下一轮模型就断片。
        """
        before = len(state.get("messages") or [])
        try:
            patch = await summarizer.abefore_model(state, None)  # type: ignore[arg-type]
        except Exception:
            # 摘要失败不应中断对话：宁可这轮多带点上下文，也别让用户拿不到回复。
            logger.exception("上下文摘要失败，本轮按原样继续（消息数=%d）", before)
            return {}
        if not patch:
            return {}
        _localize_summary(patch)
        logger.info("上下文已压缩：%d 条 → 摘要 + 最近 %d 条", before, CHAT_SUMMARY_KEEP)
        return patch

    async def image_agent(state: AgentState) -> dict[str, Any]:
        """图片识别节点：所有意图都会先经过这里。

        一次最多读 MAX_IMAGES 张：每张都是一次视觉模型调用，图片多了又慢又贵。
        单张失败只降级成一行说明，不让整轮对话挂掉。

        ⚠️ 没有新图时必须显式清空 attachment_text / vision_tokens：
        checkpointer 会把上一轮的值留在状态里，不清的话它会拼进之后**每一轮**
        的提问（表现为「换了话题它还在说那张图」），配额也会重复计。
        """
        attachments = state.get("attachments") or []
        images: list[dict[str, Any]] = []
        unsupported: list[dict[str, Any]] = []
        for item in attachments:
            (images if _looks_like_image(item) and item.get("url") else unsupported).append(item)

        blocks: list[str] = []
        if unsupported:
            # 明确告诉模型"这些附件处理不了"，而不是让它对着一堆文件名瞎猜
            names = "、".join(str(s.get("name") or "未知文件") for s in unsupported)
            blocks.append(f"（附件 {names} 暂不支持解析，如被问到请说明文档解析功能还未上线）")
            if any(not s.get("url") for s in unsupported):
                logger.warning(
                    "image_agent：附件缺 url（前端未走 /assistant/upload？）：%s",
                    [{"name": s.get("name"), "type": s.get("type")} for s in unsupported],
                )

        if not images:
            return {"attachment_text": "\n".join(blocks), "vision_tokens": 0}

        # 超出上限的图片明确说明，而不是静默丢掉
        overflow = images[MAX_IMAGES:]
        images = images[:MAX_IMAGES]
        if overflow:
            names = "、".join(str(o.get("name") or "图片") for o in overflow)
            blocks.append(f"（图片 {names} 超出单轮 {MAX_IMAGES} 张上限，本轮未读取）")

        results = await asyncio.gather(
            *(describe_image(item["url"]) for item in images), return_exceptions=True
        )

        vision_tokens = 0
        for item, result in zip(images, results):
            name = item.get("name") or "图片"
            if isinstance(result, BaseException):
                logger.warning("图片识别失败：%s（%s）", name, result)
                blocks.append(f"- {name}：识别失败，未获取到内容")
                continue
            text, tokens = result
            vision_tokens += tokens
            if not text:
                blocks.append(f"- {name}：没有识别到可用内容")
            else:
                blocks.append(f"- {name}：\n{text}")
        return {"attachment_text": "\n".join(blocks), "vision_tokens": vision_tokens}

    async def route(state: AgentState) -> dict[str, Any]:
        """判意图 + 写入本轮用户消息。

        意图只用来选「工具集 + 提示词」，不再直接决定查哪个库——
        查什么由模型通过 tool_calls 决定。
        """
        if state.get("mode") == "casual":
            intent = INTENT_CASUAL
        else:
            intent = infer_intent(state.get("query", ""))

        content = state.get("query") or "请看看我发的附件"
        attachment_text = state.get("attachment_text")
        if attachment_text:
            content = f"{content}\n\n【附件内容】\n{attachment_text}"

        logger.info(
            "route：thread=%s mode=%s intent=%s 附件=%d",
            state.get("thread_id"), state.get("mode"), intent, len(state.get("attachments") or []),
        )
        return {
            "intent": intent,
            "messages": [HumanMessage(content=content)],
            "tool_rounds": 0,
            # 识图已经花掉的 token 先记上，agent 节点再往上累加
            "turn_tokens": state.get("vision_tokens", 0),
        }

    async def agent(state: AgentState) -> dict[str, Any]:
        """调模型：要么直接回答，要么返回 tool_calls 要求调工具。"""
        rounds = state.get("tool_rounds", 0)
        tools = tools_for_intent(state.get("intent"))
        # 跑满轮数后不再给工具，逼模型基于已有信息作答，避免死循环
        runnable = model.bind_tools(tools) if tools and rounds < MAX_TOOL_ROUNDS else model

        system_prompt = (
            CASUAL_SYSTEM_PROMPT if state.get("intent") == INTENT_CASUAL else SYSTEM_PROMPT
        )
        messages = [SystemMessage(content=system_prompt), *(state.get("messages") or [])]

        resp: AIMessage = await runnable.ainvoke(messages)
        calls = getattr(resp, "tool_calls", None) or []
        if calls:
            logger.info(
                "agent：模型请求 %d 个工具 → %s",
                len(calls), [c.get("name") for c in calls],
            )
        return {
            "messages": [resp],
            "answer": resp.content or "",
            "tool_rounds": rounds + 1,
            "turn_tokens": state.get("turn_tokens", 0) + _tokens_of(resp, messages),
        }

    async def persist(state: AgentState) -> dict[str, Any]:
        """写业务记录（可读版） + 累加当日 token 用量。"""
        thread_id = state.get("thread_id")
        if thread_id:
            await store.append(thread_id, "user", state.get("query", ""))
            await store.append(thread_id, "assistant", state.get("answer", ""))
        user_key = state.get("user_key")
        if user_key:
            try:
                await add_usage(user_key, state.get("turn_tokens", 0))
            except Exception:
                logger.exception("写入 token 用量失败（%s）", user_key)
        return {}

    def should_use_tools(state: AgentState) -> str:
        """有 tool_calls 且还有轮数就执行工具，否则收尾。"""
        last = (state.get("messages") or [None])[-1]
        calls = getattr(last, "tool_calls", None) or []
        if calls and state.get("tool_rounds", 0) < MAX_TOOL_ROUNDS:
            return "tools"
        return "persist"

    graph = StateGraph(AgentState)
    graph.add_node("summarize", summarize)
    graph.add_node("image_agent", image_agent)
    graph.add_node("route", route)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(_ALL_TOOLS))
    graph.add_node("persist", persist)

    graph.add_edge(START, "summarize")
    graph.add_edge("summarize", "image_agent")
    graph.add_edge("image_agent", "route")
    graph.add_edge("route", "agent")
    graph.add_conditional_edges(
        "agent",
        should_use_tools,
        {"tools": "tools", "persist": "persist"},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("persist", END)

    return graph.compile(checkpointer=checkpointer)
