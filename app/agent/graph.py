"""Agent 编排（LangGraph 状态图）——function calling 版。

    START → summarize → image_agent → route → agent ─┬─(模型要工具)→ tools ─┐
                                                      └─(不要工具)──────────┴→ persist → END

各节点职责：
    summarize   上下文压缩（SummarizationMiddleware）：早期对话 → 一段摘要
    image_agent 图片识别：附件里的图片先交给视觉模型转成文字，**所有档位共用**
    route       算出本轮意图（只用于日志和接口返回）+ 把本轮用户消息
                （含图片文字）写进 messages
    agent       调模型。模型可以直接回答，也可以返回 tool_calls 要求调工具
    tools       ToolNode：执行模型点名的工具，把结果作为 ToolMessage 追加回 messages
    persist     写 tb_chat_message（业务记录） + 累加当日 token 用量

档位（mode）决定两件事，都发生在装配阶段，**与这句话说了什么无关**：
    tools_for_mode   给哪些工具（本地三个恒给；联网只给 casual）
    prompt_for_mode  用哪份 system prompt（每档一份）
后果：同一档位下第 1 轮和第 N 轮的工具清单完全一样，追问不会「丢工具」。
不要让用户消息反过来影响工具集——那会让判错直接变成「工具消失 + 模型编造」。

为什么让模型选工具（而不是代码 if/else 决定）：
    代码只能处理「一个档位对应一个工具」；模型可以组合（先查商品再查活动）、
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

from app.agent.state import INTENT_CASUAL, AgentState, infer_intent
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

# 档位不匹配时的统一话术：让模型自己判断，代码不替用户改档。
# 抽成常量是因为 5 份 prompt 里 4 份都要用，改一处就全生效。
_SWITCH_HINT = """
档位提醒：如果用户的问题明显不属于当前档位，照常回答，但末尾用一句话
建议他切换到更合适的档位（二手估价 / 活动推荐 / 资料解读 / 智能问答）。
不要替他做决定，也不要假装自己已经切过去了。"""

# 「二手估价」档：报价必须有参照物，否则模型会凭印象编一个数字
MARKET_SYSTEM_PROMPT = (
    """你是 CampusHub 校园助手，当前在「二手估价」档位。
用户想了解闲置物品能卖多少，或者在售商品的行情。

工具使用规则：
1. 需要真实在售数据时调用 search_secondhand_items_tool 查库，不要凭印象报价。
2. 查不到同类商品就直说没找到，不要编造价格。
3. 报价要说明参照物（哪几件在售、什么成色），没有参照物的估价就是瞎猜。
4. 想看某件商品的描述和交易地点，用 get_secondhand_detail_tool 查详情。

回答风格：中文、口语化、不超过 3 句话；给区间而不是单一数字。"""
    + _SWITCH_HINT
)

# 「活动推荐」档
ACTIVITY_SYSTEM_PROMPT = (
    """你是 CampusHub 校园助手，当前在「活动推荐」档位。
用户想找校园活动、讲座、比赛、社团招新。

工具使用规则：
1. 用 search_campus_activities 检索真实的活动与帖子，不要编造活动名。
2. 检索为空就直说最近没有，不要凑数。
3. 列活动时带上时间、地点、报名方式（检索结果里有才说）。

回答风格：中文、口语化，不超过 4 句话；多个活动列成短条目。"""
    + _SWITCH_HINT
)

# 「资料解读」档：PDF 解析未上线，必须让模型明说，不能假装读过内容
DOC_SYSTEM_PROMPT = (
    """你是 CampusHub 校园助手，当前在「资料解读」档位。
用户上传讲义或截图，想让你划重点、做总结。

规则：
1. 图片已经由视觉模型转成文字，附在提问后面的【附件内容】里，直接基于那段文字回答。
2. PDF / Word 解析功能还未上线：用户传了这类文件要明确说暂不支持，绝不假装读到了内容。
3. 总结按「重点 / 考点 / 需要补充」分段，不要大段复述原文。

回答风格：中文，条理清楚，可以用短列表。"""
    + _SWITCH_HINT
)

# 「智能问答」档：默认档位，也是未知 mode 的兜底
CHAT_SYSTEM_PROMPT = (
    """你是 CampusHub 校园助手，帮同学查二手商品、活动和社区讨论。

工具使用规则：
1. 需要商品 / 活动 / 帖子的真实数据时，调用对应工具，不要凭印象回答。
2. 工具返回空结果时，直接说没找到，不要编造。
3. 只依据工具返回的内容描述商品名、价格、成色、时间和地点。

回答风格：中文、口语化、不超过 3 句话；有多件商品时简要列出来。"""
    + _SWITCH_HINT
)

# 「随便聊聊」档：本档没有校园库工具，所以要主动引导用户切档
CASUAL_SYSTEM_PROMPT = """你是 CampusHub 校园助手，当前是「随便聊聊」模式，可以聊任何话题。

规则：
1. 需要最新或外部信息时才调用联网搜索工具；日常闲聊直接回答，不要调用。
2. 引用检索结果时不要编造链接、数字和日期。
3. 这个档位没有校园库查询能力：被问到二手商品、活动、帖子时，
   明确告诉用户切到「二手估价」「活动推荐」或「智能问答」档再问，不要凭常识编。

回答风格：中文、口语化、简洁，不超过 4 句话。"""

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

# 工具集是**静态装配**：只跟「这次会话允许哪些能力」有关，跟用户这句话说了
# 什么无关。装配在每轮开始前完成，同一档位下第 1 轮和第 N 轮拿到的清单完全一样。
#
# 旧版按意图裁剪（_TOOLSETS）的教训：判错时工具会直接消失，模型连补救入口都
# 没有，只能拿历史复述——「最便宜的是哪个」被答成「那 5 条里最便宜的」就是
# 这么来的。参考 Codex 的 build_tool_router：它的入参里根本没有用户消息，
# 裁剪维度全是模型能力 / 权限 / 预算。
_LOCAL_TOOLS: list[BaseTool] = SECONDHAND_TOOLS + CAMPUS_SEARCH_TOOLS
# ToolNode 执行时按工具名反查，所以必须收全量，不能用 tools_for_mode 的结果
_ALL_TOOLS: list[BaseTool] = _LOCAL_TOOLS + WEB_TOOLS


def tools_for_mode(mode: str | None) -> list[BaseTool]:
    """本轮给模型哪些工具（装配阶段定死，不随对话内容变）。

    联网（Tavily）是付费能力，只在用户显式选了「随便聊聊」时开放——这是
    能力开关，不是意图猜测。本地工具永远全给：三个工具的 description 已经
    写清了各自用途，让模型自己读、自己选，比代码猜意图可靠得多。
    """
    if mode == INTENT_CASUAL:
        return WEB_TOOLS
    return _LOCAL_TOOLS


# 档位 → system prompt。每档一份：视角、工具用法、回答风格都不同
# （估价要给区间、活动要带时间地点、资料解读要讲清 PDF 还没上线）。
_MODE_PROMPTS: dict[str, str] = {
    "market": MARKET_SYSTEM_PROMPT,
    "activity": ACTIVITY_SYSTEM_PROMPT,
    "doc": DOC_SYSTEM_PROMPT,
    "chat": CHAT_SYSTEM_PROMPT,
    "casual": CASUAL_SYSTEM_PROMPT,
}


def prompt_for_mode(mode: str | None) -> str:
    """档位 → system prompt。未知档位退回「智能问答」，不报错也不拦。"""
    return _MODE_PROMPTS.get(mode or "", CHAT_SYSTEM_PROMPT)


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
        """写本轮用户消息 + 记录意图。

        意图现在只用来打日志和填 chat.py 的响应字段，**不再决定工具集和提示词**
        ——工具和 prompt 都由 mode 决定（见 tools_for_mode / prompt_for_mode）。
        查什么由模型通过 tool_calls 决定。
        """
        mode = state.get("mode") or ""
        # intent 只用于日志和 chat.py 的响应字段，**不参与工具集和 prompt 的选择**。
        # 保留它是为了让排查问题时还能看到「这句话被认成什么意图」。
        intent = INTENT_CASUAL if mode == INTENT_CASUAL else infer_intent(state.get("query", ""))

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
        tools = tools_for_mode(state.get("mode"))
        # 跑满轮数后不再给工具，逼模型基于已有信息作答，避免死循环
        runnable = model.bind_tools(tools) if tools and rounds < MAX_TOOL_ROUNDS else model

        system_prompt = prompt_for_mode(state.get("mode"))
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
