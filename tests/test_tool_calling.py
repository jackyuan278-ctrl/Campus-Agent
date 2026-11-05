"""function calling 测试：模型自己决定调不调工具、调哪个。"""
import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

import app.agent.graph as graph_module
import app.agent.tools.secondhand as secondhand_module
import app.agent.tools.web as web_module

WEB_HITS = [
    {
        "title": "2026 年值得看的十部电影",
        "url": "https://example.com/movies",
        "content": "年度片单：科幻、剧情、动画各有佳作。",
        "score": 0.91,
        "source": "web",
    }
]

# 二手工具在测试里不能真连 MySQL，统一换成这份假数据
SECONDHAND_HITS = [
    {"id": 1, "name": "捷安特自行车", "price": 180.0, "condition_desc": "7成新"},
    {"id": 2, "name": "美利达自行车", "price": 260.0, "condition_desc": "9成新"},
]

LOCAL_TOOL_NAMES = [
    "search_secondhand_items_tool",
    "get_secondhand_detail_tool",
    "search_campus_activities",
]

SCRIPT: list[AIMessage] = []
CALLS: list = []
BOUND: list[list[str]] = []


class ScriptedModel(GenericFakeChatModel):
    """按 SCRIPT 顺序返回预设消息；记录 bind_tools 收到的工具名。"""

    def bind_tools(self, tools, **kwargs):
        BOUND.append([t.name for t in tools])
        return self

    async def ainvoke(self, messages, config=None, **kwargs):
        index = len(CALLS)
        CALLS.append(messages)
        return SCRIPT[index] if index < len(SCRIPT) else SCRIPT[-1]


class _DummyStore:
    async def append(self, *args, **kwargs):
        return None


@pytest.fixture(autouse=True)
def _reset():
    SCRIPT.clear()
    CALLS.clear()
    BOUND.clear()


@pytest.fixture
def graph(monkeypatch):
    async def fake_search(query, max_results=5):
        return WEB_HITS

    async def fake_secondhand(**kwargs):
        return SECONDHAND_HITS

    monkeypatch.setattr(graph_module, "get_chat_model", lambda **kw: ScriptedModel(messages=iter([AIMessage(content="x")])))
    monkeypatch.setattr(graph_module, "MemoryStore", _DummyStore)
    monkeypatch.setattr(web_module, "search_web", fake_search)
    monkeypatch.setattr(secondhand_module, "search_secondhand_items", fake_secondhand)
    return graph_module.build_graph(MemorySaver())


def _run(graph, query, mode, thread_id):
    async def go():
        return await graph.ainvoke(
            {"query": query, "thread_id": thread_id, "mode": mode},
            {"configurable": {"thread_id": thread_id}},
        )

    return asyncio.run(go())


def test_model_calls_tool_then_answers(graph):
    """模型第一轮要工具 → ToolNode 执行 → 结果回灌 → 第二轮出答案。"""
    SCRIPT.extend([
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "web_search_tool",
                    "args": {"query": "最近有什么好看的电影"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(content="根据检索，推荐三部电影。"),
    ])

    result = _run(graph, "最近有什么好看的电影", "casual", "t-tool")

    assert result["answer"] == "根据检索，推荐三部电影。"
    assert len(CALLS) == 2, "应该调了两次模型：一次要工具、一次出答案"
    assert BOUND[0] == ["web_search_tool"], "casual 档位只该绑联网工具"

    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert "2026 年值得看的十部电影" in str(tool_msgs[0].content)


def test_model_answers_without_tool(graph):
    """模型不要工具时，一轮就结束，不多花调用。"""
    SCRIPT.append(AIMessage(content="你好呀，有什么想聊的？"))

    result = _run(graph, "你好", "casual", "t-notool")

    assert result["answer"] == "你好呀，有什么想聊的？"
    assert len(CALLS) == 1
    assert not any(isinstance(m, ToolMessage) for m in result["messages"])


def test_tools_are_static_per_mode():
    """工具集只跟档位有关，跟「这句话被认成什么意图」无关。

    旧版按意图裁剪（tools_for_intent），判成 other 就返回空集，
    模型连工具名都看不到。现在四个查校园库的档位拿到同一份清单。
    """
    for mode in ("chat", "market", "activity", "doc"):
        assert [t.name for t in graph_module.tools_for_mode(mode)] == LOCAL_TOOL_NAMES, mode

    # 联网是付费能力，只给显式选了「随便聊聊」的档位
    assert [t.name for t in graph_module.tools_for_mode("casual")] == ["web_search_tool"]

    # 未知 / 缺失档位退回本地工具，绝不能是空集（空集 = 模型没有任何工具可用）
    assert [t.name for t in graph_module.tools_for_mode(None)] == LOCAL_TOOL_NAMES


def test_tool_rounds_exhausted_drops_tool_calls(graph):
    """跑满轮数后模型还要工具：必须把 tool_calls 丢掉。

    留着的话，下一轮把历史发给上游（OpenAI 兼容协议）会因
    「assistant 带 tool_calls 却没有配对 ToolMessage」判 400，会话永久卡死。
    """
    def call(idx: int) -> dict:
        return {
            "name": "search_secondhand_items_tool",
            "args": {"keyword": "自行车"},
            "id": f"c{idx}",
            "type": "tool_call",
        }

    # 前 MAX_TOOL_ROUNDS 次会真的执行工具，最后一次（轮数已满）要工具但不会执行
    SCRIPT.extend([AIMessage(content="", tool_calls=[call(i)]) for i in range(1, 5)])

    result = _run(graph, "有二手自行车吗", "market", "t-rounds")

    last = result["messages"][-1]
    assert not (getattr(last, "tool_calls", None) or []), "最后一条消息不该残留未执行的 tool_calls"
    assert result["answer"], "丢掉 tool_calls 后要有兜底正文，不能是空字符串"


def test_follow_up_keeps_tools(graph):
    """追问必须还能查库：档位没变，工具集就不该变。

    回归用例：用户先问「有二手自行车吗」，再追问「最便宜的是哪个」。
    后者不含任何关键词，旧版被判成 other → 空工具集 → 模型只能拿上一轮
    那 5 条旧结果复述，答出来的是「那 5 条里最便宜的」而不是全库最便宜的。
    """
    SCRIPT.extend([
        # 第 1 轮：模型要工具 → 拿结果 → 作答
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_secondhand_items_tool",
                    "args": {"keyword": "自行车"},
                    "id": "call_1",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(content="找到两台自行车。"),
        # 第 2 轮追问：模型再要一次工具（这次多带了价格上限）
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search_secondhand_items_tool",
                    "args": {"keyword": "自行车", "max_price": 200},
                    "id": "call_2",
                    "type": "tool_call",
                }
            ],
        ),
        AIMessage(content="最便宜的是 180 元那台。"),
    ])

    _run(graph, "有二手自行车吗", "market", "t-followup")
    _run(graph, "最便宜的是哪个", "market", "t-followup")

    # 每轮都是「要工具 → 出答案」两次模型调用，所以 4 次 bind_tools
    assert len(BOUND) == 4, f"每轮都该绑工具，实际绑了 {len(BOUND)} 次"
    for index, bound in enumerate(BOUND):
        assert "search_secondhand_items_tool" in bound, f"第 {index} 次调用工具清单里没有二手工具"
