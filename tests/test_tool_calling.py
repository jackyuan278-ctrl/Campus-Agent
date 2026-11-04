"""function calling 测试：模型自己决定调不调工具、调哪个。"""
import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver

import app.agent.graph as graph_module
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

    monkeypatch.setattr(graph_module, "get_chat_model", lambda **kw: ScriptedModel(messages=iter([AIMessage(content="x")])))
    monkeypatch.setattr(graph_module, "MemoryStore", _DummyStore)
    monkeypatch.setattr(web_module, "search_web", fake_search)
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


def test_tools_are_scoped_by_intent():
    """不同意图绑不同工具集，避免模型跨域乱调。"""
    assert [t.name for t in graph_module.tools_for_intent("secondhand")] == [
        "search_secondhand_items_tool",
        "get_secondhand_detail_tool",
    ]
    assert [t.name for t in graph_module.tools_for_intent("activity")] == [
        "search_campus_activities"
    ]
    assert [t.name for t in graph_module.tools_for_intent("casual")] == ["web_search_tool"]
    assert graph_module.tools_for_intent("other") == []
