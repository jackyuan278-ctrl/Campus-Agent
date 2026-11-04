"""上下文压缩测试：超过阈值时用摘要替换早期消息，而不是按条截断。

阈值在 build_graph 里读的是模块级常量，这里 monkeypatch 成很小的值，
只跑 5 轮就能触发，避免测试里跑 20 多轮。
"""
import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

import app.agent.graph as graph_module

SUMMARY_TEXT = "用户在找高数教材，预算 400 元。"


class _DummyStore:
    """替掉 MemoryStore，避免测试写 tb_chat_message。"""

    async def append(self, *args, **kwargs):
        return None


@pytest.fixture
def small_threshold_graph(monkeypatch):
    """阈值调到 6 条触发、保留 2 条，返回 (graph, 调用记录)。"""
    monkeypatch.setattr(graph_module, "CHAT_SUMMARY_TRIGGER", 6)
    monkeypatch.setattr(graph_module, "CHAT_SUMMARY_KEEP", 2)
    monkeypatch.setattr(graph_module, "MemoryStore", _DummyStore)

    calls: list[str] = []

    class Fake(GenericFakeChatModel):
        async def ainvoke(self, messages, config=None, **kwargs):
            is_summary = isinstance(messages, str)
            calls.append("summary" if is_summary else "chat")
            return AIMessage(content=SUMMARY_TEXT if is_summary else "（回答）有 3 本教材在售。")

    monkeypatch.setattr(
        graph_module, "get_chat_model", lambda **kw: Fake(messages=iter([AIMessage(content="x")]))
    )
    return graph_module.build_graph(MemorySaver()), calls


def test_summary_replaces_early_messages(small_threshold_graph):
    """5 轮对话（不压缩应有 10 条）应被压缩，且首条是中文摘要。"""
    graph, calls = small_threshold_graph
    config = {"configurable": {"thread_id": "t-summary"}}

    async def run():
        result = {}
        for i in range(5):
            result = await graph.ainvoke(
                {"query": f"第{i}轮问题", "thread_id": "t-summary", "tool_results": []},
                config,
            )
        return result

    result = asyncio.run(run())
    messages = result["messages"]

    assert len(messages) < 10, f"没有压缩，仍有 {len(messages)} 条"
    summaries = [m for m in messages if m.additional_kwargs.get("lc_source") == "summarization"]
    assert len(summaries) == 1
    assert str(summaries[0].content).startswith("【此前对话摘要】")
    assert SUMMARY_TEXT in str(summaries[0].content)
    assert calls.count("summary") >= 1


def test_no_summary_below_threshold(small_threshold_graph):
    """只有 2 轮（4 条）时不该触发摘要，不该多花一次模型调用。"""
    graph, calls = small_threshold_graph
    config = {"configurable": {"thread_id": "t-nosummary"}}

    async def run():
        result = {}
        for i in range(2):
            result = await graph.ainvoke(
                {"query": f"第{i}轮问题", "thread_id": "t-nosummary", "tool_results": []},
                config,
            )
        return result

    result = asyncio.run(run())
    assert len(result["messages"]) == 4
    assert calls.count("summary") == 0
    assert not any(
        m.additional_kwargs.get("lc_source") == "summarization" for m in result["messages"]
    )
