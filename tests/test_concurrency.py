"""并发行为测试：同一会话必须串行，否则对话历史会被静默覆盖。

这类 bug 最难查的地方在于「没有异常」：请求全部返回 200，日志干干净净，
只有用户会发现「我刚才说的话它不记得了」。
"""
import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

import app.agent.graph as graph_module
from app.agent.locks import thread_lock


class _NewIdModel(GenericFakeChatModel):
    """每次 ainvoke 都新建 AIMessage（唯一 id），模拟真实模型。

    ⚠️ 不要用 `itertools.cycle([AIMessage(...)])` 复用一个实例：add_messages 按 id
    去重，复用会变成「替换」，丢消息的现象会失真——排查时踩过这个坑。
    """

    def bind_tools(self, tools, **kwargs):
        return self

    async def ainvoke(self, messages, config=None, **kwargs):
        return AIMessage(content="好的")


class _DummyStore:
    async def append(self, *args, **kwargs):
        return None


@pytest.fixture
def graph_and_saver(monkeypatch):
    monkeypatch.setattr(
        graph_module, "get_chat_model", lambda **kw: _NewIdModel(messages=iter([]))
    )
    monkeypatch.setattr(graph_module, "MemoryStore", _DummyStore)
    saver = MemorySaver()
    return graph_module.build_graph(saver), saver


def _messages_of(saver, thread_id):
    tup = saver.get_tuple({"configurable": {"thread_id": thread_id}})
    return tup.checkpoint.get("channel_values", {}).get("messages", []) if tup else []


def test_same_thread_requests_are_serialized(graph_and_saver):
    """同一会话 4 个并发请求：加锁后应留下 8 条消息（4 问 4 答）。

    checkpointer 是「读状态 → 执行 → 写回」，没有版本冲突检测。不加锁时 4 个请求
    都读到同一份旧状态，各自跑完再写回，后写的把先写的整段盖掉——实测只剩
    1 问 1 答，另外 3 轮对话凭空消失。
    """
    graph, saver = graph_and_saver
    thread_id = "t-serialized"

    async def run():
        async def one(index: int):
            # 这条锁就是被测对象：去掉它，下面的断言会失败
            async with thread_lock(thread_id):
                return await graph.ainvoke(
                    {"query": f"第{index}句", "thread_id": thread_id, "mode": "chat"},
                    {"configurable": {"thread_id": thread_id}},
                )

        return await asyncio.gather(*[one(i) for i in range(4)])

    results = asyncio.run(run())

    assert all(r.get("answer") == "好的" for r in results), "每个请求本身都应成功"
    messages = _messages_of(saver, thread_id)
    assert len(messages) == 8, f"应保留 4 问 4 答，实际 {len(messages)} 条：{[m.content for m in messages]}"
    human = [m.content for m in messages if m.type == "human"]
    assert human == ["第0句", "第1句", "第2句", "第3句"], "四轮提问都要在，顺序不能乱"


def test_different_threads_do_not_block_each_other(graph_and_saver):
    """锁是按会话的：不同会话并发不应该互相排队。"""
    graph, _ = graph_and_saver

    async def run():
        async def one(thread_id: str):
            async with thread_lock(thread_id):
                return await graph.ainvoke(
                    {"query": "你好", "thread_id": thread_id, "mode": "chat"},
                    {"configurable": {"thread_id": thread_id}},
                )

        return await asyncio.gather(*[one(f"t-parallel-{i}") for i in range(3)])

    results = asyncio.run(run())
    assert len(results) == 3
    assert all(r.get("answer") == "好的" for r in results)
