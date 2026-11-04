"""图片入参测试：附件经视觉模型转成文字后进入对话。"""
import asyncio

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

import app.agent.graph as graph_module

CAPTURED: dict = {}


class Fake(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

    async def ainvoke(self, messages, config=None, **kwargs):
        CAPTURED["messages"] = messages
        return AIMessage(content="这是图片里的一道极限题")


class _DummyStore:
    async def append(self, *args, **kwargs):
        return None


@pytest.fixture(autouse=True)
def _reset():
    CAPTURED.clear()


@pytest.fixture
def graph(monkeypatch):
    async def fake_describe(url, prompt=""):
        CAPTURED["url"] = url
        return "题干：求极限 lim(x→0) sin(x)/x", 197

    monkeypatch.setattr(
        graph_module, "get_chat_model", lambda **kw: Fake(messages=iter([AIMessage(content="x")]))
    )
    monkeypatch.setattr(graph_module, "MemoryStore", _DummyStore)
    monkeypatch.setattr(graph_module, "describe_image", fake_describe)
    return graph_module.build_graph(MemorySaver())


def _run(graph, query, thread_id, attachments=None):
    async def go():
        return await graph.ainvoke(
            {
                "query": query,
                "thread_id": thread_id,
                "mode": "doc",
                "attachments": attachments or [],
            },
            {"configurable": {"thread_id": thread_id}},
        )

    return asyncio.run(go())


def test_image_is_described_and_injected(graph):
    """图片 URL 交给视觉模型，结果以【附件内容】拼进用户消息。"""
    result = _run(
        graph,
        "这题怎么做",
        "t-img",
        [{"name": "题目.png", "type": "image/png", "size": 1024, "url": "https://x/a.png"}],
    )

    assert CAPTURED["url"] == "https://x/a.png"
    user_content = str(CAPTURED["messages"][-1].content)
    assert "这题怎么做" in user_content
    assert "【附件内容】" in user_content
    assert "求极限" in user_content
    assert "题目.png" in user_content
    assert result["answer"] == "这是图片里的一道极限题"
    # 识图的 197 token 也要算进本轮配额
    assert result["turn_tokens"] >= 197


def test_attachment_only_message_gets_default_text(graph):
    """只发图片不发文字时，给一个默认提问，避免空消息进模型。"""
    _run(
        graph,
        "",
        "t-img-only",
        [{"name": "a.png", "type": "image/png", "size": 10, "url": "https://x/b.png"}],
    )

    user_content = str(CAPTURED["messages"][-1].content)
    assert "请看看我发的附件" in user_content
    assert "【附件内容】" in user_content


def test_non_image_attachment_is_ignored(graph):
    """PDF 这类附件暂不解析（MinerU 还没接），不应该触发视觉模型。"""
    _run(
        graph,
        "这份讲义讲了什么",
        "t-pdf",
        [{"name": "讲义.pdf", "type": "application/pdf", "size": 2048, "url": "https://x/a.pdf"}],
    )

    assert "url" not in CAPTURED, "PDF 不该调用视觉模型"
    content = str(CAPTURED["messages"][-1].content)
    # 不再静默忽略：明确告诉模型「解析未上线」，让它转告用户
    assert "【附件内容】" in content
    assert "暂不支持解析" in content
    assert "讲义.pdf" in content


def test_image_text_does_not_leak_into_next_turn(graph):
    """上一轮的图片文字不能跟着新提问走（checkpointer 会记住状态）。

    这是真实踩过的 bug：发过一张图之后，后面每一轮都还在说那张图，
    而且视图的 197 token 也被重复计入配额。
    """
    _run(
        graph,
        "这是什么",
        "t-leak",
        [{"name": "a.png", "type": "image/png", "size": 10, "url": "https://x/a.png"}],
    )
    assert "【附件内容】" in str(CAPTURED["messages"][-1].content)

    second = _run(graph, "那它值钱吗", "t-leak", [])

    current = str(CAPTURED["messages"][-1].content)
    assert current == "那它值钱吗", f"新提问被旧图污染：{current!r}"
    assert second["vision_tokens"] == 0
    # 历史里仍保留上一轮的图片内容，追问「它」才答得上来——这是期望行为
    history_text = " ".join(str(m.content) for m in CAPTURED["messages"][:-1])
    assert "求极限" in history_text
