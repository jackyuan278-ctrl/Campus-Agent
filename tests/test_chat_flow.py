"""对话链路测试：SSE 流式、历史读写、清空。

用 conftest 里的 client 夹具：lifespan 会跑，模型是假模型，不产生 API 费用。
"""
from tests.conftest import FAKE_REPLY


def test_sse_stream_frames(client):
    """默认返回 SSE：逐条 data: {"delta":...}，最后以 [DONE] 收尾。"""
    resp = client.post(
        "/assistant/chat",
        json={"message": "有高数教材吗", "thread_id": "t-sse"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]

    frames = [f for f in resp.text.strip().split("\n\n") if f]
    assert frames[-1] == "data: [DONE]"
    assert any('"delta"' in f for f in frames)

    client.delete("/assistant/chat/messages", params={"thread_id": "t-sse"})


def test_json_mode(client):
    """?stream=false 时返回一次性 JSON，结构为 data.content / data.intent。"""
    resp = client.post(
        "/assistant/chat?stream=false",
        json={"message": "有高数教材吗", "thread_id": "t-json"},
    )
    assert resp.status_code == 200
    body = resp.json()["data"]
    assert body["content"] == FAKE_REPLY
    assert body["thread_id"] == "t-json"
    assert body["intent"] == "secondhand"

    client.delete("/assistant/chat/messages", params={"thread_id": "t-json"})


def test_history_and_clear(client):
    """两轮之后历史应有 4 条；清空后归零（业务记录 + checkpoint 一起清）。"""
    thread_id = "t-history"
    client.post("/assistant/chat?stream=false", json={"message": "第一句", "thread_id": thread_id})
    client.post("/assistant/chat?stream=false", json={"message": "第二句", "thread_id": thread_id})

    messages = client.get("/assistant/chat/messages", params={"thread_id": thread_id}).json()["data"]
    assert len(messages) == 4
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant"]

    client.delete("/assistant/chat/messages", params={"thread_id": thread_id})
    assert client.get("/assistant/chat/messages", params={"thread_id": thread_id}).json()["data"] == []
