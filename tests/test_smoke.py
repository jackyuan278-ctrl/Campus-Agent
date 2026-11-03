"""基础测试：应用可导入、路由契约符合预期。

注意：TestClient 没用 with 上下文，lifespan 不会执行，
所以这里不依赖 .env 的配置完整性。
"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    """存活探针。"""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_chat_not_implemented():
    """/assistant/chat 当前应为 501（显式未实现），而不是静默返回 null。"""
    resp = client.post("/assistant/chat", json={"message": "hi", "thread_id": "t1"})
    assert resp.status_code == 501


def test_chat_requires_message():
    """请求体缺 message 时应由 pydantic 拦截，返回 422。"""
    resp = client.post("/assistant/chat", json={"thread_id": "t1"})
    assert resp.status_code == 422


def test_chat_rejects_blank_message():
    """纯空白 message 被 strip 后应为空，返回 422。"""
    resp = client.post("/assistant/chat", json={"message": "   ", "thread_id": "t1"})
    assert resp.status_code == 422


def test_chat_rejects_too_long_message():
    """超过 4000 字符的 message 应被拒绝，返回 422。"""
    resp = client.post("/assistant/chat", json={"message": "x" * 4001, "thread_id": "t1"})
    assert resp.status_code == 422
