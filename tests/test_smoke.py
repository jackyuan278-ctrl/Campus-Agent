"""基础测试：应用可导入、路由契约符合预期、参数校验生效。

注意：这里的 client 没用 with 上下文，lifespan 不执行，
所以状态图未初始化 —— 正好用来测「图没起来时返回 503」。
"""
from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import make_token

client = TestClient(app)
# 这些用例只关心「过了鉴权之后」的行为，统一带一个合法 token
AUTH = {"Authorization": f"Bearer {make_token()}"}


def test_health():
    """存活探针：健康检查是公开接口，不需要登录。"""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_chat_requires_login():
    """没带 token 一律 401（对齐 campus-hub 网关）。"""
    resp = client.post("/assistant/chat", json={"message": "hi", "thread_id": "t1"})
    assert resp.status_code == 401


def test_chat_returns_503_without_lifespan():
    """lifespan 没跑时不该是 500，而应是明确的 503。"""
    resp = client.post(
        "/assistant/chat",
        json={"message": "hi", "thread_id": "t1"},
        headers=AUTH,
    )
    assert resp.status_code == 503


def test_chat_requires_message():
    """请求体缺 message 且无附件时应由校验器拦截，返回 422。"""
    resp = client.post("/assistant/chat", json={"thread_id": "t1"}, headers=AUTH)
    assert resp.status_code == 422


def test_chat_rejects_blank_message():
    """纯空白 message 被 strip 后应为空（且无附件），返回 422。"""
    resp = client.post("/assistant/chat", json={"message": "   ", "thread_id": "t1"}, headers=AUTH)
    assert resp.status_code == 422


def test_chat_rejects_too_long_message():
    """超过 4000 字符的 message 应被拒绝，返回 422。"""
    resp = client.post(
        "/assistant/chat",
        json={"message": "x" * 4001, "thread_id": "t1"},
        headers=AUTH,
    )
    assert resp.status_code == 422
