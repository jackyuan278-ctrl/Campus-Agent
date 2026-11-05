"""接口层测试：附件上传、配额拦截、鉴权与会话归属。"""
import io

from fastapi.testclient import TestClient

import app.api.v1.chat as chat_module
import app.api.v1.upload as upload_module
from app.main import app
from app.memory.quota import QuotaState
from tests.conftest import TEST_USER_ID, make_token


def test_upload_returns_url(client, monkeypatch):
    """上传接口：预检额度 → 传文件 → 成功后才扣次数。

    顺序是被刻意断言的：如果先扣次数再上传，OSS 失败时用户会白损失一次当天的
    上传额度（网络抖几下当天就没法传了）。
    """
    captured = {}
    calls = []

    async def fake_check(user_key):
        calls.append("check_quota")

    async def fake_consume(user_key):
        calls.append("consume")
        captured["user_key"] = user_key

    async def fake_upload(data, filename, content_type):
        calls.append("upload")
        captured.update(data=data, filename=filename, content_type=content_type)
        return {"url": "https://example.com/a.png", "key": "k/a.png", "size": len(data)}

    monkeypatch.setattr(upload_module, "check_upload_quota", fake_check)
    monkeypatch.setattr(upload_module, "consume_upload", fake_consume)
    monkeypatch.setattr(upload_module, "upload_bytes", fake_upload)

    resp = client.post(
        "/assistant/upload",
        files={"file": ("note.png", io.BytesIO(b"fake-bytes"), "image/png")},
    )

    assert resp.status_code == 200
    body = resp.json()["data"]
    assert body["url"] == "https://example.com/a.png"
    assert captured["filename"] == "note.png"
    # 上传次数记在登录用户头上，而不是 anon
    assert captured["user_key"] == f"user:{TEST_USER_ID}"
    assert calls == ["check_quota", "upload", "consume"], "顺序必须是先查、再传、最后扣"


def test_quota_exceeded_blocks_request(client, monkeypatch):
    """配额用尽时返回 429，且一次模型都不调。"""
    async def fake_quota(user_key):
        return QuotaState(user_key=user_key, used=100_000, limit=100_000)

    monkeypatch.setattr(chat_module, "get_quota", fake_quota)

    resp = client.post(
        "/assistant/chat?stream=false",
        json={"message": "你好", "thread_id": "t-quota"},
    )

    assert resp.status_code == 429
    assert resp.json()["code"] == 3001
    assert "额度" in resp.json()["message"]


def test_quota_allows_request_when_under_limit(client, monkeypatch):
    """还有余额时正常放行。"""
    async def fake_quota(user_key):
        return QuotaState(user_key=user_key, used=1_000, limit=100_000)

    monkeypatch.setattr(chat_module, "get_quota", fake_quota)

    resp = client.post(
        "/assistant/chat?stream=false",
        json={"message": "你好", "thread_id": "t-quota-ok"},
    )

    assert resp.status_code == 200
    client.delete("/assistant/chat/messages", params={"thread_id": "t-quota-ok"})


def test_endpoints_require_login():
    """没有 token 一律 401（对齐 campus-hub 网关的语义）。"""
    with TestClient(app) as c:
        cases = [
            ("post", "/assistant/chat?stream=false", {"json": {"message": "hi", "thread_id": "t"}}),
            ("get", "/assistant/chat/messages?thread_id=t", {}),
            ("delete", "/assistant/chat/messages?thread_id=t", {}),
            ("post", "/assistant/upload", {"files": {"file": ("a.png", io.BytesIO(b"x"), "image/png")}}),
        ]
        for method, url, kwargs in cases:
            resp = getattr(c, method)(url, **kwargs)
            assert resp.status_code == 401, f"{method} {url} 应要求登录，实际 {resp.status_code}"


def test_other_users_thread_is_forbidden(client):
    """登录了也不能碰别人的会话：第一个用户发起，第二个用户来读/删 → 403。"""
    thread_id = "t-owned-by-1001"
    client.post(
        "/assistant/chat?stream=false",
        json={"message": "第一句", "thread_id": thread_id},
    )

    intruder_headers = {"Authorization": f"Bearer {make_token('9999')}"}
    for method in ("get", "delete"):
        resp = getattr(client, method)(
            "/assistant/chat/messages",
            params={"thread_id": thread_id},
            headers=intruder_headers,
        )
        assert resp.status_code == 403, f"{method} 别人的会话应 403，实际 {resp.status_code}"

    client.delete("/assistant/chat/messages", params={"thread_id": thread_id})
