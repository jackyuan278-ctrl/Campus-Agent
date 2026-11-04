"""pytest 公共夹具：用假模型 + 合法 JWT 启动应用，不产生真实 API 费用。"""
import itertools
import time

import jwt
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import app.agent.graph as graph_module
from app.config import get_settings
from app.main import app

FAKE_REPLY = "（测试回复）有 3 件教辅在售。"
TEST_USER_ID = "1001"


def make_token(user_id: str = TEST_USER_ID) -> str:
    """造一个和 campus-hub 同签发规则的 token（HS256，sub=userId）。"""
    settings = get_settings()
    now = int(time.time())
    return jwt.encode(
        {"sub": user_id, "iat": now, "exp": now + 3600},
        settings.jwt_secret,
        algorithm="HS256",
    )


class FakeChatModel(GenericFakeChatModel):
    """假模型：补上 bind_tools。

    GenericFakeChatModel 继承自 BaseChatModel，bind_tools 默认抛 NotImplementedError，
    而图里会调用 model.bind_tools(...)。这里返回自身，效果等价于「工具绑了但不用」。
    """

    def bind_tools(self, tools, **kwargs):
        return self


def make_fake_model(reply: str = FAKE_REPLY) -> FakeChatModel:
    return FakeChatModel(messages=itertools.cycle([AIMessage(content=reply)]))


@pytest.fixture
def client(monkeypatch):
    """带 lifespan 的客户端：模型换成假模型，请求自动带上合法 token。"""
    monkeypatch.setattr(
        graph_module,
        "get_chat_model",
        lambda **kwargs: make_fake_model(),
    )
    with TestClient(app) as test_client:
        # 所有 assistant 接口都要求登录，测试统一以 TEST_USER_ID 的身份请求
        test_client.headers.update({"Authorization": f"Bearer {make_token()}"})
        yield test_client
    # 清理 app.state，避免测试之间互相污染（运行时连接已随 lifespan 关闭）
    app.state.agent_runtime = None
