"""LLM 网关：模型无关的抽象，后续可切换 deepseek/glm/ollama 等。

设计：编排层不直接依赖具体模型 SDK，通过本网关统一调用。
"""
import logging
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)


class LLMGateway:
    """统一的大模型调用入口（占位实现，接入真实 SDK 时替换）。"""

    def __init__(self) -> None:
        self._settings = get_settings()

    @property
    def configured(self) -> bool:
        """是否已配置 LLM（key + base_url + model 齐全）。"""
        return self._settings.llm_configured

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """发送对话，返回文本回复。

        当前为占位实现：接入 OpenAI 兼容 SDK 后，这里调用真实接口。
        """
        if not self.configured:
            logger.warning("LLM 未配置，跳过真实调用")
            return "（未配置 LLM，请在 .env 中填写 GLM_API_KEY / GLM_BASE_URL / GLM_CHAT_MODEL）"
        # TODO: 接入 openai 兼容客户端
        # client = OpenAI(api_key=self._settings.llm_api_key, base_url=self._settings.llm_base_url)
        # resp = client.chat.completions.create(model=self._settings.llm_model, messages=messages, **kwargs)
        # return resp.choices[0].message.content
        raise NotImplementedError("LLM 调用待接入")


# 全局单例
llm_gateway = LLMGateway()
