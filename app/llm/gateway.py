"""LLM 网关：模型无关的抽象，后续可切换 deepseek/glm/ollama 等。

设计：编排层不直接依赖具体模型 SDK，通过本网关统一调用。
- 全异步：使用 AsyncOpenAI，与 FastAPI / Redis / MySQL 链路一致
- 客户端单例：懒建 + 复用连接池，应用退出时 aclose
- 异常统一成 LLMError（AppError 子类），由 main.py 的全局 handler 转 JSON
"""
import logging
from typing import Any

from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError

from app.common.exceptions import AppError
from app.config import get_settings

logger = logging.getLogger(__name__)


class LLMError(AppError):
    """上游模型调用失败（超时 / 限流 / 鉴权 / 网络）。"""

    code = 2001
    status_code = 502
    message = "模型服务暂时不可用"


class LLMGateway:
    """统一的大模型调用入口。"""

    def __init__(self) -> None:
        # 客户端懒建：import 本模块不产生任何网络资源
        self._client: AsyncOpenAI | None = None

    @property
    def _settings(self):
        """每次现取（get_settings 有 lru_cache，开销为零），避免 import 期固化配置。"""
        return get_settings()

    @property
    def configured(self) -> bool:
        """是否已配置 LLM（key + base_url + model 齐全）。"""
        return self._settings.llm_configured

    def _get_client(self) -> AsyncOpenAI:
        """获取（或首次创建）共享客户端，复用其内部连接池。"""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._settings.llm_api_key,
                base_url=self._settings.llm_base_url,
                timeout=30.0,      # 兜住卡死：SDK 默认 600s 太长
                max_retries=2,     # 429 / 5xx 自动重试两次
            )
            logger.info(
                "创建 LLM 客户端：base_url=%s model=%s",
                self._settings.llm_base_url,
                self._settings.llm_model,
            )
        return self._client

    async def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """发送对话，返回文本回复。

        Args:
            messages: OpenAI 格式的消息列表，如 [{"role": "user", "content": "..."}]
            **kwargs: 透传给 SDK（temperature、max_tokens、tools 等）

        Raises:
            LLMError: 未配置、超时、限流或上游报错
        """
        if not self.configured:
            raise LLMError("未配置 LLM，请检查 .env 的 GLM_API_KEY / GLM_BASE_URL / GLM_CHAT_MODEL")

        try:
            resp = await self._get_client().chat.completions.create(
                model=self._settings.llm_model,
                messages=messages,
                **kwargs,
            )
        except APITimeoutError as exc:
            logger.error("LLM 调用超时：model=%s", self._settings.llm_model)
            raise LLMError("模型调用超时，请稍后重试") from exc
        except RateLimitError as exc:
            logger.warning("LLM 调用被限流：model=%s", self._settings.llm_model)
            raise LLMError("模型调用被限流，请稍后重试") from exc
        except APIError as exc:
            logger.error("LLM 调用失败：%s", exc)
            raise LLMError("模型调用失败") from exc

        # 模型只返回 tool_calls 时 content 为 None，统一收敛成空串
        content = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        logger.info(
            "LLM 完成：model=%s tokens=%s",
            self._settings.llm_model,
            getattr(usage, "total_tokens", None),
        )
        return content

    async def aclose(self) -> None:
        """关闭客户端连接池（由 main.py 的 lifespan 在应用退出时调用）。"""
        if self._client is not None:
            await self._client.close()
            self._client = None
            logger.info("已关闭 LLM 客户端")


# 全局单例
llm_gateway = LLMGateway()
