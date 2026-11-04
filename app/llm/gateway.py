"""LLM 网关：模型无关的抽象，后续可切换 deepseek/glm/ollama 等。

设计：编排层不直接依赖具体模型 SDK，通过本网关统一调用。
- 全异步：使用 AsyncOpenAI，与 FastAPI / Redis / MySQL 链路一致
- 客户端单例：懒建 + 复用连接池，应用退出时 aclose
- 异常统一成 LLMError（AppError 子类），由 main.py 的全局 handler 转 JSON
"""
import logging
from typing import Any

from langchain_openai import ChatOpenAI
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


def get_chat_model(*, temperature: float = 0.3, max_tokens: int = 1500) -> ChatOpenAI:
    """构造 LangChain 的 ChatOpenAI（指向 GLM 的 OpenAI 兼容接口）。

    和 LLMGateway.chat() 共用同一份 .env 配置，区别只是形态：
    - LLMGateway.chat()：一次性返回完整字符串，适合简单调用
    - get_chat_model()：返回 Runnable，支持 astream / 事件回调 / bind_tools，
      给 LangGraph 节点和 SSE 流式输出用
    """
    settings = get_settings()
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=temperature,
        max_tokens=max_tokens,   # 推理模型要留足预算，否则内容被思考吃光
        timeout=30,
        max_retries=2,
        # 让流式响应也带 usage（OpenAI 协议默认只在非流式返回）。
        # 不开启的话，SSE 路径拿不到真实 token 数，配额只能靠估算。
        stream_usage=True,
        # glm-5.3-flash 是「始终思考」的模型，关不掉，但能调档位。
        # 我们是简单问答，用 low 换更快的首字和更低的 token 消耗。
        # 注意参数形状：thinking 必须是对象，type 只认 enabled/disabled，档位放 level。
        extra_body={"thinking": {"type": "enabled", "level": "low"}},
    )


# 全局单例
llm_gateway = LLMGateway()
