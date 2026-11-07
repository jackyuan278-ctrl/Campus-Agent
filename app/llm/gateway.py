"""LLM 客户端构造：全项目只有一个入口。

历史包袱已清理：这里曾经还有一个 LLMGateway 类（一次性返回完整字符串的调用方式），
但自从编排改成 LangGraph 之后，所有模型调用都走 `get_chat_model()` 返回的 Runnable
（需要 astream / 事件回调 / bind_tools），那套同步网关再没人用过——留着只会让人
以为有两条链路，排查时先怀疑错地方。

两个约束写在这里，避免下次重构时踩回去：

1. `max_tokens` 至少 1500。glm-5.3-flash 是「始终思考」的模型，给它 20 个 token
   做预算，正文会被推理吃光，返回空串（实测过）。
2. `stream_usage=True`。OpenAI 协议默认只在非流式响应里带 usage，
   不开这个的话 SSE 路径拿不到真实 token 数，配额只能靠估算。
"""
import logging

from langchain_openai import ChatOpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)


def get_chat_model(*, temperature: float = 0.3, max_tokens: int = 1500) -> ChatOpenAI:
    """构造 LangChain 的 ChatOpenAI（指向 GLM 的 OpenAI 兼容接口）。

    返回的是 Runnable：支持 ainvoke / astream / 事件回调 / bind_tools，
    给 LangGraph 节点和 SSE 流式输出用。
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
