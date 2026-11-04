"""视觉理解：把图片转成文字（OCR / 看图），供文本模型使用。

为什么单独一层，而不是直接把图片塞进主对话：
- 主模型 glm-5.3-flash 是纯文本模型，图片进不去；
- 而它的 function calling 是我们整条工具链的基础，不能为了看图换掉它；
- 所以让视觉模型（glm-4.6v，同一个 GLM key）只负责「看图说人话」，
  输出一段文字，主流程照旧跑文本 + 工具。

代价：多一次模型调用（图片越多越慢），所以节点里限制一次最多读 3 张。
"""
import logging

from openai import APIError, AsyncOpenAI

from app.common.exceptions import AppError
from app.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = """请用中文描述这张图片，供后续对话使用：
1. 图里有文字就逐字提取（OCR），保留标题、编号和换行；
2. 是题目就把题干、选项、已知条件完整抄下来；
3. 是物品照片就描述外观、成色、可见的型号或标签；
4. 只写你确实看到的内容，不要猜测或补全。

控制在 400 字以内，直接输出描述正文。"""

_client: AsyncOpenAI | None = None


class VisionError(AppError):
    """图片识别失败。"""

    code = 2002
    status_code = 502
    message = "图片识别失败"


def _get_client() -> AsyncOpenAI:
    """懒加载：import 时不读配置、不建连接池。"""
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=60,       # 视觉模型比文本慢，给宽一点
            max_retries=1,
        )
    return _client


async def describe_image(url: str, prompt: str = DEFAULT_PROMPT) -> tuple[str, int]:
    """图片 URL → (文字描述, 消耗 token 数)。

    返回 token 数是为了让配额统计把它算进去——识图一次几十到几百 token，
    图片多了不比文本便宜，漏掉会让「每日上限」失真。
    失败抛 VisionError（由调用方决定是否降级）。
    """
    settings = get_settings()
    try:
        resp = await _get_client().chat.completions.create(
            model=settings.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=1200,
        )
    except APIError as exc:
        logger.error("视觉模型调用失败：%s", exc)
        raise VisionError("图片识别失败，请稍后重试") from exc

    content = (resp.choices[0].message.content or "").strip()
    usage = getattr(resp, "usage", None)
    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    logger.info("识图完成：%s tokens=%s", settings.vision_model, total_tokens)
    return content, total_tokens


async def aclose() -> None:
    global _client
    if _client is not None:
        await _client.close()
        _client = None
