"""对话相关的请求/响应模型。"""
import ipaddress
from typing import Annotated
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

# 单条消息长度上限：挡住误传的大文本，同时给多轮上下文留余量。
# 这里不设 min_length——只发附件、不发文字是合法用法，由 model_validator 兜「至少一样」。
MessageText = Annotated[str, StringConstraints(strip_whitespace=True, max_length=4000)]
ThreadId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]

# 单轮附件数上限。和前端的 ATTACHMENT_RULES.maxCount 对齐，
# 也和 graph.MAX_IMAGES 一致——超出的图片本来就不会被读取。
MAX_ATTACHMENTS = 5


def _is_safe_attachment_url(url: str) -> bool:
    """附件 URL 只能指向公开的 http(s) 资源。

    这些 URL 会被原样交给视觉模型去拉取，所以必须挡掉内网地址：
    云环境的元数据地址（169.254.169.254）、回环、私有网段——否则等于给了
    调用方一个让外部服务探测内网的口子。
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    host = parsed.hostname or ""
    if not host:
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        # 不是 IP，按域名处理：挡掉明显的内网名
        return host != "localhost" and not host.endswith((".local", ".internal"))
    return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved)


class Attachment(BaseModel):
    """一个已上传到 OSS 的附件。

    只收元信息 + URL：文件本身在 /assistant/upload 里已经落 OSS 了，
    这里再传二进制没有意义（JSON 里也塞不下）。
    """

    name: str = Field(default="", max_length=200)
    type: str = Field(default="", max_length=100)
    size: int = Field(default=0, ge=0)
    url: str | None = Field(default=None, description="OSS 公网地址，视觉模型靠它读图")

    @field_validator("url")
    @classmethod
    def _check_url(cls, value: str | None) -> str | None:
        if value and not _is_safe_attachment_url(value):
            raise ValueError("附件地址非法：只接受公开的 http/https 地址")
        return value

    @property
    def is_image(self) -> bool:
        return self.type.startswith("image/")


class ChatRequest(BaseModel):
    """一次对话请求。字段与前端 campus-web 的 assistant 模块对齐。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    message: MessageText = Field(default="", description="用户输入，0~4000 字符")
    thread_id: ThreadId = Field(description="会话 ID，1~64 字符")
    mode: str = Field(default="chat", description="对话模式：chat/market/activity/doc/casual")
    attachments: list[Attachment] = Field(
        default_factory=list,
        max_length=MAX_ATTACHMENTS,
        description=f"附件（已上传，带 OSS URL），最多 {MAX_ATTACHMENTS} 个",
    )

    @model_validator(mode="after")
    def _require_text_or_attachment(self) -> "ChatRequest":
        """只发图不发字是合法的，但不能两者都空。"""
        if not self.message and not self.attachments:
            raise ValueError("message 与 attachments 至少要有一个")
        return self
