"""对话相关的请求/响应模型。"""
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

# 单条消息长度上限：挡住误传的大文本，同时给多轮上下文留余量。
# 这里不设 min_length——只发附件、不发文字是合法用法，由 model_validator 兜「至少一样」。
MessageText = Annotated[str, StringConstraints(strip_whitespace=True, max_length=4000)]
ThreadId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]


class Attachment(BaseModel):
    """一个已上传到 OSS 的附件。

    只收元信息 + URL：文件本身在 /assistant/upload 里已经落 OSS 了，
    这里再传二进制没有意义（JSON 里也塞不下）。
    """

    name: str = Field(default="", max_length=200)
    type: str = Field(default="", max_length=100)
    size: int = Field(default=0, ge=0)
    url: str | None = Field(default=None, description="OSS 公网地址，视觉模型靠它读图")

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
        default_factory=list, description="附件（已上传，带 OSS URL）"
    )

    @model_validator(mode="after")
    def _require_text_or_attachment(self) -> "ChatRequest":
        """只发图不发字是合法的，但不能两者都空。"""
        if not self.message and not self.attachments:
            raise ValueError("message 与 attachments 至少要有一个")
        return self
