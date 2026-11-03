"""对话相关的请求/响应模型。"""
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

# 单条消息长度上限：挡住误传的大文本，同时给多轮上下文留余量
MessageText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
ThreadId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]


class ChatRequest(BaseModel):
    """一次对话请求。

    strip_whitespace 在长度校验之前执行，所以纯空白串会被判为空、直接 422。
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    message: MessageText = Field(description="用户输入，1~4000 字符，纯空白视为非法")
    thread_id: ThreadId = Field(description="会话 ID，1~64 字符")
