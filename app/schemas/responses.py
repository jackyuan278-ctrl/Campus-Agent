"""Responses API 兼容层的请求/响应 schema（供 Codex CLI / Cursor 等外部客户端用）。

这个文件**完全独立**于 chat.py 的 schema——外部客户端走 OpenAI Responses API 协议，
字段名、结构和 /assistant/chat 用的 ChatRequest 不一样。
"""

from pydantic import BaseModel, ConfigDict, Field


class ResponsesRequest(BaseModel):
    """Responses API 请求（只实现了单轮 input + 可选 thread_id 续聊）。"""

    model_config = ConfigDict(str_strip_whitespace=True)

    # 占位字段：外部客户端（Codex CLI / Cursor）会传 model 来指定用哪个，
    # 但我们忽略它——实际走 app.config 里的 llm_model。
    model: str = Field(default="glm-5.3-flash", description="模型名（忽略，走服务端 config）")
    instructions: str | None = Field(default=None, description="可选覆盖 system prompt")
    input: str = Field(..., description="用户输入文本（单轮）", max_length=4000)
    stream: bool = Field(default=True, description="true 时走 SSE，false 时一次性返回 JSON")
    thread_id: str | None = Field(default=None, description="可选，不传就服务端生成")
    mode: str = Field(default="chat", description="对话档位：chat/market/activity/doc/casual")


class ResponsesEvent(BaseModel):
    """SSE 事件帧（type 决定事件含义）。"""

    type: str = Field(
        ...,
        description=(
            "response.created / response.output_text.delta / "
            "response.function_call / response.completed"
        ),
    )
    delta: str | None = Field(default=None, description="逐 token 文本增量")
    item_id: str | None = Field(default=None, description="文本块/工具调用的唯一 ID")


class ResponsesFinalResponse(BaseModel):
    """非流式（stream=false）时的一次性返回。"""

    thread_id: str
    output_text: str
    mode: str
