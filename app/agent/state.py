"""Agent 状态定义。"""
from typing import Any

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """一次对话的状态：用户输入、中间步骤、最终输出。"""

    session_id: str = ""
    query: str = ""
    intent: str = ""  # secondhand / activity / community / other
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    answer: str = ""


def infer_intent(query: str) -> str:
    """意图识别。"""
    ...
