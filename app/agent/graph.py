"""Agent 编排：意图识别 → 工具调用 / RAG → 生成回答。

    query → infer_intent → 按意图分发（tools / rag） → 组装 AgentState
"""
from app.agent.state import AgentState


def run_agent(query: str, session_id: str = "") -> AgentState:
    """执行一次 Agent 对话，返回完整状态。"""
    ...
