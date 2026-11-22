"""code_developer skill 的 LangGraph tools：代码解释、调试思路、重构建议。

**纯 LLM tools** — 函数体不调外部 API，只是让模型触发自我反思：
tool 返回一个结构化 prompt，实际分析由模型在 tool 调用后的 AIMessage 里完成。
"""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

from app.agent.skills import SkillRegistry

logger = logging.getLogger(__name__)


# ── 1. 代码解释 ───────────────────────────────────────────────────

@tool
async def explain_code(code: str, language: str = "auto", focus: str = "整体逻辑") -> str:
    """让 LLM 解释一段代码的含义。

    Args:
        code: 要解释的代码片段。
        language: 编程语言。auto 自动识别 / python / javascript / java / go / cpp。
        focus: 解释侧重点。可选：整体逻辑 / 逐行解读 / 数据结构 / 算法思路 / 边界条件。
    """
    # 让模型自己决定要不要真的解释（纯 LLM tool）
    prompt = (
        f"【代码解释请求】\n"
        f"语言：{language}\n"
        f"侧重点：{focus}\n\n"
        f"代码片段：\n```\n{code}\n```\n\n"
        f"请按照侧重点给出清晰、易懂的解释。"
    )
    return json.dumps(
        {"mode": "llm_self_reflect", "prompt_for_model": prompt, "tool": "explain_code"},
        ensure_ascii=False,
    )


# ── 2. 调试思路 ───────────────────────────────────────────────────

@tool
async def debug_hint(error: str, code: str = "", context: str = "") -> str:
    """根据报错信息给调试思路。

    Args:
        error: 报错信息 / Traceback。
        code: 相关代码片段（可选）。
        context: 额外上下文（如「这是一个 FastAPI 路由」「这是异步代码」）。
    """
    prompt = (
        f"【调试请求】\n"
        f"报错信息：\n```\n{error}\n```\n\n"
        f"相关代码：\n```\n{code}\n```\n\n"
        f"上下文：{context}\n\n"
        f"请分析可能的根因，并给出 2~3 个排查方向。"
    )
    return json.dumps(
        {"mode": "llm_self_reflect", "prompt_for_model": prompt, "tool": "debug_hint"},
        ensure_ascii=False,
    )


# ── 3. 重构建议 ────────────────────────────────────────────────────

@tool
async def suggest_refactor(code: str, goal: str = "可读性", language: str = "auto") -> str:
    """给一段代码的重构建议。

    Args:
        code: 要重构的代码片段。
        goal: 重构目标。可选：可读性 / 性能 / 可维护性 / 安全性 / 测试友好。
        language: 编程语言。auto 自动识别。
    """
    prompt = (
        f"【重构建议请求】\n"
        f"重构目标：{goal}\n"
        f"语言：{language}\n\n"
        f"原始代码：\n```\n{code}\n```\n\n"
        f"请分析现有问题，给出具体的重构建议和示例代码。"
    )
    return json.dumps(
        {"mode": "llm_self_reflect", "prompt_for_model": prompt, "tool": "suggest_refactor"},
        ensure_ascii=False,
    )


# ── SkillRegistry 注册 ────────────────────────────────────────────

CODE_DEV_TOOLS = [explain_code, debug_hint, suggest_refactor]

CODE_DEV_PROMPT = """你是代码助手，擅长解释代码、排查 bug、给出重构建议。

可用工具：
1. explain_code — 解释代码含义（可指定侧重点）
2. debug_hint — 根据报错给调试思路
3. suggest_refactor — 重构建议

注意：
- 收到 tool 返回后，**不要把 JSON 原样吐给用户**，而是按照返回的 prompt_for_model 组织自然语言回答
- 用户给代码时要确认语言和用途，这会让你的建议更精准
"""


def register(r: SkillRegistry) -> None:
    """把代码助手工具包注册进 SkillRegistry — 给 chat 档。"""
    r.register_tools("chat", CODE_DEV_TOOLS)
    r.register_prompt("chat", CODE_DEV_PROMPT)
