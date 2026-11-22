"""weekly_report skill 的 LangGraph tools：任务周报、章节生成、OKR 复盘。

**纯 LLM tools** — 函数体返回结构化 prompt，实际内容由模型生成。
"""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

from app.agent.skills import SkillRegistry

logger = logging.getLogger(__name__)


# ── 1. 任务列表 → 周报摘要 ────────────────────────────────────────

@tool
async def summarize_tasks(task_list: str, highlight_count: int = 3) -> str:
    """从任务列表生成周报摘要。

    Args:
        task_list: 本周任务列表（纯文本，如「1. 完成 XX 功能 2. 修了 YY bug」）。
        highlight_count: 重点突出几项，默认 3。
    """
    prompt = (
        f"【周报摘要请求】\n"
        f"本周任务：\n```\n{task_list}\n```\n\n"
        f"请生成一份周报摘要，包含：\n"
        f"1. 一句话本周总结\n"
        f"2. Top {highlight_count} 亮点产出（按重要性排序）\n"
        f"3. 遇到的问题 & 解决方案\n"
        f"4. 下周计划（基于当前任务自动推导）\n\n"
        f"格式：简洁、专业，适合发给 leader / 团队。"
    )
    return json.dumps(
        {"mode": "llm_self_reflect", "prompt_for_model": prompt, "tool": "summarize_tasks"},
        ensure_ascii=False,
    )


# ── 2. 周报章节生成 ────────────────────────────────────────────────

@tool
async def generate_section(
    section: str,
    context: str = "",
    audience: str = "leader",
    tone: str = "专业简洁",
) -> str:
    """生成周报的某个章节。

    Args:
        section: 章节名称，如「本周成果」「下周计划」「风险预警」「指标数据」。
        context: 章节素材（要点）。
        audience: 受众。可选：leader / 团队 / 自我复盘。
        tone: 语调风格。可选：专业简洁 / 详细充实 / 轻松口语。
    """
    prompt = (
        f"【周报章节生成请求】\n"
        f"章节：{section}\n"
        f"素材：{context or '（暂无，请根据章节名自动生成合理框架）'}\n"
        f"受众：{audience}\n"
        f"风格：{tone}\n\n"
        f"请生成该章节的完整内容，结构清晰、要点突出。"
    )
    return json.dumps(
        {"mode": "llm_self_reflect", "prompt_for_model": prompt, "tool": "generate_section"},
        ensure_ascii=False,
    )


# ── 3. OKR 复盘 ────────────────────────────────────────────────────

@tool
async def okr_reflection(
    objective: str,
    key_results: str,
    progress: str,
    period: str = "weekly",
) -> str:
    """对 OKR 做阶段性复盘反思。

    Args:
        objective: O（目标）的一句话描述。
        key_results: KR 列表（纯文本）。
        progress: 当前进度描述。
        period: 复盘周期。可选：weekly / monthly / quarterly。
    """
    prompt = (
        f"【OKR 复盘请求】\n"
        f"O（目标）：{objective}\n"
        f"KR：\n{key_results}\n"
        f"当前进度：{progress}\n"
        f"周期：{period}\n\n"
        f"请给出：\n"
        f"1. 客观进度评估（每个 KR 的完成度打分 0~100%）\n"
        f"2. 做对了什么（继续保持）\n"
        f"3. 哪里出了偏差（需要纠正）\n"
        f"4. 下个周期的行动调整\n"
        f"5. 一句话复盘总结"
    )
    return json.dumps(
        {"mode": "llm_self_reflect", "prompt_for_model": prompt, "tool": "okr_reflection"},
        ensure_ascii=False,
    )


# ── SkillRegistry 注册 ────────────────────────────────────────────

WEEKLY_TOOLS = [summarize_tasks, generate_section, okr_reflection]

WEEKLY_PROMPT = """你是周报 & OKR 助手，擅长帮用户从任务列表生成周报、写特定章节、做 OKR 复盘。

可用工具：
1. summarize_tasks — 任务列表 → 周报摘要
2. generate_section — 生成周报某个章节
3. okr_reflection — OKR 阶段性复盘

风格要求：
- 输出简洁、结构化，用 Markdown 表格/列表
- 周报要突出亮点和问题，不要罗列流水账
- OKR 复盘要客观，既有肯定也有改进方向
"""


def register(r: SkillRegistry) -> None:
    """把周报工具包注册进 SkillRegistry — 给 chat 档。"""
    r.register_tools("chat", WEEKLY_TOOLS)
    r.register_prompt("chat", WEEKLY_PROMPT)
