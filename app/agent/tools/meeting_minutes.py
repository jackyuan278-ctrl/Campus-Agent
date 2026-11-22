"""meeting_minutes skill 的 LangGraph tools：抽待办、写纪要、分配负责人。

**纯 LLM tools** — 函数体返回结构化 prompt，实际内容由模型生成。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool

from app.agent.skills import SkillRegistry

logger = logging.getLogger(__name__)


# ── 1. 从会议文本抽待办 ────────────────────────────────────────────

@tool
async def extract_action_items(meeting_text: str, format_type: str = "structured") -> str:
    """从会议记录/转写文本中提取 Action Items（待办事项）。

    Args:
        meeting_text: 会议全文（可以是转写稿、记录、聊天记录）。
        format_type: 输出格式。可选：structured（表格） / bullet（列表） / kanban（看板视图）。
    """
    prompt = (
        f"【提取 Action Items 请求】\n"
        f"输出格式：{format_type}\n\n"
        f"会议文本：\n```\n{meeting_text}\n```\n\n"
        f"请从中提取所有明确/隐含的待办事项，每条包含：\n"
        f"- 任务描述（动词开头，具体可执行）\n"
        f"- 负责人（如果没提，标记为「未分配」）\n"
        f"- 截止时间（如果没提，标记为「待定」）\n"
        f"- 优先级（高/中/低，基于讨论热度和紧迫性判断）\n"
        f"- 来源（在会议中是哪部分讨论出来的）\n\n"
        f"如果没有明确待办，也请输出「本次会议未产生明确 Action Items」。"
    )
    return json.dumps(
        {"mode": "llm_self_reflect", "prompt_for_model": prompt, "tool": "extract_action_items"},
        ensure_ascii=False,
    )


# ── 2. 会议纪要生成 ────────────────────────────────────────────────

@tool
async def summarize_meeting(
    meeting_text: str,
    meeting_title: str = "",
    attendees: str = "",
    detail_level: str = "标准",
) -> str:
    """把会议转写/记录整理成结构化纪要。

    Args:
        meeting_text: 会议全文。
        meeting_title: 会议标题（可选）。
        attendees: 参会人（可选）。
        detail_level: 详细程度。可选：极简 / 标准 / 详细。
    """
    prompt = (
        f"【会议纪要生成请求】\n"
        f"标题：{meeting_title or '未命名会议'}\n"
        f"参会人：{attendees or '（未提供）'}\n"
        f"详细程度：{detail_level}\n\n"
        f"会议文本：\n```\n{meeting_text}\n```\n\n"
        f"请生成一份结构化纪要，包含：\n"
        f"1. 会议基本信息（时间/主题/参会人）\n"
        f"2. 讨论议题（分点列出，附每个议题的关键结论）\n"
        f"3. 决议（明确达成了什么共识）\n"
        f"4. Action Items（待办清单）\n"
        f"5. 遗留问题（会上没结论、会后跟进）"
    )
    return json.dumps(
        {"mode": "llm_self_reflect", "prompt_for_model": prompt, "tool": "summarize_meeting"},
        ensure_ascii=False,
    )


# ── 3. 分配负责人 ─────────────────────────────────────────────────

@tool
async def assign_owner(
    action_items: str,
    team_members: str,
    context: str = "",
) -> str:
    """给待办事项推荐/分配负责人。

    Args:
        action_items: 待办清单（文本）。
        team_members: 团队成员列表（含技能信息更好，如「张三-后端 / 李四-前端 / 王五-产品」）。
        context: 额外背景信息（如「上一个项目的 Owner」）。
    """
    prompt = (
        f"【分配负责人请求】\n"
        f"团队成员：\n{team_members}\n\n"
        f"待办清单：\n{action_items}\n\n"
        f"背景：{context or '（无）'}\n\n"
        f"请为每条待办推荐最合适的负责人，并说明理由，格式：\n"
        f"- [任务] → [负责人]（理由：匹配 XX 技能 / 正在做相关工作 / 时间充裕）\n"
        f"如果某个任务所有人都不合适，建议「外部支持」。"
    )
    return json.dumps(
        {"mode": "llm_self_reflect", "prompt_for_model": prompt, "tool": "assign_owner"},
        ensure_ascii=False,
    )


# ── SkillRegistry 注册 ────────────────────────────────────────────

MEETING_TOOLS = [extract_action_items, summarize_meeting, assign_owner]

MEETING_PROMPT = """你是会议助手，擅长从会议文本里抽待办、整理纪要、分配负责人。

可用工具：
1. extract_action_items — 会议文本 → Action Items 清单
2. summarize_meeting — 生成结构化会议纪要
3. assign_owner — 给待办推荐/分配负责人

风格：
- 纪要要客观，记录讨论和决议，不要加入主观评价
- Action Items 要具体可执行，动词开头
- 分配负责人时说明理由，体现对团队成员的了解
"""


def register(r: SkillRegistry) -> None:
    """把会议助手工具包注册进 SkillRegistry — 给 chat 档。"""
    r.register_tools("chat", MEETING_TOOLS)
    r.register_prompt("chat", MEETING_PROMPT)
