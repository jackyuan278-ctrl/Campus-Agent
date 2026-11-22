"""interview skill 的 LangGraph tools：简历评审、模拟面试、薪资基准。

**纯 LLM tools** — 函数体返回结构化 prompt，实际内容由模型生成。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool

from app.agent.skills import SkillRegistry

logger = logging.getLogger(__name__)


# ── 1. 简历评审 ───────────────────────────────────────────────────

@tool
async def resume_review(resume_text: str, target_position: str = "", focus: str = "综合评估") -> str:
    """对简历打分并给出改进建议。

    Args:
        resume_text: 简历全文或关键片段。
        target_position: 目标岗位（可选，如「后端工程师」「产品经理」）。
        focus: 评审侧重点。可选：综合评估 / 技术栈匹配 / 项目亮点 / 表达与格式 / 目标岗位匹配度。
    """
    prompt = (
        f"【简历评审请求】\n"
        f"目标岗位：{target_position or '未指定'}\n"
        f"评审侧重点：{focus}\n\n"
        f"简历内容：\n```\n{resume_text}\n```\n\n"
        f"请给出：\n"
        f"1. 总分（1~100）及分项得分\n"
        f"2. 3 个最突出的亮点\n"
        f"3. 3 个需要改进的地方（具体到怎么改）\n"
        f"4. 一句话电梯 pitch"
    )
    return json.dumps(
        {"mode": "llm_self_reflect", "prompt_for_model": prompt, "tool": "resume_review"},
        ensure_ascii=False,
    )


# ── 2. 模拟面试题 ──────────────────────────────────────────────────

@tool
async def mock_interview_question(
    position: str,
    level: str = "中级",
    question_type: str = "综合",
    count: int = 5,
) -> str:
    """按岗位生成面试题。

    Args:
        position: 岗位名称，如「后端工程师」「产品经理」「数据分析师」。
        level: 岗位级别。可选：初级 / 中级 / 高级。
        question_type: 题目类型。可选：综合 / 技术 / 行为 / 系统设计 / 算法。
        count: 题目数量，默认 5。
    """
    prompt = (
        f"【模拟面试题请求】\n"
        f"岗位：{position}\n"
        f"级别：{level}\n"
        f"题型：{question_type}\n"
        f"数量：{count}\n\n"
        f"请生成 {count} 道高质量面试题，每道题附带：\n"
        f"- 考察点（面试官想考什么）\n"
        f"- 好的回答应该包含哪些要点\n"
        f"- 常见的回答误区"
    )
    return json.dumps(
        {"mode": "llm_self_reflect", "prompt_for_model": prompt, "tool": "mock_interview_question"},
        ensure_ascii=False,
    )


# ── 3. 薪资基准 ────────────────────────────────────────────────────

@tool
async def salary_benchmark(position: str, city: str = "北京", years_experience: int = 3) -> str:
    """查询某岗位的薪资范围。

    Args:
        position: 岗位名称。
        city: 城市。
        years_experience: 工作年限。
    """
    prompt = (
        f"【薪资基准请求】\n"
        f"岗位：{position}\n"
        f"城市：{city}\n"
        f"工作年限：{years_experience} 年\n\n"
        f"请基于公开的薪资数据（猎聘、BOSS 直聘、脉脉等）给出：\n"
        f"1. 薪资范围（P25 / P50 / P75）\n"
        f"2. 大厂 vs 中小厂的差异\n"
        f"3. 学历/公司背景对薪资的影响\n"
        f"4. 谈薪时的参考技巧"
    )
    return json.dumps(
        {"mode": "llm_self_reflect", "prompt_for_model": prompt, "tool": "salary_benchmark"},
        ensure_ascii=False,
    )


# ── SkillRegistry 注册 ────────────────────────────────────────────

INTERVIEW_TOOLS = [resume_review, mock_interview_question, salary_benchmark]

INTERVIEW_PROMPT = """你是求职面试助手，擅长简历评审、模拟面试、薪资咨询。

可用工具：
1. resume_review — 简历打分 + 改进建议
2. mock_interview_question — 按岗位生成面试题
3. salary_benchmark — 薪资范围参考

注意：
- 薪资数据基于公开信息，仅供参考，实际以 HR 面试沟通为准
- 收到 tool 返回后，按照 prompt_for_model 组织自然语言回答
"""


def register(r: SkillRegistry) -> None:
    """把求职面试工具包注册进 SkillRegistry — 给 chat 档。"""
    r.register_tools("chat", INTERVIEW_TOOLS)
    r.register_prompt("chat", INTERVIEW_PROMPT)
