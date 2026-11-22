"""fitness skill 的 LangGraph tools：训练计划、饮食计划、追踪建议。

**纯 LLM tools** — 函数体返回结构化 prompt，实际内容由模型生成。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool

from app.agent.skills import SkillRegistry

logger = logging.getLogger(__name__)


# ── 1. 训练计划 ────────────────────────────────────────────────────


@tool
async def plan_workout(
    goal: str,
    fitness_level: str = "新手",
    equipment: str = "无器械",
    available_days: int = 3,
    focus_area: str = "全身",
) -> str:
    """按用户目标生成训练计划。

    Args:
        goal: 训练目标。可选：减脂 / 增肌 / 塑形 / 体能提升 / 康复 / 备战比赛。
        fitness_level: 当前水平。可选：新手 / 中级 / 高级。
        equipment: 可用器材。可选：无器械 / 哑铃 / 健身房 / 弹力带 / 跑步机。
        available_days: 每周能练几天，1~7。
        focus_area: 重点部位，可选：全身 / 上肢 / 下肢 / 核心 / 背部。
    """
    prompt = (
        f"【训练计划请求】\n"
        f"目标：{goal}\n"
        f"水平：{fitness_level}\n"
        f"器材：{equipment}\n"
        f"每周 {available_days} 天\n"
        f"重点：{focus_area}\n\n"
        f"请生成一份完整训练计划，包含：\n"
        f"1. 每周训练安排（哪几天练什么）\n"
        f"2. 每天的具体动作（组数 × 次数/时间，含休息）\n"
        f"3. 动作要点 & 常见错误提醒\n"
        f"4. 热身 & 拉伸建议\n"
        f"5. 3 周进阶路线图"
    )
    return json.dumps(
        {"mode": "llm_self_reflect", "prompt_for_model": prompt, "tool": "plan_workout"},
        ensure_ascii=False,
    )


# ── 2. 饮食计划 ────────────────────────────────────────────────────


@tool
async def plan_meal(
    goal: str,
    budget_daily: float = 100,
    dietary_restriction: str = "无",
    meal_count: int = 3,
    food_preference: str = "中餐",
) -> str:
    """按预算和目标生成饮食计划。

    Args:
        goal: 饮食目标。可选：减脂 / 增肌 / 健康均衡 / 控糖 / 低碳水。
        budget_daily: 每日预算（元）。
        dietary_restriction: 忌口/过敏，如「不吃辣」「素食」「乳糖不耐」。
        meal_count: 每天几餐，2~6。
        food_preference: 饮食偏好。可选：中餐 / 西餐 / 日式 / 东南亚 / 混合。
    """
    prompt = (
        f"【饮食计划请求】\n"
        f"目标：{goal}\n"
        f"每日预算：{budget_daily} 元\n"
        f"忌口：{dietary_restriction}\n"
        f"每天 {meal_count} 餐\n"
        f"偏好：{food_preference}\n\n"
        f"请生成一份每日饮食计划，包含：\n"
        f"1. 每餐的具体菜品 + 大概成本\n"
        f"2. 宏量营养素估算（蛋白质 / 碳水 / 脂肪）\n"
        f"3. 总热量范围\n"
        f"4. 省钱小技巧（菜场/外卖/预制备餐对比）"
    )
    return json.dumps(
        {"mode": "llm_self_reflect", "prompt_for_model": prompt, "tool": "plan_meal"},
        ensure_ascii=False,
    )


# ── 3. 打卡追踪建议 ────────────────────────────────────────────────


@tool
async def track_progress_hint(
    start_weight: float = 0,
    goal_weight: float = 0,
    plan_type: str = "减脂",
    existing_tools: str = "",
) -> str:
    """给打卡和进度追踪的建议。

    Args:
        start_weight: 起始体重（kg）。填 0 表示未知。
        goal_weight: 目标体重（kg）。填 0 表示未知。
        plan_type: 计划类型。可选：减脂 / 增肌 / 塑形 / 其他。
        existing_tools: 用户已经在用的追踪工具（如「微信小程序」「Notion」）。
    """
    prompt = (
        f"【打卡追踪建议请求】\n"
        f"起始体重：{start_weight or '未知'}\n"
        f"目标体重：{goal_weight or '未知'}\n"
        f"计划类型：{plan_type}\n"
        f"已有工具：{existing_tools or '（无）'}\n\n"
        f"请给出：\n"
        f"1. 推荐的追踪维度（体重 / 体脂 / 围度 / 运动量 / 饮食）\n"
        f"2. 打卡频率建议（体重一周几次、运动打卡频率）\n"
        f"3. 免费追踪工具推荐\n"
        f"4. 如何避免打卡焦虑（断卡怎么办、平台期怎么应对）\n"
        f"5. 庆祝小进步的仪式感建议"
    )
    return json.dumps(
        {"mode": "llm_self_reflect", "prompt_for_model": prompt, "tool": "track_progress_hint"},
        ensure_ascii=False,
    )


# ── SkillRegistry 注册 ────────────────────────────────────────────

FITNESS_TOOLS = [plan_workout, plan_meal, track_progress_hint]

FITNESS_PROMPT = """你是健身饮食助手，擅长制定训练计划、饮食方案和追踪建议。

⚠️ 免责声明：以下建议仅供参考，身体不适时请停止训练并咨询医生/专业教练。

可用工具：
1. plan_workout — 按目标/器材生成训练计划
2. plan_meal — 按预算/目标生成饮食计划
3. track_progress_hint — 给打卡/追踪建议

风格：
- 计划要具体到动作、组数、时间，不要模糊
- 饮食要考虑预算和实操性，不要推荐难以买到的食材
- 强调循序渐进，不要给新手太难的计划
"""


def register(r: SkillRegistry) -> None:
    """把健身饮食工具包注册进 SkillRegistry — 给 chat 档。"""
    r.register_tools("chat", FITNESS_TOOLS)
