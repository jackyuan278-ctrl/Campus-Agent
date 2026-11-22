"""english_helper skill 的 LangGraph tools：生词释义、朗读反馈、词汇提取。

**纯 LLM tools** — 函数体返回结构化 prompt，实际内容由模型生成。
"""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

from app.agent.skills import SkillRegistry

logger = logging.getLogger(__name__)


# ── 1. 生词释义 ───────────────────────────────────────────────────

@tool
async def explain_word(word: str, context: str = "", level: str = "CET-4") -> str:
    """解释英语单词含义，附带例句和记忆技巧。

    Args:
        word: 要解释的单词或短语。
        context: 上下文（用户在什么句子里看到的）。
        level: 用户英语水平，影响释义深度。可选：CET-4 / CET-6 / IELTS 6 / IELTS 7 / 考研 / GRE。
    """
    prompt = (
        f"【生词释义请求】\n"
        f"单词：{word}\n"
        f"上下文：{context or '（无）'}\n"
        f"用户水平：{level}\n\n"
        f"请给出：\n"
        f"1. 音标 + 词性\n"
        f"2. 核心含义（3~5 个义项，按常用度排序）\n"
        f"3. 每个义项配一个例句（带中文翻译）\n"
        f"4. 常见搭配 / 近义词 / 反义词\n"
        f"5. 记忆小技巧（词根词缀、联想记忆等）"
    )
    return json.dumps(
        {"mode": "llm_self_reflect", "prompt_for_model": prompt, "tool": "explain_word"},
        ensure_ascii=False,
    )


# ── 2. 朗读/文本反馈 ──────────────────────────────────────────────

@tool
async def read_along_feedback(user_text: str, expected_text: str = "", aspect: str = "综合") -> str:
    """对用户朗读/写作的文本给学习建议。

    Args:
        user_text: 用户朗读的转写文本或写作内容。
        expected_text: 原文（如果是跟读/跟写）。
        aspect: 反馈侧重点。可选：发音 / 流利度 / 语法 / 用词 / 综合。
    """
    prompt = (
        f"【学习反馈请求】\n"
        f"用户内容：{user_text}\n"
        f"参考原文：{expected_text or '（无，用户自由表达）'}\n"
        f"反馈侧重：{aspect}\n\n"
        f"请给出：\n"
        f"1. 整体评价（简洁一句话）\n"
        f"2. 具体问题点（逐条列出，附修正）\n"
        f"3. 2~3 条改进建议\n"
        f"4. 鼓励性总结"
    )
    return json.dumps(
        {"mode": "llm_self_reflect", "prompt_for_model": prompt, "tool": "read_along_feedback"},
        ensure_ascii=False,
    )


# ── 3. 段落抽生词 ─────────────────────────────────────────────────

@tool
async def extract_vocabulary(paragraph: str, level: str = "CET-6", max_words: int = 15) -> str:
    """从英语段落里抽取词汇表（按用户水平过滤）。

    Args:
        paragraph: 英语段落原文。
        level: 用户水平，决定抽词难度。可选：CET-4 / CET-6 / IELTS / 考研 / GRE。
        max_words: 最多抽多少个词，默认 15。
    """
    prompt = (
        f"【词汇提取请求】\n"
        f"用户水平：{level}\n"
        f"最多：{max_words} 个词\n\n"
        f"段落原文：\n```\n{paragraph}\n```\n\n"
        f"请从中提取 {max_words} 个左右的核心/生僻词，按出现顺序排列，每个包含：\n"
        f"- 单词（含词性）\n"
        f"- 中文释义\n"
        f"- 在原文中的例句（带上下文）\n"
        f"- 难度标注（基础 / 进阶 / 高级）"
    )
    return json.dumps(
        {"mode": "llm_self_reflect", "prompt_for_model": prompt, "tool": "extract_vocabulary"},
        ensure_ascii=False,
    )


# ── SkillRegistry 注册 ────────────────────────────────────────────

ENGLISH_TOOLS = [explain_word, read_along_feedback, extract_vocabulary]

ENGLISH_PROMPT = """你是英语学习助手，擅长解释生词、给口语/写作反馈、提取词汇表。

可用工具：
1. explain_word — 单词释义 + 例句 + 记忆技巧
2. read_along_feedback — 朗读/写作学习建议
3. extract_vocabulary — 从段落里抽生词表

风格要求：
- 例句贴近用户水平，不要太偏
- 反馈先给肯定再提建议，保持鼓励性
- 词汇表按难度标注，方便用户分级学习
"""


def register(r: SkillRegistry) -> None:
    """把英语学习工具包注册进 SkillRegistry — 给 chat 档。"""
    r.register_tools("chat", ENGLISH_TOOLS)
    r.register_prompt("chat", ENGLISH_PROMPT)
