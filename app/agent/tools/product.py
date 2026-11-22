"""product skill 的 LangGraph tools：商品比价、价格历史、最优推荐。

暂无开放免费比价 API → 纯 mock 数据，来自 mcp_cfg.MOCK_PRODUCTS。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool

from app.agent.skills import SkillRegistry
from mcp import config as mcp_cfg

logger = logging.getLogger(__name__)


# ── 1. 商品比价 ───────────────────────────────────────────────────


@tool
async def compare_products(keyword: str, top_k: int = 5) -> str:
    """按关键词搜索同款商品并对比多平台价格（京东/天猫/拼多多）。

    Args:
        keyword: 搜索关键词，如「降噪耳机」「MacBook」。
        top_k: 返回候选商品数，默认 5。
    """
    # 纯 mock：暂无开放 API
    matched: list[dict[str, Any]] = []
    for cat, items in mcp_cfg.MOCK_PRODUCTS.items():
        if keyword.lower() in cat.lower() or keyword.lower() in cat.lower().replace("_", ""):
            matched.extend(items)

    if not matched:
        # 通用 mock
        matched = [
            {
                "name": f"{keyword}（mock 商品）",
                "jd_price": 999,
                "tmall_price": 989,
                "pdd_price": 949,
                "rating": 4.5,
            }
        ]

    matched = matched[:top_k]

    # 算最优平台
    for p in matched:
        prices = {k: p[k] for k in ("jd_price", "tmall_price", "pdd_price") if p.get(k)}
        if prices:
            p["best_platform"] = min(prices, key=prices.get).replace("_price", "")
            p["best_price"] = min(prices.values())

    return json.dumps(
        {
            "keyword": keyword,
            "data_source": "mock",
            "note": "⚠️ 当前为 mock 数据。真实比价需要接入各电商平台开放 API。",
            "products": matched,
        },
        ensure_ascii=False,
    )


# ── 2. 价格历史（纯 mock）────────────────────────────────────────


@tool
async def get_product_price_history(product_name: str, days: int = 30) -> str:
    """查询商品近期价格走势。

    Args:
        product_name: 商品名称，如 "Sony WH-1000XM5"。
        days: 查询天数，默认 30。
    """
    # mock：生成一条假走势
    import random

    base_price = 2500 if "Sony" in product_name or "XM5" in product_name else 1000
    history = []
    for i in range(days, 0, -1):
        price = base_price + random.randint(-100, 100)
        history.append({"day": f"day_{i}", "price": price})

    return json.dumps(
        {
            "product_name": product_name,
            "days": days,
            "data_source": "mock",
            "note": "⚠️ 当前为 mock 数据。真实价格历史需要接入电商平台爬虫或开放 API。",
            "history": history,
            "current_price": base_price + random.randint(-50, 50),
            "trend": "stable",  # stable / up / down
        },
        ensure_ascii=False,
    )


# ── 3. 最优购买时机（纯 mock）────────────────────────────────────


@tool
async def find_best_deal(category: str, budget_min: float = 0, budget_max: float = 99999) -> str:
    """按品类推荐当前最划算的商品。

    Args:
        category: 商品品类，如「耳机」「笔记本」。
        budget_min: 预算下限，默认 0。
        budget_max: 预算上限，默认不限。
    """
    matched: list[dict[str, Any]] = []
    for cat, items in mcp_cfg.MOCK_PRODUCTS.items():
        if category in cat or cat in category:
            matched.extend(items)

    # 按预算过滤
    filtered = [
        p
        for p in matched
        if budget_min
        <= min(p.get("jd_price", 0), p.get("tmall_price", 0), p.get("pdd_price", 0))
        <= budget_max
    ]

    if not filtered:
        filtered = [
            {
                "name": f"{category} 推荐款（mock）",
                "jd_price": 599,
                "tmall_price": 579,
                "pdd_price": 549,
                "rating": 4.6,
            }
        ]

    # 找最低价的
    for p in filtered:
        prices = {k: p[k] for k in ("jd_price", "tmall_price", "pdd_price") if p.get(k)}
        if prices:
            p["best_platform"] = min(prices, key=prices.get).replace("_price", "")
            p["best_price"] = min(prices.values())

    return json.dumps(
        {
            "category": category,
            "budget_min": budget_min,
            "budget_max": budget_max,
            "data_source": "mock",
            "note": "⚠️ 当前为 mock 数据。真实比价需要接入电商平台开放 API。",
            "recommendations": filtered[:3],
            "tip": "通常拼多多价格最低，京东/天猫售后更有保障，建议综合考虑。",
        },
        ensure_ascii=False,
    )


# ── SkillRegistry 注册 ────────────────────────────────────────────

PRODUCT_TOOLS = [compare_products, get_product_price_history, find_best_deal]

PRODUCT_PROMPT = """你是商品比价助手，擅长帮用户对比多平台价格、查价格走势、找最划算购买时机。

⚠️ 当前所有数据都是 mock 示例，展示给用户时要加「当前为演示数据」的备注。

可用工具：
1. compare_products — 关键词搜索比价
2. get_product_price_history — 价格历史走势
3. find_best_deal — 按品类找最优推荐

输出风格：先给一个比价表格，再给购买建议。
"""


def register(r: SkillRegistry) -> None:
    """把商品比价工具包注册进 SkillRegistry — 给 chat 档。"""
    r.register_tools("chat", PRODUCT_TOOLS)
    r.register_prompt("chat", PRODUCT_PROMPT)
