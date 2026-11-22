"""MCP 全局配置 + mock 数据集中管理。

**设计原则**：
- 社区成熟 MCP 优先（wttr.in 免费天气、AKShare 免费股票、Semantic Scholar 免费文献）
- 没有成熟免费 MCP 的（航班/酒店/比价）→ 纯 mock
- 所有外部 API key 环境变量占位，留空等待配置
- MOCK_MODE=true 时全部走 mock，不访问真实外网

社区 MCP 映射（全部国内能直接用）：
    天气 → wttr.in (免费 HTTP，直接 httpx)
    股票 → AKShare (pip install akshare，A 股/美股/港股)
    文献 → Semantic Scholar API (api.semanticscholar.org)
    联网搜索 → 复用已有 tavily
"""

from __future__ import annotations

import json
import os
from typing import Any, Final

# ── Mock 总开关 ────────────────────────────────────────────────────
MOCK_MODE: Final[bool] = os.environ.get("MOCK_MODE", "true").lower() in ("true", "1", "yes")


# ── 环境变量占位（全部留空等待配置）───────────────────────────────
SEMANTIC_SCHOLAR_API_KEY: Final[str] = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
LOCAL_KNOWLEDGE_DB_PATH: Final[str] = os.environ.get("LOCAL_KNOWLEDGE_DB_PATH", "")


def is_mock() -> bool:
    return MOCK_MODE


# ══════════════════════════════════════════════════════════════════
#                       MOCK 数据（集中管理）
# ══════════════════════════════════════════════════════════════════

MOCK_WEATHER: dict[str, Any] = {
    "北京": {
        "current": {"temp_c": 8, "feels_c": 5, "humidity": 35, "desc": "晴"},
        "forecast": [
            {"date": "2026-11-23", "max_temp_c": 12, "min_temp_c": 2, "desc": "晴"},
            {"date": "2026-11-24", "max_temp_c": 14, "min_temp_c": 3, "desc": "多云"},
            {"date": "2026-11-25", "max_temp_c": 11, "min_temp_c": 0, "desc": "大风"},
        ],
    },
    "上海": {
        "current": {"temp_c": 18, "feels_c": 16, "humidity": 72, "desc": "小雨"},
        "forecast": [
            {"date": "2026-11-23", "max_temp_c": 20, "min_temp_c": 14, "desc": "多云"},
            {"date": "2026-11-24", "max_temp_c": 22, "min_temp_c": 15, "desc": "阴"},
            {"date": "2026-11-25", "max_temp_c": 19, "min_temp_c": 13, "desc": "小雨"},
        ],
    },
    "厦门": {
        "current": {"temp_c": 24, "feels_c": 25, "humidity": 78, "desc": "晴"},
        "forecast": [
            {"date": "2026-11-23", "max_temp_c": 26, "min_temp_c": 19, "desc": "晴"},
            {"date": "2026-11-24", "max_temp_c": 27, "min_temp_c": 20, "desc": "晴"},
            {"date": "2026-11-25", "max_temp_c": 25, "min_temp_c": 18, "desc": "多云"},
        ],
    },
}

MOCK_FLIGHTS: dict[str, Any] = {
    "北京_上海": [
        {"flight": "CA1501", "depart": "07:30", "arrive": "09:45", "price": 580, "airline": "国航"},
        {"flight": "MU5102", "depart": "14:00", "arrive": "16:20", "price": 620, "airline": "东航"},
        {"flight": "CA1523", "depart": "19:00", "arrive": "21:15", "price": 490, "airline": "国航"},
    ],
    "上海_北京": [
        {"flight": "CA1502", "depart": "08:00", "arrive": "10:15", "price": 560, "airline": "国航"},
        {"flight": "MU5103", "depart": "15:30", "arrive": "17:50", "price": 610, "airline": "东航"},
    ],
    "北京_厦门": [
        {"flight": "CA1895", "depart": "08:30", "arrive": "11:55", "price": 890, "airline": "国航"},
        {"flight": "MF8172", "depart": "14:20", "arrive": "17:40", "price": 750, "airline": "厦航"},
    ],
}

MOCK_HOTELS: list[dict[str, Any]] = [
    {"name": "五星级大酒店", "rating": 4.8, "price_per_night": 1288, "distance": "市中心 1.2km"},
    {"name": "商务快捷酒店", "rating": 4.5, "price_per_night": 468, "distance": "市中心 3km"},
    {"name": "青年旅舍", "rating": 4.2, "price_per_night": 128, "distance": "大学城 0.5km"},
]

MOCK_GUIDES: dict[str, Any] = {
    "厦门": [
        {
            "title": "厦门 3 天自由行攻略",
            "url": "https://mafengwo.cn/example",
            "content": "Day1：鼓浪屿 → 三丘田码头上船...",
            "score": 4.9,
        },
        {
            "title": "厦门必吃美食 Top 10",
            "url": "https://xiaohongshu.com/example",
            "content": "沙茶面、海蛎煎、土笋冻...",
            "score": 4.8,
        },
        {
            "title": "厦门交通指南",
            "url": "https://zhihu.com/example",
            "content": "厦门地铁 1/2/3 号线覆盖主要景点...",
            "score": 4.5,
        },
    ],
}

# ⚠️ 风控：mock stock 数据里**绝对不能出现**预测/荐股/买卖建议
MOCK_STOCK_QUOTE: dict[str, Any] = {
    "AAPL": {
        "symbol": "AAPL",
        "name": "Apple Inc.",
        "price": 215.82,
        "change_pct": 1.23,
        "volume": "52.3M",
        "market": "US",
    },
    "600519": {
        "symbol": "600519",
        "name": "贵州茅台",
        "price": 1728.50,
        "change_pct": -0.85,
        "volume": "2.1M",
        "market": "SH",
    },
    "000001": {
        "symbol": "000001",
        "name": "平安银行",
        "price": 11.35,
        "change_pct": 0.53,
        "volume": "85.6M",
        "market": "SZ",
    },
}

MOCK_STOCK_FINANCIALS: dict[str, Any] = {
    "AAPL": {
        "symbol": "AAPL",
        "revenue_2024": "$391.04B",
        "net_income_2024": "$93.74B",
        "roe_2024": 147.3,
        "pe_ttm": 34.2,
    },
    "600519": {
        "symbol": "600519",
        "revenue_2024": "1585亿元",
        "net_income_2024": "763亿元",
        "roe_2024": 32.5,
        "pe_ttm": 28.1,
    },
}

MOCK_STOCK_NEWS: dict[str, Any] = {
    "AAPL": [
        {"title": "Apple 发布 Q4 财报，营收同比增长 5%", "source": "Reuters", "date": "2026-11-18"},
        {"title": "Apple Car 项目据称 2028 年量产", "source": "Bloomberg", "date": "2026-11-15"},
    ],
}

# ⚠️ 风控：股票输出必须附带以下风险声明（强制拼入返回值）
STOCK_RISK_WARNING: Final[str] = (
    "\n\n⚠️ 股市有风险，投资需谨慎。"
    "以上信息仅为公开数据整理，不构成任何投资建议。"
    "历史表现不代表未来收益，请自行判断。"
)

MOCK_PAPERS: list[dict[str, Any]] = [
    {
        "paper_id": "649def34f8be52c8b66281af98ae884c09aef38b",
        "title": "Attention Is All You Need",
        "authors": [
            "Ashish Vaswani",
            "Noam Shazeer",
            "Niki Parmar",
            "Jakob Uszkoreit",
            "Llion Jones",
            "Aidan N. Gomez",
            "Łukasz Kaiser",
            "Illia Polosukhin",
        ],
        "year": 2017,
        "venue": "NeurIPS",
        "citation_count": 128432,
        "abstract": "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely...",
    },
    {
        "paper_id": "abc123",
        "title": "LangGraph: Building Stateful Multi-Agent Applications with LLMs",
        "authors": ["LangChain Team"],
        "year": 2024,
        "venue": "arXiv",
        "citation_count": 450,
        "abstract": "LangGraph is a library for building stateful, multi-actor applications with LLMs...",
    },
]

MOCK_PRODUCTS: dict[str, Any] = {
    "耳机_降噪": [
        {
            "name": "Sony WH-1000XM5",
            "jd_price": 2499,
            "tmall_price": 2399,
            "pdd_price": 2299,
            "rating": 4.9,
        },
        {
            "name": "Bose QC45",
            "jd_price": 2299,
            "tmall_price": 2199,
            "pdd_price": 2099,
            "rating": 4.8,
        },
        {
            "name": "AirPods Max",
            "jd_price": 4299,
            "tmall_price": 4199,
            "pdd_price": 4099,
            "rating": 4.7,
        },
    ],
    "笔记本": [
        {
            "name": "MacBook Pro 14 M3 Pro",
            "jd_price": 14999,
            "tmall_price": 14799,
            "pdd_price": 14599,
            "rating": 4.9,
        },
        {
            "name": "联想 ThinkPad X1 Carbon",
            "jd_price": 12999,
            "tmall_price": 12799,
            "pdd_price": 12599,
            "rating": 4.7,
        },
    ],
}
