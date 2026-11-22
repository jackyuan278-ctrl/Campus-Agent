"""travel skill 的 LangGraph tools：天气、航班、酒店、攻略。

设计原则：
- **不建数据库表** — 全部走外部 API 或 mock 数据源
- **真实优先，mock 兜底** — wttr.in 免费天气、其余暂无开放 API
- **全部异步** — 跟现有工具保持一致
- **返回值必须 JSON 可序列化** — Decimal/时间全部转基础类型
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from langchain_core.tools import tool

from app.agent.skills import SkillRegistry
from mcp import config as mcp_cfg  # mock 数据集中在这里

logger = logging.getLogger(__name__)


# ── 1. 天气查询（真实：wttr.in；mock：mcp_cfg.MOCK_WEATHER）────────

@tool
async def get_weather(city: str, days: int = 3) -> str:
    """查询城市天气预报。真实数据来自 wttr.in（免费、无需 API key）。

    Args:
        city: 城市名，中英文都行（如「北京」「上海」「Tokyo」）。
        days: 预报天数，1~7。默认 3 天。
    """
    days = max(1, min(days, 7))

    # 先尝试真实数据源
    if not mcp_cfg.is_mock():
        try:
            url = f"https://wttr.in/{city}?format=j1&n={days}"
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()

            current = data.get("current_condition", [{}])[0]
            forecasts = []
            for day in (data.get("weather") or [])[:days]:
                forecasts.append({
                    "date": day.get("date"),
                    "max_temp_c": day.get("maxtempC"),
                    "min_temp_c": day.get("mintempC"),
                    "sunrise": day.get("astronomy", [{}])[0].get("sunrise"),
                    "sunset": day.get("astronomy", [{}])[0].get("sunset"),
                    "desc": (day.get("hourly", [{}])[4].get("weatherDesc") or [{}])[0].get("value"),
                })

            return json.dumps(
                {
                    "city": city,
                    "data_source": "wttr.in",
                    "current": {
                        "temp_c": current.get("temp_C"),
                        "feels_c": current.get("FeelsLikeC"),
                        "humidity": current.get("humidity"),
                        "desc": (current.get("weatherDesc") or [{}])[0].get("value"),
                    },
                    "forecast": forecasts,
                },
                ensure_ascii=False,
            )
        except Exception as exc:
            logger.warning("天气真实 API 失败 city=%s: %s，降级 mock", city, exc)

    # mock 兜底
    mock_city = mcp_cfg.MOCK_WEATHER.get(city)
    if mock_city is None:
        # 模糊匹配：北京/上海等有预设，其他城市返回通用 mock
        mock_city = {
            "current": {"temp_c": 20, "feels_c": 18, "humidity": 60, "desc": "多云"},
            "forecast": [
                {"date": "2026-11-23", "max_temp_c": 22, "min_temp_c": 14, "desc": "晴"},
                {"date": "2026-11-24", "max_temp_c": 21, "min_temp_c": 13, "desc": "多云"},
                {"date": "2026-11-25", "max_temp_c": 19, "min_temp_c": 12, "desc": "阴"},
            ],
        }

    forecast_trimmed = mock_city.get("forecast", [])[:days]
    return json.dumps(
        {
            "city": city,
            "data_source": "mock",
            "current": mock_city.get("current", {}),
            "forecast": forecast_trimmed,
            "note": "⚠️ 当前为 mock 数据，仅作演示。",
        },
        ensure_ascii=False,
    )


# ── 2. 航班搜索（暂无真实 API → 纯 mock，来自 mcp_cfg.MOCK_FLIGHTS）────

@tool
async def search_flights(
    from_city: str,
    to_city: str,
    date: str,
    passengers: int = 1,
) -> str:
    """查询航班。暂无稳定免费开放 API → 纯 mock 设计，等真实 API key 到位后替换。

    Args:
        from_city: 出发城市（如「北京」「PEK」）。
        to_city: 目的城市（如「上海」「SHA」）。
        date: 出发日期，格式 YYYY-MM-DD（如 2026-12-01）。
        passengers: 乘客数，默认 1。
    """
    # 纯 mock 数据源（Skyscanner / 携程开放平台均需付费 key，暂不接入）
    route_key = f"{from_city}_{to_city}"
    flights = mcp_cfg.MOCK_FLIGHTS.get(route_key)
    if flights is None:
        # 通用 mock
        flights = [
            {
                "flight": f"MOCK{from_city[0]}{to_city[0]}001",
                "depart": "09:00",
                "arrive": "11:30",
                "price": 800,
                "airline": "模拟航空",
            }
        ]

    return json.dumps(
        {
            "from": from_city,
            "to": to_city,
            "date": date,
            "passengers": passengers,
            "data_source": "mock",
            "note": "⚠️ 当前为 mock 数据，仅作演示。真实航班查询需要接入 Skyscanner API 或携程开放平台。",
            "flights": flights,
        },
        ensure_ascii=False,
    )


# ── 3. 酒店搜索（暂无真实 API → 纯 mock，来自 mcp_cfg.MOCK_HOTELS）────

@tool
async def search_hotels(
    city: str,
    check_in: str,
    check_out: str,
    guests: int = 1,
    budget_max: int = 2000,
) -> str:
    """搜索酒店。暂无开放真实 API → 纯 mock 设计，等真实 API key 到位后替换。

    Args:
        city: 城市名。
        check_in: 入住日期 YYYY-MM-DD。
        check_out: 退房日期 YYYY-MM-DD。
        guests: 人数，默认 1。
        budget_max: 每晚预算上限（元），默认 2000。
    """
    # 纯 mock 数据源（Booking.com / 携程均需付费 key，暂不接入）
    mock_hotels = [
        {**h, "name": f"{city}{h['name']}"}
        for h in mcp_cfg.MOCK_HOTELS
        if h["price_per_night"] <= budget_max
    ]
    if not mock_hotels:
        # 预算太低时返回最贵的一个（让用户知道有什么）
        mock_hotels = [
            {**h, "name": f"{city}{h['name']}"}
            for h in sorted(mcp_cfg.MOCK_HOTELS, key=lambda x: x["price_per_night"])[:1]
        ]

    return json.dumps(
        {
            "city": city,
            "check_in": check_in,
            "check_out": check_out,
            "guests": guests,
            "budget_max": budget_max,
            "data_source": "mock",
            "note": "⚠️ 当前为 mock 数据，仅作演示。真实酒店查询需要接入 Booking.com API 或携程。",
            "hotels": mock_hotels,
        },
        ensure_ascii=False,
    )


# ── 4. 攻略搜索（真实：复用 web_search；mock：mcp_cfg.MOCK_GUIDES）──────

@tool
async def search_travel_guide(destination: str, style: str = "自由行") -> str:
    """搜索目的地旅游攻略。覆盖景点推荐、美食、交通、避坑指南。

    Args:
        destination: 目的地（如「厦门」「京都」「三亚」）。
        style: 旅行风格，影响攻略侧重点。可选：自由行 / 亲子 / 情侣 / 穷游 / 摄影。
    """
    # 先尝试真实数据源（复用 web_search）
    if not mcp_cfg.is_mock():
        try:
            from app.agent.tools.web import web_search_tool as _web_search

            queries = [
                f"{destination} {style}攻略 必去景点",
                f"{destination} 美食推荐 本地小吃",
                f"{destination} 交通指南 怎么去",
            ]
            all_results: list[dict[str, Any]] = []
            for q in queries:
                raw = await _web_search.ainvoke({"query": q, "max_results": 3})
                if isinstance(raw, list):
                    all_results.extend(raw)

            # 去重（按 url）
            seen_urls = set()
            deduped: list[dict[str, Any]] = []
            for r in all_results:
                url = r.get("url")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    deduped.append(r)

            if deduped:
                return json.dumps(
                    {
                        "destination": destination,
                        "style": style,
                        "data_source": "web_search",
                        "guides": deduped[:8],
                    },
                    ensure_ascii=False,
                )
        except Exception as exc:
            logger.warning("攻略搜索失败 destination=%s: %s，降级 mock", destination, exc)

    # mock 兜底
    guides = mcp_cfg.MOCK_GUIDES.get(destination, [
        {"title": f"{destination} {style}攻略（mock）", "url": "", "content": "暂无攻略数据", "score": 0},
    ])
    return json.dumps(
        {
            "destination": destination,
            "style": style,
            "data_source": "mock",
            "note": "⚠️ 当前为 mock 数据，仅作演示。",
            "guides": guides,
        },
        ensure_ascii=False,
    )


# ── SkillRegistry 注册 ────────────────────────────────────────────

TRAVEL_TOOLS = [get_weather, search_flights, search_hotels, search_travel_guide]

TRAVEL_PROMPT = """你是旅游规划助手，擅长帮用户一键搞定旅行全流程：
1. 查天气预报（get_weather）
2. 查航班（search_flights）
3. 查酒店（search_hotels）
4. 搜攻略（search_travel_guide）

规划原则：
- 根据用户给出的目的地、时间、预算，**主动**把天气→航班→酒店→攻略这四步都查一遍，不要等用户逐个指令
- 航班/酒店目前是 mock 数据，给用户看的时候要加「当前为演示数据」的备注
- 攻略搜索优先展示高质量来源（马蜂窝、小红书、知乎专栏、携程游记）
- 输出格式：先给一个行程概览表，再给详细建议
"""


def register(r: SkillRegistry) -> None:
    """把旅游工具包注册进 SkillRegistry.

    - "travel" 档：独立档位，只给旅游工具 + 旅游 prompt（前端点"旅游规划"按钮进这里）
    - "chat" 档：chat 是通用问答，也附带旅游能力（用户在智能问答里自然提到旅游也能调）
    """
    r.register_tools("travel", TRAVEL_TOOLS)
    r.register_prompt("travel", TRAVEL_PROMPT)
    r.register_tools("chat", TRAVEL_TOOLS)
