"""旅游规划工具包：天气、航班、酒店、攻略。

设计原则：
- **不建数据库表** — 全部走外部 API 或公开数据源
- **外部 API 需要 key 时用 mock 占位** — 等用户给真实 key 再换
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

logger = logging.getLogger(__name__)

# ── 1. 天气查询（wttr.in，完全免费，不需要 key）────────────────────


@tool
async def get_weather(city: str, days: int = 3) -> str:
    """查询城市天气预报。数据来自 wttr.in（免费、无需 API key）。

    Args:
        city: 城市名，中英文都行（如「北京」「上海」「Tokyo」）。
        days: 预报天数，1~7。默认 3 天。
    """
    days = max(1, min(days, 7))
    url = f"https://wttr.in/{city}?format=j1&n={days}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:
        logger.warning("天气查询失败 city=%s: %s", city, exc)
        return json.dumps(
            {"error": f"天气查询失败：{exc}", "city": city, "hint": "请检查城市名是否正确"},
            ensure_ascii=False,
        )

    # wttr.in 返回 JSON 结构比较深，只摘最有用的字段
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


# ── 2. 航班搜索（mock 占位，等用户给真实数据源）───────────────────


@tool
async def search_flights(
    from_city: str,
    to_city: str,
    date: str,
    passengers: int = 1,
) -> str:
    """查询航班。**当前为 mock 数据** — 返回示例航班信息，等真实 API key 到位后替换。

    Args:
        from_city: 出发城市（如「北京」「PEK」）。
        to_city: 目的城市（如「上海」「SHA」）。
        date: 出发日期，格式 YYYY-MM-DD（如 2026-12-01）。
        passengers: 乘客数，默认 1。
    """
    # TODO: 接入真实航班 API（Skyscanner / 携程开放平台 / 去哪儿）
    mock_routes = {
        ("北京", "上海"): [
            {"flight": "CA1501", "depart": "07:30", "arrive": "09:45", "price": 580, "airline": "国航"},
            {"flight": "MU5102", "depart": "14:00", "arrive": "16:20", "price": 620, "airline": "东航"},
            {"flight": "CA1523", "depart": "19:00", "arrive": "21:15", "price": 490, "airline": "国航"},
        ],
        ("上海", "北京"): [
            {"flight": "CA1502", "depart": "08:00", "arrive": "10:15", "price": 560, "airline": "国航"},
            {"flight": "MU5103", "depart": "15:30", "arrive": "17:50", "price": 610, "airline": "东航"},
        ],
    }
    key = (from_city, to_city)
    flights = mock_routes.get(key) or [
        {"flight": f"MOCK{from_city[0]}{to_city[0]}001", "depart": "09:00", "arrive": "11:30", "price": 800, "airline": "模拟航空"}
    ]

    return json.dumps(
        {
            "from": from_city,
            "to": to_city,
            "date": date,
            "passengers": passengers,
            "note": "⚠️ 当前为 mock 数据，仅作演示。真实航班查询需要接入 Skyscanner API 或携程开放平台。",
            "flights": flights,
        },
        ensure_ascii=False,
    )


# ── 3. 酒店搜索（mock 占位）───────────────────────────────────────


@tool
async def search_hotels(
    city: str,
    check_in: str,
    check_out: str,
    guests: int = 1,
    budget_max: int = 2000,
) -> str:
    """搜索酒店。**当前为 mock 数据** — 返回示例酒店信息，等真实 API key 到位后替换。

    Args:
        city: 城市名。
        check_in: 入住日期 YYYY-MM-DD。
        check_out: 退房日期 YYYY-MM-DD。
        guests: 人数，默认 1。
        budget_max: 每晚预算上限（元），默认 2000。
    """
    # TODO: 接入真实酒店 API（Booking.com / 美团酒店 / 携程）
    mock_hotels = [
        {"name": f"{city}五星级大酒店", "rating": 4.8, "price_per_night": 1288, "distance": "市中心 1.2km"},
        {"name": f"{city}商务快捷酒店", "rating": 4.5, "price_per_night": 468, "distance": "市中心 3km"},
        {"name": f"{city}青年旅舍", "rating": 4.2, "price_per_night": 128, "distance": "大学城 0.5km"},
    ]

    return json.dumps(
        {
            "city": city,
            "check_in": check_in,
            "check_out": check_out,
            "guests": guests,
            "budget_max": budget_max,
            "note": "⚠️ 当前为 mock 数据，仅作演示。真实酒店查询需要接入 Booking.com API 或携程。",
            "hotels": mock_hotels,
        },
        ensure_ascii=False,
    )


# ── 4. 攻略搜索（复用 web_search，包装成旅游专用）────────────────


@tool
async def search_travel_guide(destination: str, style: str = "自由行") -> str:
    """搜索目的地旅游攻略。覆盖景点推荐、美食、交通、避坑指南。

    Args:
        destination: 目的地（如「厦门」「京都」「三亚」）。
        style: 旅行风格，影响攻略侧重点。可选：自由行 / 亲子 / 情侣 / 穷游 / 摄影。
    """
    # 复用 web_search — 包装一个更精准的 query
    try:
        from app.agent.tools.web import web_search as _web_search

        queries = [
            f"{destination} {style}攻略 必去景点",
            f"{destination} 美食推荐 本地小吃",
            f"{destination} 交通指南 怎么去",
        ]
        all_results = []
        for q in queries:
            results = await _web_search(query=q, max_results=3)
            all_results.extend(results)
        # 去重（按 url）
        seen_urls = set()
        deduped = []
        for r in all_results:
            url = r.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                deduped.append(r)
        return json.dumps(
            {
                "destination": destination,
                "style": style,
                "guides": deduped[:8],
            },
            ensure_ascii=False,
        )
    except Exception as exc:
        logger.warning("攻略搜索失败 destination=%s: %s", destination, exc)
        return json.dumps(
            {"error": f"攻略搜索失败：{exc}", "destination": destination, "hint": "检查联网搜索是否可用"},
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
- 航班/酒店 API 目前是 mock 数据，给用户看的时候要加「当前为演示数据」的备注
- 攻略搜索优先展示高质量来源（马蜂窝、小红书、知乎专栏、携程游记）
- 输出格式：先给一个行程概览表，再给详细建议
"""


def register(r: SkillRegistry) -> None:
    """把旅游工具包注册进 SkillRegistry — 只给 travel 档。"""
    r.register_tools("travel", TRAVEL_TOOLS)
    r.register_prompt("travel", TRAVEL_PROMPT)
