"""stock skill 的 LangGraph tools：行情、财报、新闻。

⚠️ 强制风控：所有返回值必须追加 STOCK_RISK_WARNING。
真实数据源：AKShare（pip install akshare），免费拉 A 股/美股行情。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool

from app.agent.skills import SkillRegistry
from mcp import config as mcp_cfg

logger = logging.getLogger(__name__)


def _with_warning(data: Any) -> str:
    """把返回值转 JSON 后强制追加风控声明。"""
    if isinstance(data, str):
        # 已经是字符串，直接拼
        return data + mcp_cfg.STOCK_RISK_WARNING
    return json.dumps(data, ensure_ascii=False) + mcp_cfg.STOCK_RISK_WARNING


# ── 1. 股票实时行情（真实：AKShare；mock：mcp_cfg.MOCK_STOCK_QUOTE）────


@tool
async def get_stock_quote(symbol: str, market: str = "auto") -> str:
    """查询股票实时行情。真实数据来自 AKShare（东方财富）。

    Args:
        symbol: 股票代码。A 股如 600519（茅台）、000001（平安银行）；美股如 AAPL、TSLA。
        market: 市场标识。auto 自动识别 / A 股 / US。默认 auto。
    """
    # 先尝试真实数据源
    if not mcp_cfg.is_mock():
        try:
            import akshare as ak  # type: ignore

            # A 股代码以 6/0/3 开头（上交所/深交所/创业板）
            is_cn = market in ("A", "CN") or (
                market == "auto" and symbol.isdigit() and len(symbol) == 6
            )

            if is_cn:
                df = ak.stock_zh_a_spot_em()  # A 股实时行情
                row = df[df["代码"] == symbol]
                if row.empty:
                    raise ValueError(f"A 股代码 {symbol} 不存在")
                r = row.iloc[0]
                result = {
                    "symbol": symbol,
                    "name": str(r.get("名称", "")),
                    "price": float(r.get("最新价", 0)),
                    "change_pct": float(r.get("涨跌幅", 0)),
                    "volume": str(r.get("成交量", "")),
                    "market": "CN",
                    "data_source": "akshare",
                }
            else:
                df = ak.stock_us_spot_em()  # 美股实时行情
                row = df[df["代码"].str.upper() == symbol.upper()]
                if row.empty:
                    raise ValueError(f"美股代码 {symbol} 不存在")
                r = row.iloc[0]
                result = {
                    "symbol": symbol.upper(),
                    "name": str(r.get("名称", "")),
                    "price": float(r.get("最新价", 0)),
                    "change_pct": float(r.get("涨跌幅", 0)),
                    "volume": str(r.get("成交量", "")),
                    "market": "US",
                    "data_source": "akshare",
                }

            return _with_warning(result)
        except Exception as exc:
            logger.warning("股票行情真实 API 失败 symbol=%s: %s，降级 mock", symbol, exc)

    # mock 兜底
    quote = mcp_cfg.MOCK_STOCK_QUOTE.get(symbol.upper())
    if quote is None:
        quote = {
            "symbol": symbol.upper(),
            "name": f"{symbol.upper()}（mock）",
            "price": 100.0,
            "change_pct": 0.0,
            "volume": "1.0M",
            "market": "UNKNOWN",
            "data_source": "mock",
        }
    else:
        quote = {**quote, "data_source": "mock"}

    return _with_warning(quote)


# ── 2. 股票财务数据（mock 为主，真实 AKShare 财务接口较重）────────────


@tool
async def get_stock_financials(symbol: str) -> str:
    """查询股票关键财务指标。

    Args:
        symbol: 股票代码，如 600519、AAPL。
    """
    # AKShare 财务接口较重，且需解析复杂 DataFrame → 纯 mock 设计
    # 真实接入时：ak.stock_financial_analysis_indicator(symbol) 等
    financials = mcp_cfg.MOCK_STOCK_FINANCIALS.get(symbol.upper())
    if financials is None:
        financials = {
            "symbol": symbol.upper(),
            "revenue": "N/A",
            "net_income": "N/A",
            "roe": "N/A",
            "pe_ttm": "N/A",
            "data_source": "mock",
        }
    else:
        financials = {**financials, "data_source": "mock"}

    return _with_warning(financials)


# ── 3. 股票相关新闻（暂无免费开放 API → mock）───────────────────────


@tool
async def get_stock_news(symbol: str, limit: int = 5) -> str:
    """查询与股票相关的近期新闻。

    Args:
        symbol: 股票代码，如 AAPL、600519。
        limit: 返回条数，默认 5。
    """
    # 新闻暂无稳定免费开放 API → 直接 mock
    news = mcp_cfg.MOCK_STOCK_NEWS.get(
        symbol.upper(),
        [
            {"title": f"{symbol.upper()} 暂无公开新闻（mock）", "source": "mock", "date": "N/A"},
        ],
    )[:limit]

    result = {
        "symbol": symbol.upper(),
        "data_source": "mock",
        "news": news,
        "note": "⚠️ 当前为 mock 数据。真实新闻需要接入东方财富/新浪财经 API。",
    }
    return _with_warning(result)


# ── SkillRegistry 注册 ────────────────────────────────────────────

STOCK_TOOLS = [get_stock_quote, get_stock_financials, get_stock_news]

STOCK_PROMPT = """你是股票信息查询助手，擅长帮用户查行情、财报、新闻。

⚠️ 严格遵守：
- 只展示数据，**绝不给出买卖建议、不预测涨跌**
- 每次输出末尾会自动附带风险声明，不要自己再加
- 如果用户问「这只票能买吗」「要不要卖」，礼貌回复「我只能帮你查公开数据，投资决策请自行判断」

可用工具：
1. get_stock_quote — 实时行情（A 股/美股）
2. get_stock_financials — 关键财务指标
3. get_stock_news — 近期相关新闻
"""


def register(r: SkillRegistry) -> None:
    """把股票工具包注册进 SkillRegistry。

    - finance（行情资讯）档：主档位，给股票工具 + 风险守卫 prompt
    - chat（智能问答）档：兜底，也附带股票能力（用户在智能问答里提行情能调）
    """
    r.register_tools("finance", STOCK_TOOLS)
    r.register_prompt("finance", STOCK_PROMPT)
    r.register_tools("chat", STOCK_TOOLS)
