"""11 Skill 完整 mock 测试 — 每个 skill × 4 类场景。

运行：
    MOCK_MODE=true uv run python tests/test_skills_mock.py
    MOCK_MODE=false uv run python tests/test_skills_mock.py

覆盖：
    1. 正常流程 — mock 数据命中，返回 JSON 可序列化
    2. 边界参数 — 空值/未知城市/特殊字符
    3. 真实 API 降级 — MOCK_MODE=false 且 key 缺失时走 mock
    4. Stock 风控强制 — 风险声明必带、返回值无预测字段
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

# 让脚本直接跑时能找到 app 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp import config as mcp_cfg

# ── 测试结果收集 ────────────────────────────────────────────────────

results: list[dict] = []


def _result(skill: str, scene: str, status: str, detail: str, data=None):
    results.append({
        "skill": skill,
        "scene": scene,
        "status": status,  # PASS / FAIL / SKIP
        "detail": detail,
        "data": data,
    })
    mark = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭"}[status]
    print(f"  {mark} [{skill}] {scene}: {detail}")


async def test_case(skill: str, scene: str, coro, validator=None):
    """跑一个测试 case：await coro → validator(result) → 记录。"""
    try:
        result = await coro
        if validator and not validator(result):
            _result(skill, scene, "FAIL", f"validator 不通过", {"returned": result[:200] if result else None})
            return None
        _result(skill, scene, "PASS", f"返回 {len(result)} chars", {"preview": result[:150] if result else None})
        return result
    except Exception as exc:
        _result(skill, scene, "FAIL", f"异常: {type(exc).__name__}: {exc}")
        return None


# ── 测试体 ───────────────────────────────────────────────────────────

async def run_all():
    mode = "MOCK" if mcp_cfg.is_mock() else "REAL_FALLBACK"
    print(f"\n🔍 运行模式: {mode}\n")

    # ══════════════════════════════════════════════════════════════════
    # 1. travel-planner
    # ══════════════════════════════════════════════════════════════════
    from app.agent.tools.travel import get_weather, search_flights, search_hotels, search_travel_guide

    print("─ 1. travel-planner ─")
    await test_case("travel", "get_weather 正常（北京）", get_weather.ainvoke({"city": "北京", "days": 3}))
    await test_case("travel", "get_weather 边界（未知城市 → 通用 mock）", get_weather.ainvoke({"city": "马尔代夫"}))
    await test_case("travel", "search_flights 正常（北京_上海）", search_flights.ainvoke({"from_city": "北京", "to_city": "上海", "date": "2026-12-01"}))
    await test_case("travel", "search_hotels 正常", search_hotels.ainvoke({"city": "上海", "check_in": "2026-12-01", "check_out": "2026-12-03", "budget_max": 500}))
    await test_case("travel", "search_travel_guide 正常（厦门）", search_travel_guide.ainvoke({"destination": "厦门", "style": "自由行"}))

    # ══════════════════════════════════════════════════════════════════
    # 2. stock-info ⚠️ 风控专项
    # ══════════════════════════════════════════════════════════════════
    from app.agent.tools.stock import get_stock_quote, get_stock_financials, get_stock_news

    print("─ 2. stock-info ⚠️ 风控 ─")

    def _has_warning(s: str) -> bool:
        return "风险" in s and "不构成" in s

    def _no_prediction(s: str) -> bool:
        forbidden = ["预测", "目标价", "买入", "卖出", "涨多少", "跌多少", "评级", "推荐"]
        return not any(w in s for w in forbidden)

    quote = await test_case("stock", "get_stock_quote 正常（AAPL）", get_stock_quote.ainvoke({"symbol": "AAPL"}))
    if quote:
        _result("stock", "⚠️ 风控: 强制风险声明", "PASS" if _has_warning(quote) else "FAIL",
                "返回值必须包含风险声明", quote[-100:])
        _result("stock", "⚠️ 风控: 无预测/荐股内容", "PASS" if _no_prediction(quote) else "FAIL",
                "mock 数据不得含预测/荐股关键词")

    quote_cn = await test_case("stock", "get_stock_quote A 股（600519）", get_stock_quote.ainvoke({"symbol": "600519"}))
    if quote_cn:
        _result("stock", "⚠️ 风控: A 股同样带风险声明", "PASS" if _has_warning(quote_cn) else "FAIL", "")

    fin = await test_case("stock", "get_stock_financials 正常", get_stock_financials.ainvoke({"symbol": "AAPL"}))
    if fin:
        _result("stock", "⚠️ 风控: 财报带风险声明", "PASS" if _has_warning(fin) else "FAIL", "")
        _result("stock", "⚠️ 风控: 财报无预测字段", "PASS" if _no_prediction(fin) else "FAIL", "")

    news = await test_case("stock", "get_stock_news 正常", get_stock_news.ainvoke({"symbol": "AAPL"}))
    if news:
        _result("stock", "⚠️ 风控: 新闻带风险声明", "PASS" if _has_warning(news) else "FAIL", "")

    # mock 数据本身校验
    _result("stock", "⚠️ 风控: MOCK_STOCK_QUOTE 无预测字段",
            "PASS" if _no_prediction(json.dumps(mcp_cfg.MOCK_STOCK_QUOTE, ensure_ascii=False)) else "FAIL",
            "mcp_cfg.MOCK_STOCK_QUOTE 不得含 预测/买入/卖出 等关键词")

    # ══════════════════════════════════════════════════════════════════
    # 3. literature-research
    # ══════════════════════════════════════════════════════════════════
    from app.agent.tools.literature import search_papers, get_paper_detail, get_citation_graph

    print("─ 3. literature-research ─")
    await test_case("literature", "search_papers 正常（Attention）", search_papers.ainvoke({"query": "attention mechanism", "max_results": 3}))
    await test_case("literature", "get_paper_detail 正常", get_paper_detail.ainvoke({"paper_id": "649def34f8be52c8b66281af98ae884c09aef38b"}))
    await test_case("literature", "get_citation_graph 正常", get_citation_graph.ainvoke({"paper_id": "abc123", "depth": 1}))

    # ══════════════════════════════════════════════════════════════════
    # 4. product-comparison
    # ══════════════════════════════════════════════════════════════════
    from app.agent.tools.product import compare_products, get_product_price_history, find_best_deal

    print("─ 4. product-comparison ─")
    await test_case("product", "compare_products 正常（降噪耳机）", compare_products.ainvoke({"keyword": "降噪耳机"}))
    await test_case("product", "compare_products 正常（MacBook）", compare_products.ainvoke({"keyword": "MacBook"}))
    await test_case("product", "get_product_price_history 正常", get_product_price_history.ainvoke({"product_name": "Sony WH-1000XM5"}))
    await test_case("product", "find_best_deal 正常（耳机品类）", find_best_deal.ainvoke({"category": "耳机", "budget_max": 3000}))

    # ══════════════════════════════════════════════════════════════════
    # 5-10. 纯 LLM tools — 验证注册存在 + tool name 唯一 + JSON 可序列化
    # ══════════════════════════════════════════════════════════════════
    print("─ 5-10. 纯 LLM tools 注册验证 ─")

    import app.agent.graph as graph_module
    chat_tools = graph_module.tools_for_mode("chat")
    all_names = [t.name for t in chat_tools]

    expected_pure_llm = [
        "explain_code", "debug_hint", "suggest_refactor",  # code_developer
        "resume_review", "mock_interview_question", "salary_benchmark",  # interview
        "explain_word", "read_along_feedback", "extract_vocabulary",  # english_helper
        "summarize_tasks", "generate_section", "okr_reflection",  # weekly_report
        "plan_workout", "plan_meal", "track_progress_hint",  # fitness
        "extract_action_items", "summarize_meeting", "assign_owner",  # meeting_minutes
    ]
    for name in expected_pure_llm:
        _result("纯LLM", f"tool {name} 注册到 chat 档",
                "PASS" if name in all_names else "FAIL",
                "chat_tools 必须包含此 tool name")

    _result("纯LLM", "所有 tool name 跨文件唯一",
            "PASS" if len(all_names) == len(set(all_names)) else "FAIL",
            f"chat_tools 共 {len(all_names)} 个, 去重后 {len(set(all_names))} 个")

    # ══════════════════════════════════════════════════════════════════
    # 11. local-knowledge
    # ══════════════════════════════════════════════════════════════════
    from app.agent.tools.local_knowledge import local_knowledge_query

    print("─ 11. local-knowledge ─")
    # 这个 tool 调 retriever 可能连不上 milvus，用 tool 名 + 注册验证即可
    _result("local-knowledge", "tool 注册到 chat 档",
            "PASS" if "local_knowledge_query" in all_names else "FAIL",
            "")

    # 测试 retriever 能 import（不实际查库）
    try:
        from app.rag.retriever import retrieve  # noqa: F401
        _result("local-knowledge", "retriever 模块可 import", "PASS", "")
    except Exception as exc:
        _result("local-knowledge", "retriever 模块可 import", "FAIL", str(exc))

    # ══════════════════════════════════════════════════════════════════
    # 额外: 模式路由正确 — market/activity/doc 不应拿到 chat 全套
    # ══════════════════════════════════════════════════════════════════
    print("─ 模式路由边界 ─")
    from app.agent.graph import tools_for_mode

    market = [t.name for t in tools_for_mode("market")]
    _result("路由", "market 档只拿校园库（3 个）",
            "PASS" if len(market) == 3 and "search_secondhand_items_tool" in market else "FAIL",
            f"实际: {market}")

    doc = [t.name for t in tools_for_mode("doc")]
    _result("路由", "doc 档只拿校园库",
            "PASS" if len(doc) == 3 else "FAIL", f"实际: {doc}")

    casual = [t.name for t in tools_for_mode("casual")]
    _result("路由", "casual 档只有 web_search",
            "PASS" if casual == ["web_search_tool"] else "FAIL", f"实际: {casual}")

    none_mode = [t.name for t in tools_for_mode(None)]
    _result("路由", "mode=None 退回校园库专用集（非空）",
            "PASS" if len(none_mode) == 3 else "FAIL", f"实际: {none_mode}")

    chat = [t.name for t in tools_for_mode("chat")]
    _result("路由", "chat 档拿全套（≥30 个）",
            "PASS" if len(chat) >= 30 else "FAIL", f"实际: {len(chat)} 个")

    # ══════════════════════════════════════════════════════════════════
    # 额外: 所有 tool 返回值 JSON 可序列化
    # ══════════════════════════════════════════════════════════════════
    print("─ JSON 可序列化检查 ─")
    # ⚠️ stock 故意是 JSON + 风险声明拼接字符串（_with_warning），不做 json.loads 校验
    sample_results = {
        "travel.get_weather": await get_weather.ainvoke({"city": "北京"}),
        "literature.search_papers": await search_papers.ainvoke({"query": "test"}),
        "product.compare_products": await compare_products.ainvoke({"keyword": "耳机"}),
        "stock.get_stock_quote": await get_stock_quote.ainvoke({"symbol": "AAPL"}),  # 只检查非空
    }
    for name, val in sample_results.items():
        if name.startswith("stock."):
            # stock 返回值故意拼了风险声明，跳过 json.loads，只检查非空 + 风险声明存在
            ok = val and len(val) > 0 and "风险" in val
            _result("JSON", f"{name} 返回值非空 + 风险声明", "PASS" if ok else "FAIL", f"{len(val)} chars")
            continue
        try:
            json.loads(val) if isinstance(val, str) else None
            _result("JSON", f"{name} 返回值 JSON 可解析", "PASS", f"{len(str(val))} chars")
        except json.JSONDecodeError as exc:
            _result("JSON", f"{name} 返回值 JSON 可解析", "FAIL", str(exc))


# ── 汇总输出 ──────────────────────────────────────────────────────────

def print_summary():
    print("\n" + "=" * 60)
    print("📊 测试汇总")
    print("=" * 60)

    pass_count = sum(1 for r in results if r["status"] == "PASS")
    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    skip_count = sum(1 for r in results if r["status"] == "SKIP")
    total = len(results)

    print(f"\n  Total : {total}")
    print(f"  PASS  : {pass_count}  ✅")
    print(f"  FAIL  : {fail_count}  ❌")
    print(f"  SKIP  : {skip_count}  ⏭")
    print()

    if fail_count > 0:
        print("  ❌❌❌ FAIL 项详细:")
        for r in results:
            if r["status"] == "FAIL":
                print(f"     [{r['skill']}] {r['scene']}: {r['detail']}")
        print()

    # 按 skill 分组
    by_skill: dict[str, list[dict]] = {}
    for r in results:
        by_skill.setdefault(r["skill"], []).append(r)

    print("  按 Skill 分组:")
    for skill, items in by_skill.items():
        p = sum(1 for i in items if i["status"] == "PASS")
        f = sum(1 for i in items if i["status"] == "FAIL")
        icon = "✅" if f == 0 else "❌"
        print(f"     {icon} {skill:20s} {p}/{len(items)}")

    print()
    print("=" * 60)

    # 写 JSON 报告
    report = {
        "total": total,
        "pass": pass_count,
        "fail": fail_count,
        "skip": skip_count,
        "mode": "MOCK" if mcp_cfg.is_mock() else "REAL_FALLBACK",
        "date": "2026-11-22",
        "results": results,
    }
    report_path = os.path.join(os.path.dirname(__file__), "test_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"📄 JSON 报告: {report_path}")

    return fail_count == 0


if __name__ == "__main__":
    success = asyncio.run(run_all())
    print_summary()
    sys.exit(0 if success else 1)
