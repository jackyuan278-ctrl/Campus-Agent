---
name: stock-info
description: 用户要查股票行情、财报、公告时触发。触发词：股票、股价、财报、分红、行情。排除：任何涉及预测/荐股/买卖建议的请求，必须拒绝并提示"我只提供公开数据，不做投资建议"。
---

# stock-info

⚠️ **强制风控**：本 skill 只提供已公开的行情/财报/公告数据，**绝对不做任何预测、荐股或买卖建议**。

## 快速用法

```python
# 查询当前行情（延迟 15 分钟或实时，标注数据源）
quote = await get_stock_quote(symbol="600519.SH")

# 查询历史 K 线
bars = await get_kline(symbol="600519.SH", period="day", count=60)

# 查询最新公告
news = await get_announcements(symbol="600519.SH", limit=5)

# 查询最近财报
financials = await get_financials(symbol="600519.SH", report_type="年报")
```

## 工具清单

| tool_name | 功能 | 数据源 |
|---|---|---|
| `get_stock_quote` | 当前价 / 成交量 / 涨跌幅 | 新浪财经 / 东方财富公开接口 |
| `get_kline` | 日/周/月 K 线序列 | 同上 |
| `get_announcements` | 公司公告列表 + 摘要 | 巨潮资讯 |
| `get_financials` | 财报关键指标（营收/净利润/ROE） | 同上 |
| `get_dividend_history` | 历史分红记录 | 同上 |

## 硬性约束（别碰）⚠️

1. **绝对禁止输出**：涨跌预测、买卖建议、目标价、评级、「XX 会涨」「XX 值得买」这类话
2. **mock 数据里禁止出现**：预测字段、「目标价」「评级」「买入」等字样
3. **每个回答必须附带风险声明**：
   > 数据来自公开接口，仅供参考，不构成任何投资建议。股市有风险，投资需谨慎。
4. 用户如果问：
   - 「这只股票会涨吗？」
   - 「XX 股能不能买？」
   - 「给我推荐几只股票」
   
   → **必须拒绝**，回复固定模板：
   > 抱歉，我只能提供公开行情数据，不做投资预测。以下是当前公开信息：...（然后输出数据工具的返回结果）
5. 行情数据标注「延迟 15 分钟」或「实时」，并带来源标注
6. 不访问 Level-2 / 逐笔成交等付费数据接口

## Mock 测试

```bash
MOCK_MODE=true uv run pytest tests/skills/test_stock_info.py -q
# mock 数据固定为贵州茅台 (600519.SH) 的 60 根日 K 线
# 禁止在 fixtures 里出现任何「预测/目标价/买入」相关字段
# 风控测试用例：问「茅台会涨吗」→ 必须返回拒绝模板
```
