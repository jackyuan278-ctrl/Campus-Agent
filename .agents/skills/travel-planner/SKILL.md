---
name: travel-planner
description: 用户要查天气、规划行程、订机票酒店时触发。触发词：旅游、行程、攻略、天气、航班、酒店、旅行。排除：日常代码改动、索引重建。
---

# travel-planner

给用户提供天气查询、行程规划、航班酒店信息检索的 skill。聚合多个公开 API，优先展示用户当前所在城市的本地信息。

## 快速用法

```python
# 直接调用 MCP 工具（mock 模式也能跑通）
result = await get_weather(city="北京", date="2026-11-23")
result = await search_flights(from_city="上海", to_city="成都", date="2026-12-01")
result = await plan_itinerary(destination="杭州", days=3, budget=1500)
```

## 工具清单

| tool_name | 功能 | 数据源 |
|---|---|---|
| `get_weather` | 查询城市当前天气与未来 7 天预报 | 和风天气 API / `mock_weather.py` |
| `search_flights` | 按出发地+目的地+日期搜航班列表 | 民航开放接口 / `mock_flights.py` |
| `search_hotels` | 按城市+日期+价格区间搜酒店 | 携程开放接口 / `mock_hotels.py` |
| `plan_itinerary` | 生成 N 天行程草案（景点 + 交通 + 预算） | 内部规则引擎 |
| `convert_currency` | 汇率换算 | 央行公开汇率 |

## 硬性约束（别碰）

1. **绝对不能替用户下单或支付**——只展示搜索结果和参考链接
2. 航班/酒店价格每 30 秒刷新一次，标注「仅供参考」
3. 天气只报公开预报，不报极端预警以外的个人建议
4. 行程生成不得包含未授权的敏感个人信息（身份证号、护照号）

## Mock 测试

```bash
# 所有外部 API 在 MOCK_MODE=true 下走本地 fixtures
MOCK_MODE=true uv run python -c "
from app.skills.travel_planner import get_weather
import asyncio
print(asyncio.run(get_weather('北京')))
"
# fixtures 位于 tests/mock/travel_planner/
```
