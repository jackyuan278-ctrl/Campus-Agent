---
name: product-comparison
description: 用户要比价、看历史价格、找优惠促销时触发。触发词：比价、多少钱、优惠、促销、买哪个好、评测。排除：其他。
---

# product-comparison

跨电商平台抓取商品价格、历史走势、促销活动的 skill。帮用户「等降还是现在买」的决策。

## 快速用法

```python
# 多平台比价
result = await compare_prices(keyword="Kindle Paperwhite 2024", platforms=["京东","天猫","拼多多"])

# 历史价格曲线
history = await price_history(product_id="jd_100012043978", days=90)

# 当前有没有促销
promos = await scan_promotions(category="电子书阅读器")
```

## 工具清单

| tool_name | 功能 | 数据源 |
|---|---|---|
| `compare_prices` | 同品跨平台当前价 + 到货日对比 | 京东/天猫/拼多多开放接口 |
| `price_history` | 某商品近 N 天价格序列 | 内部价格追踪库 |
| `scan_promotions` | 某品类下当前进行中的促销 | 同上 |
| `product_reviews_summary` | 评论摘要（好评点/吐槽点 TOP3） | 电商评论聚合 |
| `alert_threshold` | 设置降价提醒阈值（仅记录，不主动推送） | 内部存储 |

## 硬性约束（别碰）

1. **价格每 10 分钟刷新一次**，回答里必须带上「数据截止时间」
2. 历史价格只供参考，标注「不构成投资/消费建议」
3. 不自动下单，只返回商品链接
4. 评论摘要必须声明「基于公开评论的 AI 归纳」

## Mock 测试

```bash
MOCK_MODE=true uv run python -c "
from app.skills.product_comparison import compare_prices
print(compare_prices('Kindle Paperwhite 2024'))
"
# mock 返回三平台统一价 + 各平台到货日
# fixtures: tests/mock/product_comparison/
```
