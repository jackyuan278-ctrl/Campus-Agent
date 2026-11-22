---
name: fitness-diet
description: 用户要健身计划、饮食安排、打卡追踪时触发。触发词：健身、减肥、增肌、饮食、运动、卡路里。排除：临床诊断、处方建议、代码改动、学术调研。
---

# fitness-diet

基于用户身高体重目标生成健身计划 + 饮食建议的 skill。支持打卡追踪和进度回顾。

## 快速用法

```python
# 生成 4 周减脂计划
plan = await generate_workout_plan(goal="减脂", level="中级", days_per_week=4)

# 计算 TDEE / BMR
tdee = await calculate_tdee(height=175, weight=72, age=26, activity="中度活跃")

# 今日饮食记录 + 卡路里汇总
today = await log_meal(
    meal="午餐", items=[("米饭", 150, "g"), ("鸡胸肉", 200, "g"), ("西兰花", 100, "g")]
)
```

## 工具清单

| tool_name | 功能 | 数据源 |
|---|---|---|
| `generate_workout_plan` | 按目标/水平/频率生成训练计划 | 内部动作库 + LLM |
| `calculate_tdee` | Mifflin-St Jeor 公式算 BMR×活动系数 | 标准公式 |
| `log_meal` | 记录一顿饭 + 卡路里/营养素拆分 | 中国食物成分表 |
| `progress_summary` | 周/月减重/增肌趋势图 | 用户历史日志 |
| `safe_rep_range` | 某动作对某体重的推荐次数区间 | NSCA 指南 |

## 硬性约束（别碰）

1. **绝对不能给临床诊断或治疗建议**——涉及伤病立即回复「请咨询专业医生」
2. 饮食建议必须声明「AI 辅助参考，不替代注册营养师」
3. 所有数值标注「估算值」，实际因人而异
4. 不给极端节食（日摄入 < BMR×0.8）或过度训练的计划

## Mock 测试

```bash
MOCK_MODE=true uv run pytest tests/skills/test_fitness_diet.py -q
# mock 食物库固定返回中国食物成分表 2020 版
# generate_workout_plan 返回 4 周 16 练标准计划
```
