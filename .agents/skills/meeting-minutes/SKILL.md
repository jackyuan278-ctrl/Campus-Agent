---
name: meeting-minutes
description: 用户要整理会议纪要、抽待办、分配负责人时触发。触发词：会议纪要、待办、action item、minutes。排除：日程管理（用 calendar skill）、任务管理（用 task skill）、代码改动。
---

# meeting-minutes

把会议录音转写或原始文字整理成结构化纪要（议题 / 结论 / action items），并可对接日历分配。

## 快速用法

```python
# 从原始文本整理成纪要
minutes = await structure_minutes(raw_text="...")

# 只抽待办清单
actions = await extract_action_items(minutes_text="...")

# 把待办同步到日历（可选）
# await assign_to_calendar(actions, calendar_id="xxx")   # 需要授权
```

## 工具清单

| tool_name | 功能 | 数据源 |
|---|---|---|
| `structure_minutes` | 原始文字 → 议题/讨论/结论/待办 四段 | LLM + 会议模板库 |
| `extract_action_items` | 从任意文本里抽 action item + 负责人 + DDL | LLM + 正则（`@name` / `5/20` / `下周一`） |
| `summarize_topic` | 把单一议题的讨论过程压缩成 3 行 | LLM |
| `generate_followup` | 根据待办清单生成下次会议议程草案 | 内部模板 |

## 硬性约束（别碰）

1. **不修改待办的负责人或 DDL**——只抽取，分配由用户确认
2. 会议里涉及未同意的敏感内容（薪资、个人评价）不进纪要
3. 转写内容是「智能整理」，声明「可能遗漏细节，建议人工复核」
4. 不同参会人姓名/称谓统一归一（「Tom」「老大」「汤哥」→ 用户选择显示名）

## Mock 测试

```bash
MOCK_MODE=true uv run python -c "
from app.skills.meeting_minutes import structure_minutes
print(structure_minutes('今天讨论上线时间，Tom 说下周五。'))
"
# mock 返回四段结构化 Markdown，action item 自动识别
```
