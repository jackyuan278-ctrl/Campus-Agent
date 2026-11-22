---
name: weekly-report
description: 用户要写周报、做 OKR 复盘、生成汇报材料时触发。触发词：周报、日报、OKR、复盘、汇报、总结。排除：其他。
---

# weekly-report

从工作日志、Git 提交、日历日程里自动抽取，生成周报/月报/OKR 复盘初稿的 skill。

## 快速用法

```python
# 从 Git 提交 + 日志生成周报
report = await generate_weekly_report(
    git_repos=["~/proj/a", "~/proj/b"],
    days=7,
    author="tom"
)

# OKR 复盘草案
okr_review = await generate_okr_review(
    objectives=["上线智能客服", "重构 RAG 索引"],
    period="2026-Q4"
)
```

## 工具清单

| tool_name | 功能 | 数据源 |
|---|---|---|
| `generate_weekly_report` | 按仓库/作者/时间窗抽取 commit + 汇总 | git log + 内部日志 |
| `generate_okr_review` | 按 OKR 模板生成复盘初稿 | 内部模板库 |
| `summarize_meeting_notes` | 把会议纪要压缩成 3 条要点 | LLM |
| `extract_action_items` | 从非结构化文本里抽待办清单 | LLM + 正则 |
| `format_markdown` | 把结构化数据渲染成可复制的 Markdown | 内部模板 |

## 硬性约束（别碰）

1. **周报内容必须带来源标注**（commit hash / 文档链接），不凭空编造
2. 不访问未授权仓库——只扫当前用户有权限的本地路径
3. OKR 自评分数由用户自己打，AI 只给事实陈述
4. 汇报里涉及他人的工作必须经用户确认后再出现

## Mock 测试

```bash
MOCK_MODE=true uv run pytest tests/skills/test_weekly_report.py -q
# mock git log 返回预置的 12 条 commit
# 输出固定为标准周报 Markdown 模板
```
