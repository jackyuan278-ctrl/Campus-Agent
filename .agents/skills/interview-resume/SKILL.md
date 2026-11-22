---
name: interview-resume
description: 用户要写简历、准备面试、薪资谈判时触发。触发词：简历、面试、求职、薪资、offer、内推。排除：业务代码改动。
---

# interview-resume

辅助用户打磨简历、模拟面试、估算薪资区间的 skill。模板 + 行业数据 + 对话式追问。

## 快速用法

```python
# 简历 ATS 打分
score = await resume_ats_score(resume_text="...", target_job="后端开发")

# 生成 STAR 面试题
questions = await generate_interview_questions(role="产品经理", difficulty="中", count=5)

# 薪资区间估算（基于公开数据）
salary = await estimate_salary(city="上海", role="后端开发", years=3)
```

## 工具清单

| tool_name | 功能 | 数据源 |
|---|---|---|
| `resume_ats_score` | ATS 关键词匹配打分 + 缺失项提示 | 内部关键词库 |
| `generate_interview_questions` | 按岗位/难度生成行为/技术面试题 | 内部题库 + LLM 生成 |
| `mock_interview_answer` | 用户回答后给出追问建议 | LLM 对话链 |
| `estimate_salary` | 城市×岗位×年限的薪资分位区间 | BOSS 直聘 / 猎聘公开聚合 |
| `cover_letter_draft` | 按 JD 和简历生成求职信初稿 | LLM |

## 硬性约束（别碰）

1. **薪资数据标注来源和更新日期**，不要给「保证能拿到 X」的承诺
2. 面试题生成不得包含用户隐私（在读学校、公司等）
3. 不做「简历必过」「包 offer」这类营销性表述
4. `estimate_salary` 只返回 P25/P50/P75 三档，不返回单一数值

## Mock 测试

```bash
MOCK_MODE=true uv run pytest tests/skills/test_interview_resume.py -q
# fixtures: tests/mock/interview_resume/
# mock salary 固定返回 {"P25": 20, "P50": 28, "P75": 38, "source": "BOSS直聘 2026Q3"}
```
