---
name: english-helper
description: 用户要学生词、英文精读、语法练习时触发。触发词：英语、单词、精读、语法、发音、vocabulary。排除：翻译、简历、代码、学术论文、旅游规划等非英语学习类请求。
---

# english-helper

英语学习辅助 skill：单词卡片生成、文章精读拆解、语法点练习、发音标注（IPA）。

## 快速用法

```python
# 查词 + 生成记忆卡片
card = await word_card(word="serendipity", level="CET6")

# 精读拆解（语法 + 难词 + 逻辑结构）
analysis = await analyze_reading(text="...")

# 出一道语法题
quiz = await grammar_quiz(topic="虚拟语气", level="中级")
```

## 工具清单

| tool_name | 功能 | 数据源 |
|---|---|---|
| `word_card` | 音标 + 词性 + 例句 + 记忆技巧 | Oxford Dict + 内部词频 |
| `analyze_reading` | 难词标注 + 从句拆解 + 逻辑结构树 | LLM + 正则解析 |
| `grammar_quiz` | 按主题/难度生成选择题 | 内部题库 |
| `ipa_transcribe` | 给任意英文文本生成 IPA 音标 | CMU Pronouncing Dict |
| `level_assess` | 10 道题估算 CEFR 等级（A1~C2） | 内部题库 |

## 硬性约束（别碰）

1. **不替代真人外教**——明确声明「AI 辅助学习，发音/写作建议仅供参考」
2. 例句优先用真实英文原著/新闻，不用瞎编的句子
3. 音标用 DJ（Daniel Jones）体系，标注版本
4. `grammar_quiz` 答案必须带解析，不能只给 A/B/C

## Mock 测试

```bash
MOCK_MODE=true uv run python -c "
from app.skills.english_helper import word_card
print(word_card('serendipity'))
"
# mock 返回预置的 CET6 卡片，例句来自 TED
```
