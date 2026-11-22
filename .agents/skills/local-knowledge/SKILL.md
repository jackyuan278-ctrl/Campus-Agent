---
name: local-knowledge
description: 用户要查校园知识库、FAQ、政策通知时触发。触发词：图书馆、教务处、奖学金、补考、选课、校园 FAQ。排除：联网搜索。
---

# local-knowledge

基于本地校园知识库（RAG 向量索引 + FAQ 精确匹配）回答校园政策、办事流程、常见问题的 skill。**不联网搜索**，所有答案来自校内已备案的文档。

## 快速用法

```python
# 走 RAG 检索（召回 + 重排）
answer = await query_campus_knowledge(question="怎么申请奖学金？", top_k=4)

# 精确 FAQ 匹配（0.1s 响应）
faq = await faq_lookup(keyword="补考")

# 列出当前开放的办事事项
services = await list_services(category="教务")
```

## 工具清单

| tool_name | 功能 | 数据源 |
|---|---|---|
| `query_campus_knowledge` | 自然语言问答（RAG） | 本地 `data/rag/vectors.npy` + `docs.json` |
| `faq_lookup` | 关键词精确匹配 FAQ | 本地 `data/rag/faq.json` |
| `list_services` | 按类别列出办事事项 + 办理入口链接 | 同上 |
| `get_service_detail` | 某事项的办理材料/流程/时限详情 | 同上 |

## 硬性约束（别碰）

1. **绝不联网搜索**——超出本地知识库范围的问题必须回复「抱歉，这个问题暂时不在我的校园知识范围内」
2. 每个回答附带「来源文档名 + 页码」，便于用户回溯
3. 知识更新频率标注（如「数据截止 2026-10-15」）
4. 涉及具体金额/时间的事项，建议用户再去官网或教务处柜台确认
5. 不要手改 `data/rag/*` 文件（索引重建由 rag-indexing skill 负责）

## Mock 测试

```bash
MOCK_MODE=true uv run pytest tests/skills/test_local_knowledge.py -q
# mock 索引是 10 条预置 FAQ，不走 Milvus / embedding
# faq_lookup("补考") → 固定返回教务处补考流程
```
