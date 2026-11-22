---
name: literature-research
description: 用户要查学术论文、看引用关系、文献调研时触发。触发词：论文、文献、引用、学术、research、paper。排除：日常代码改动。
---

# literature-research

面向学术场景的文献检索与引用分析 skill。支持从 Semantic Scholar / arXiv / DBLP 拉取元数据，绘制引用图谱。

## 快速用法

```python
papers = await search_papers(query="large language model reasoning", year_from=2023, limit=10)
refs   = await get_citation_graph(paper_id="arXiv:2305.18290", depth=2)
cite   = await format_citation(paper_id="...", style="apa")   # apa / ieee / gb7714
```

## 工具清单

| tool_name | 功能 | 数据源 |
|---|---|---|
| `search_papers` | 关键词/作者/年份检索论文 | Semantic Scholar / arXiv 公共 API |
| `get_paper_detail` | 论文元数据 + 摘要 + PDF 链接 | 同上 |
| `get_citation_graph` | 引用关系图（入/出引用，支持深度 1~3） | Semantic Scholar 图谱 |
| `format_citation` | 按 APA / IEEE / GB/T 7714 格式化参考文献 | 内部 bibtex 解析器 |
| `recommend_similar` | 基于 embedding 的相似论文推荐 | 内部向量索引 |

## 硬性约束（别碰）

1. 不提供论文全文下载——只给 arXiv / 出版社的官方跳转链接
2. 引用计数以数据源公布为准，不要自己算
3. `format_citation` 遇到缺失字段时必须标注 `[缺失: venue]`，不要瞎编

## Mock 测试

```bash
MOCK_MODE=true uv run pytest tests/skills/test_literature_research.py -q
# fixtures: tests/mock/literature_research/papers.json
# mock 模式下 citation_graph 固定返回 5 节点星形图
```
