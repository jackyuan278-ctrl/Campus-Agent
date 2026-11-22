---
name: rag-indexing
description: 重建向量索引、调 chunker、改 embedding、修 Milvus、处理指纹问题时触发。触发词：重建索引、ingest、embedding、chunker、Milvus、指纹。排除：日常代码改动、部署。
---

# rag-indexing

两套独立向量索引，别混为一谈。

## 两套索引对比

| | 校园库 | 附件库 |
|---|---|---|
| 存储 | `data/rag/vectors.npy` + `docs.json` | `data/rag/milvus.db`（Milvus Lite） |
| 规模 | 约 570 条，1024 维 | 按上传增长，一份文档几十块 |
| 生命周期 | 只读，全量重建 | 按文件增删 |
| 隔离方式 | 全校共用 | `file_url` 标量过滤 |
| 检索 | 稠密 + BM25 → RRF → 规则重排 | 单路向量 top4 |

## 重建校园索引

```bash
uv run python -m app.rag.ingest --limit 0           # 全量；指纹命中 → 0 次 embedding
uv run python -m app.rag.ingest --limit 50          # 只处理前 50 条（调试）
uv run python -m app.rag.ingest --limit 0 --full    # 忽略指纹，强制重算（计费）
```

索引统计（只读）：

```bash
uv run python .agents/skills/campus-agent/scripts/index_stats.py
```

## 指纹机制

- 指纹 = `sha1(text)[:16]`
- `--full` 会跳过指纹缓存，所有 chunk 重新算 embedding（**计费**，谨慎）
- 时间、页码这类元数据放结构化字段，**不要拼进 `text`**，否则会触发全量重算

## Milvus Lite 锁警告

- Milvus Lite 是单进程独占的，目录里会有 `LOCK` 文件
- 启动服务期间**不要同时跑 ingest 或 doc_ingest**，会锁死 `milvus.db`
- 生产环境注意：多 worker（uvicorn `--workers N`）会互相抢锁，建议单 worker

## doc_store.get_all_chunks 用法（排查附件）

```python
from app.rag.doc_store import doc_store

chunks = doc_store.get_all_chunks(file_url="https://oss.xxx.com/foo.pdf")
# 返回该文件的所有切块，字段包含 text / page / block_type
```

## 硬性约束

1. 不要手改 `data/rag/vectors.npy` 或 `docs.json` — 行号对齐，错位不报错只答错
2. 不要手改 `data/rag/milvus.db` — 二进制格式，走上传/删除 API
3. 改 `chunker.py` 或 `embedding` 模型后**必须跑 `--full`**，旧指纹会全部失效
4. 改 `text` 拼接逻辑 = 全量重算向量，预计调用次数 × 每次 0.01 元 embedding 费用，先确认预算
