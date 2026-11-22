# RAG 索引重建全流程

> 本文件说明 Campus-Agent 的两套向量索引（校园库 / 附件库）各自的构建机制、命令参数、指纹去重和增量更新策略。

---

## 1. 两套索引的差异速查

| 维度 | 校园库 | 附件库 |
|------|--------|--------|
| **物理存储** | `data/rag/vectors.npy` + `docs.json` | `data/rag/milvus.db`（Milvus Lite） |
| **构建入口** | `python -m app.rag.ingest`（离线 CLI） | 附件上传后自动触发（`doc_ingest`） |
| **数据来源** | MySQL 业务库（tb_activity + tb_question） | OSS 上用户上传的 PDF 文件 |
| **粒度** | 一条活动 / 一条问答 = 一个文档 | 一个 PDF → 切块（约 512 token/块） → 每块一条向量 |
| **隔离** | 全校共用，无过滤 | 按 `file_url` 隔离，检索时限定单附件范围 |
| **更新方式** | 手动全量重建 / RabbitMQ 增量单条 | 随上传实时入库，删除附件时按 file_url 清理 |
| **指纹机制** | `sha1(text)[:16]` 去重 + 复用旧向量 | 无 |
| **写入锁** | `write_lock.py` 异步互斥 | Milvus Lite 自带文件锁 |

---

## 2. 校园库重建命令

### 基本用法

```bash
# 默认：每张表取最近 200 条（快速验证用）
uv run python -m app.rag.ingest

# 每张表取最近 500 条
uv run python -m app.rag.ingest --limit 500

# 全量：每张表全部记录（生产部署必须用这个）
uv run python -m app.rag.ingest --limit 0

# 全量 + 强制重算所有向量（索引疑似损坏 / embedding 模型换了时用）
uv run python -m app.rag.ingest --limit 0 --full
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--limit N` | `200` | 每张表取多少条。`0` 表示全量。负数无效。 |
| `--full` | `false` | 不复用旧向量，全部重新 embedding。默认 `false`（增量模式）。 |

### 为什么有全量和增量两种模式？

`campus-hub` 的造数脚本（`database/13_seed_community.sql`）用 `ELT()` 拼标题正文，1000 条问答实际只有 ~34 种不同的文本。全量入库会让近似重复的向量占满检索候选池（top_k×3 = 15 条候选，去重后只剩 1 条有效）。

增量模式下，指纹命中的内容直接复用旧向量，**新增 / 变更的文档才花 embedding 调用**。

---

## 3. 指纹机制详解

### 计算方式

```python
import hashlib

def fingerprint(text: str) -> str:
    """取 sha1 前 16 位。2^64 空间，5000 条文档撞车概率可忽略。"""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
```

### 用途一：内容去重（同批次内）

```
原始数据集 = [doc_A, doc_B, doc_A_copy, doc_C]   # doc_A_copy 与 doc_A 文本完全相同
                     ↓
fingerprint 去重   → [doc_A, doc_B, doc_C]       # doc_A_copy 被丢弃
```

### 用途二：跨批次复用旧向量（增量模式）

```
旧索引（上次 ingest 生成）:
  row 0: fingerprint=a1b2, vector=[0.12, -0.34, ...]
  row 1: fingerprint=c3d4, vector=[0.56,  0.78, ...]

新数据集（本次取）:
  doc_X: fingerprint=a1b2  ← 命中！直接复用旧索引 row 0 的向量
  doc_Y: fingerprint=e5f6  ← 未命中，需要调 embedding API
```

关键代码路径：`ingest.py` → `build_index()` → 先按指纹建旧向量映射表 → 遍历新文档时查表复用。

### 指纹的 text 拼接规则

活动文档和问答文档各有独立的 text 拼接函数，**且只有一份实现**：

```python
# app.rag.ingest — activity_doc
"活动：{title}。地点：{location}。{description[:500]}"

# app.rag.ingest — question_doc  
"问答：{title}。{content[:500]}"
```

时间字段（`start_time` / `end_time` / `signup_end_time`）**不拼进 text**——它们是结构化过滤和排序用的，拼进去会让 fingerprint 频繁变化，导致每次 ingest 都重算所有活动向量。

### 为什么不用完整 sha1？

sha1 全 40 位同样可用，但前 16 位（64 bit）对 5000 条文档来说空间远足够。完整 sha1 的好处是理论上零撞车，代价是 docs.json 里每个 fingerprint 字段多 24 字节——对 5000 条来说可忽略，但 16 位更紧凑也够用。

---

## 4. Milvus Lite 单进程独占警告

### 这是硬约束，别碰

Milvus Lite 的底层（faiss-cpu）使用文件锁，**同一时刻只能被一个进程打开**。

**禁止场景**：
- 服务正在运行时（Milvus Lite 已被 `/assistant/upload` 占用），同时跑 `python -m app.rag.ingest`
- 两个容器实例同时挂载同一个 `milvus.db` 卷

**后果**：进程会卡在 open 调用上超时，不会报错、不会终止——看起来像服务"僵住了"。

### 正确做法

```bash
# 先停服务再重建索引
docker compose down
docker compose run --rm agent python -m app.rag.ingest --limit 0
docker compose up -d
```

或者**并行使用两个 data 目录**：
```bash
# 开发时本地 rebuild 不影响线上数据
cp -r data data.backup.20261122
uv run python -m app.rag.ingest --limit 0
# 确认新索引 OK 后
rm -rf data.backup.20261122
```

### 写锁机制

代码层面，校园库的写入路径（消费者 + 手动 ingest）都经过 `app/rag/write_lock.py` 的 `index_write_lock()` 异步互斥锁，防止两个写入方同时 rebuild。但 Milvus Lite 不经过这个锁——它的锁是 faiss 的文件级锁，进程外部行为。

---

## 5. 新增文档入库流程

### 方式一：手动全量重建（推荐，最安全）

```bash
uv run python -m app.rag.ingest --limit 0
```

流程：
```
取 MySQL 全量数据 → fingerprint 去重 → 旧向量复用 → 新增 embedding
    → 构建 VectorIndex → index.save() 覆写 vectors.npy + docs.json
```

注意：**重建过程中检索不可用**（旧文件在覆写完成前不会被切走）。重建完成后下一次请求才会加载新索引。

### 方式二：RabbitMQ 增量更新（生产持续集成方案）

```
campus-hub 增删改 → 发消息到 campus.index.rebuild 队列
                        ↓
              app.rag.consumer 收到消息
                        ↓
              app.rag.indexer 按 id 单条更新
                        ↓
              load_activity(id) / load_question(id)
                        ↓
              单条 embedding → 重写索引（复用 write_lock）
```

未配置 `RABBITMQ_URL` 时消费者不启动，索引只能靠手动 ingest 更新。

### 方式三：附件实时入库

用户通过 `/assistant/upload` 上传 PDF 后：

```
文件落 OSS → 解析文本（pypdfium2 直抽 / MinerU 回退）
    → app.rag.doc_ingest 切块（约 512 token/块）
    → 每块 embedding → 写入 Milvus Lite（按 file_url 隔离）
```

此时**校园库不受影响**——附件库是独立的 Milvus Lite，存储位置相同但逻辑完全隔离。

---

## 6. 索引文件损坏排查

| 症状 | 可能原因 | 排查方法 | 修复 |
|------|---------|---------|------|
| 查询返回空结果 | docs.json 和 vectors.npy 行号错位 | 对比文件行数是否一致 | `--full` 强制重建 |
| 检索结果语义相关度低 | embedding 模型换了但没全量重算 | 看 docs.json 的 fingerprint 是否都是旧模型算的 | `--full` 强制重建 |
| 启动时报 `ImportError: libgomp.so.1` | faiss-cpu 缺 OpenMP 运行时 | 在容器内 `apt-get install libgomp1` | Dockerfile 已处理 |
| Milvus Lite 打开卡死 | 有其他进程占用 milvus.db | `lsof milvus.db` 看被谁锁了 | kill 掉占用进程 |
| Fingerprint 复用失败，全部重算 | `--full` 模式开了，或旧索引无 fingerprint 字段 | 检查是否是你手动删过 docs.json | 无影响，就是多花点 embedding 调用 |

### 快速检查脚本

仓库里有 `scripts/index_stats.py`（实际位置见 `.agents/skills/campus-agent/scripts/index_stats.py`），可以打印索引统计：

```bash
uv run python .agents/skills/campus-agent/scripts/index_stats.py
# 输出类似：Index: 342 docs, dim=1024, 231 activities + 111 questions
```
