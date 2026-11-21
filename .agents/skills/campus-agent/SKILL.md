---
name: campus-agent
description: Campus-Agent（FastAPI + LangGraph 的校园问答助手，单体服务）的开发手册：架构地图、启动与索引命令、检索与文档链路、硬性约束、症状到位置的排错表。当需要修改这个项目的代码、本地起服务、重建校园索引或附件向量库、部署到服务器、或排查「答不出/答错/接口 500/上传失败」这类问题时使用。
---

# Campus-Agent

一个 FastAPI 进程同时承担 HTTP 接口、LangGraph 编排和两套向量索引。生产跑在腾讯云 ECS，容器名 `campus-agent`，端口 `8000`，由 CampusHub 网关把 `/api/assistant/**` 转发进来（StripPrefix 后交给本服务）。

## 60 秒上手

```bash
uv sync                                   # 按 uv.lock 安装；Dockerfile 用 --frozen，lock 必须一起提交
cp .env.example .env                      # 必填：GLM_API_KEY、TENCENT_TOKENHUB_API_KEY、MYSQL_*
uv run uvicorn app.main:app --reload --port 8000
curl 127.0.0.1:8000/health                # {"status":"ok"}
```

索引与测试：

```bash
uv run python -m app.rag.ingest --limit 0          # 全量重建校园索引；指纹命中则 0 次 embedding
uv run python -m app.rag.ingest --limit 0 --full   # 忽略指纹强制重算向量（计费）
uv run pytest -q
uv run python .agents/skills/campus-agent/scripts/index_stats.py   # 只读：看索引规模与新鲜度
```

## 代码地图

| 路径 | 职责 |
|---|---|
| `app/main.py` | 装配、lifespan（校验配置 → 建 Milvus 目录 → 连 MySQL + `saver.setup()` → 编译图）、`/health` |
| `app/api/v1/chat.py` | `POST /assistant/chat`（SSE，`?stream=false` 返回 JSON）、`GET/DELETE /assistant/chat/messages` |
| `app/api/v1/upload.py` | `POST /assistant/upload`：落 OSS → 解析 → 切块 → 入 Milvus |
| `app/agent/graph.py` | 7 节点状态图、档位 prompt、工具装配、注入检测、当前时间注入 |
| `app/agent/runtime.py` | `AIOMySQLSaver` 生命周期、5s 探活缓存、断连重建 |
| `app/agent/locks.py` | `thread_id` 级锁：同会话串行，防 last-write-wins 覆盖历史 |
| `app/agent/tools/` | `secondhand` 二手查询、`activity` 活动/社区检索、`web` 联网（仅闲聊档挂） |
| `app/rag/ingest.py` | 校园库全量构建（活动 + 问答） |
| `app/rag/index.py` | numpy 索引的加载/原子保存/行数校验 |
| `app/rag/retriever.py` | 稠密 + BM25 双路召回 → RRF → 规则重排 |
| `app/rag/chunker.py` | 切块：父标题注入、页码、块类型 |
| `app/rag/doc_ingest.py` | 附件分流判定 + HTML/pipe 表格归一化 + 切块入库 |
| `app/rag/doc_store.py` | Milvus Lite 读写，按 `file_url` 过滤 |
| `app/llm/pdf_text.py` | 四路分流：text / mixed / table / scan |
| `app/memory/` | `store` 历史表、`quota` 日配额、`ownership` 会话归属 |
| `app/common/clock.py` | 北京时间单一真源（容器是 UTC，别直接 `datetime.now()`） |

## 一次对话怎么走

```
summarize → image_agent → route → agent ⇄ tools → strip_calls → persist
```

1. **summarize**：历史超 40 条 → 摘要压到 20 条
2. **image_agent**：图片转文字（`glm-4.6v`）、文档块检索；附件正文**只当轮注入**，不进 checkpointer
3. **route**：按档位装配工具集（market / activity / doc / chat），闲聊档才挂联网搜索
4. **agent**：调模型，system prompt = 档位 prompt + 当前时间
5. **tools**：`ToolNode`，异常降级成可读话术，最多 3 轮
6. **strip_calls**：超轮次去掉 `tool_calls`，逼模型收尾
7. **persist**：写 `tb_chat_message`、累计 token 与配额

## 两套向量索引（别混为一谈）

| | 校园库 | 附件库 |
|---|---|---|
| 存储 | `data/rag/vectors.npy` + `docs.json` | `data/rag/milvus.db`（Milvus Lite） |
| 规模 | 约 570 条，1024 维 | 按上传增长，一份文档几十块 |
| 生命周期 | 只读，全量重建 | 按文件增删 |
| 隔离方式 | 无（全校共用） | `file_url` 标量过滤 |
| 检索 | 矩阵乘 + BM25 + RRF | 单路向量 top4 |

## 硬性约束

1. **不要擅自启动/重启服务或容器**，给命令让本人执行。
2. **不要用真实付费 key 做验证**：embedding、chat、视觉都计费。要验证先问，并说明预计调用次数。
3. **不要手改 `data/rag/*`**：`vectors.npy` 与 `docs.json` 靠行号对齐，错位不会报错只会答错。改数据走 `ingest`。
4. **改 `text` = 全量重算向量**：指纹是 `sha1(text)[:16]`。时间、页码这类元数据放结构化字段，别拼进 `text`。
5. **单轮上限 30k token**、日配额 `DAILY_TOKEN_QUOTA`、上传次数 `DAILY_UPLOAD_LIMIT`。
6. **Milvus Lite 单进程独占**（目录内有 `LOCK`），不要开多 worker。

## 症状 → 先看哪里

| 症状 | 先看 |
|---|---|
| 启动即退出 | 缺必填 env（lifespan 第一段就校验）；看日志第一行 |
| 接口 401 | `JWT_SECRET` 与 CampusHub 不一致 |
| 问活动答「没找到」 | ① 索引是否有 `start_time`（旧索引要重跑 ingest）② 是否被「已结束/时间窗」过滤 ③ 日志 `检索命中 N 条` |
| 文档问答答错 | 日志 `route=... mode=doc 附件=N`；Milvus 里有没有块；切块是否被 HTML 标签污染 |
| 上传 500 | MinerU key/超时；OSS bucket 是否公共读（否则 MinerU 403） |
| 工具报错但接口 200 | 日志 `工具执行失败`：多为 SQL 字段或函数签名不匹配（`py_compile` 查不出参数个数） |
| 重启后「索引没了」 | 容器重建没挂 `./data:/app/data` |

更细的排查步骤见 `references/pitfalls.md`。

## 深入

- `references/architecture.md` — 状态字段、接口参数、表结构、配置分组、常量表
- `references/operations.md` — 本地/服务器命令、部署与回滚、重建索引、日志关键词
- `references/pitfalls.md` — 踩过的坑：现象 → 原因 → 修法
