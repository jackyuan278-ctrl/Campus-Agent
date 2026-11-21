# 架构与接口

## 运行时

```
uvicorn app.main:app  →  HTTP 8000
                       ├── LangGraph 状态图（编译一次，全程复用）
                       ├── 校园索引（numpy，进程内）
                       └── 附件索引（Milvus Lite，进程内文件）
```

启动顺序（`app/main.py` 的 lifespan）：

1. 校验必填配置 —— 缺任何一个直接抛错退出（这是「启动即退出」最常见原因）
2. `mkdir data/rag/milvus.db` 的父目录
3. 建异步引擎、`AIOMySQLSaver.setup()` 建 `checkpoint*` 表
4. `build_graph(checkpointer)` 编译状态图

**全程不碰校园索引** —— 它是懒加载，第一次检索时才读文件，之后按 mtime 判断是否重载。

## 接口

| 方法 | 路径 | 关键参数 |
|---|---|---|
| POST | `/assistant/chat` | `message`（≤4000 字）、`thread_id`（≤64）、`mode`、`attachments`（≤5 个）；`?stream=false` 返回 JSON，默认 SSE |
| GET | `/assistant/chat/messages` | `thread_id`，返回可展示历史 |
| DELETE | `/assistant/chat/messages` | `thread_id`，清空历史（**不清** OSS 文件与 Milvus 块） |
| POST | `/assistant/upload` | multipart `file`，≤10MB，图片 / PDF / Word / txt |
| GET | `/health` | 不查依赖，只证明进程活着 |

鉴权：`Authorization: Bearer <JWT>`，`JWT_SECRET` 必须与 CampusHub 一致。`sub` 是 userId；未登录时配额按 `anon:<threadId>` 计。

`mode` 的取值：`chat` / `market` / `activity` / `doc` / `casual`。它决定工具集和 system prompt；`intent` 只是日志与返回值，**不参与能力决策**（用 intent 裁工具会把「最便宜的是哪个」判成 other 导致工具集为空）。

## 状态字段（`app/agent/state.py`）

| 字段 | 含义 |
|---|---|
| `messages` | 多轮对话，`add_messages` reducer，节点返回自动追加 |
| `query` | 本轮用户原话（**不含**附件正文） |
| `thread_id` / `user_key` | 会话 ID / 配额标识 |
| `mode` / `intent` | 前端档位 / 规则推断的意图 |
| `tool_rounds` / `turn_tokens` | 已跑工具轮数（上限 3）/ 本轮累计 token（上限 30k） |
| `attachments` / `doc_urls` | 本轮附件元信息 / 本会话传过的文档地址（供追问复检） |
| `attachment_text` / `vision_tokens` | 图片转文字的结果 / 识图耗时 token |
| `answer` | 本轮最终回复 |

## 数据表（库 `campushub_agent`）

| 表 | 用途 | 写入方 |
|---|---|---|
| `tb_chat_message` | 可展示的历史消息 | `persist` |
| `tb_thread_owner` | 会话归属（防越权读别人的会话） | `ensure_owner` |
| `tb_token_usage` | 按天累计 token，算配额 | `add_usage` |
| `tb_upload_usage` | 按天累计上传次数 | 上传接口 |
| `checkpoint*` | 图状态、写入记录 | `AIOMySQLSaver.setup()` 自动建 |

注意：**`checkpoint*` 是代码自动建的，`tb_*` 四张不是** —— 仓库里没有它们的建表 SQL，只存在于服务器。换环境会「服务能起、第一次对话 500」。

## 业务库（只读）

| 库 | 表 | 用途 |
|---|---|---|
| `campushub_activity` | `tb_activity` | 校园索引的活动来源（`status IN (1,2)`） |
| `campushub_community` | `tb_question` | 校园索引的问答来源 |
| `campushub_secondhand` | `tb_item` | 二手工具直接 SQL 查询 |

## 校园索引文件

```
data/rag/vectors.npy   float32，N×1024，已 L2 归一化 → 点积即余弦
data/rag/docs.json     N 条元数据，与矩阵逐行对应（第 i 行 ↔ 第 i 条）
```

`docs.json` 单条字段：`source`（activity/question）、`id`、`title`、`content`、`extra`、`text`（**唯一参与向量化的字段**）、`fingerprint`（`sha1(text)[:16]`）、`start_time` / `end_time` / `signup_end_time`（活动专有，不参与向量化）。

索引没有 `built_at` / `schema_version`，**新代码跑旧索引不会报错**，只会静默少功能（例如没有 `start_time` 时时间过滤等于空转）。

## 关键常量（`app/common/constants.py`）

| 常量 | 值 | 作用 |
|---|---|---|
| `CHAT_HISTORY_LIMIT` | 20 | 进 prompt 的历史条数 |
| `CHAT_SUMMARY_TRIGGER` / `KEEP` | 40 / 20 | 超 40 条摘要压缩到 20 条 |
| `MAX_TOOL_ROUNDS` | 3 | 模型↔工具最大轮数 |
| `MAX_TURN_TOKENS` | 30_000 | 单轮 token 上限 |
| `MAX_IMAGES` | 5 | 单轮最多解析几张图 |
| `DOC_CHUNK_SIZE` / `OVERLAP` | 400 / 80 | 切块大小与重叠（配额倒推，非随手值） |
| `DOC_TOP_K` | 4 | 附件检索带几块进 prompt |
| `PAGE_TEXT_RICH` / `LOW` | 200 / 50 | 每页字数分流阈值 |
| `MAX_IMAGE_PAGES` | 10 | 单文档最多做视觉补全的页数 |
| `TABLE_NUMERIC_RATIO` | 0.4 | 单页数字行占比 → 判定表格页 |
| `TABLE_PAGE_RATIO` | 0.5 | 表格页过半 → 整份改走 MinerU |

检索侧（`app/rag/retriever.py`）：每路召回 `_RECALL_K = 30`，RRF `k = 60`，重排权重 `0.7 × RRF + 0.3 × 启发式`（标题命中 0.6 / 新鲜度 0.4），`min_score` 默认 0（未定阈值，别拿它当过滤开关）。

## 配置分组（`.env`）

| 组 | 键 |
|---|---|
| MySQL | `MYSQL_HOST/PORT/USER/PASSWORD`、`MYSQL_DB_ACTIVITY`、`MYSQL_DB_COMMUNITY`、`MYSQL_DB_SECONDHAND`、`MYSQL_DB_AGENT` |
| 对话模型 | `GLM_BASE_URL`、`GLM_API_KEY`、`GLM_CHAT_MODEL`、`GLM_VISION_MODEL` |
| 向量 | `TENCENT_TOKENHUB_API_KEY`、`TENCENT_TOKENHUB_MODEL_NAME`、`TENCENT_TOKENHUB_BASE_URL`、`MILVUS_LITE_PATH` |
| 检索/工具 | `TAVILY_API_KEY`、`MINERU_API_KEY`、`MINERU_BASE_URL`、`MINERU_TIMEOUT` |
| 存储 | `OSS_ACCESS_KEY_ID`、`OSS_ACCESS_KEY_SECRET`、`OSS_BUCKET`（必须公共读，否则 MinerU 403）、`OSS_ENDPOINT`、`OSS_REGION` |
| 配额与鉴权 | `JWT_SECRET`、`DAILY_TOKEN_QUOTA`、`DAILY_UPLOAD_LIMIT` |
| 其他 | `RABBITMQ_URL`、`RABBITMQ_INDEX_QUEUE`、`CORS_ALLOW_ORIGINS` |
