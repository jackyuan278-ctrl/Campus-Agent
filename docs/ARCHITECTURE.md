# 架构说明

> 本文件描述 Campus-Agent 的整体架构、模块分工和关键设计决策。对应仓库根目录的 `AGENTS.md` 中「代码地图」和「架构要点」段落的扩展版。

---

## 1. 代码地图（完整目录树）

```
campus-agent/
├── app/                          # 主应用源码
│   ├── main.py                   # FastAPI app 工厂 + lifespan 生命周期钩子
│   ├── config.py                 # 单层 Settings（.env 读入，lru_cache 单例）
│   ├── database.py               # MySQL 异步连接管理（aiomysql，多库切换）
│   ├── api/
│   │   └── v1/
│   │       ├── chat.py           # POST /assistant/chat — SSE 私有流式接口
│   │       ├── responses.py      # POST /v1/responses — Responses API 兼容层
│   │       └── upload.py         # POST /assistant/upload — 附件上传
│   ├── agent/                    # LangGraph Agent 核心 ⭐
│   │   ├── graph.py              # 7 节点状态图定义 + System Prompts
│   │   ├── state.py              # AgentState TypedDict
│   │   ├── runtime.py            # AgentRuntime：图构建 + MySQL checkpoint
│   │   ├── locks.py              # 会话级互斥锁（thread_id → asyncio.Lock）
│   │   ├── skills/
│   │   │   └── __init__.py       # SkillRegistry 注册中心
│   │   └── tools/                # 工具包（每个包末尾都有 register()）
│   │       ├── __init__.py       # 汇总导入所有工具模块并统一注册
│   │       ├── secondhand.py     # 二手查询工具
│   │       ├── activity.py       # 校园活动检索工具
│   │       └── web.py            # 联网搜索（仅 casual 档可用）
│   ├── llm/                      # LLM 封装层
│   │   ├── gateway.py            # get_chat_model() / get_embedding_model() 工厂
│   │   ├── vision.py             # 图片视觉识别（glm-4.6v）
│   │   ├── pdf_text.py           # PDF 文本层直抽（pypdfium2，~0.01s）
│   │   └── mineru.py             # MinerU 云端回退（~60s，文本层不可用时才用）
│   ├── rag/                      # RAG 两套索引
│   │   ├── index.py              # 校园库：vectors.npy + docs.json
│   │   ├── bm25.py               # BM25 稀疏检索
│   │   ├── retriever.py          # 融合检索入口（稠密 + BM25 → RRF）
│   │   ├── fuse.py               # Reciprocal Rank Fusion 实现
│   │   ├── chunker.py            # 文档切块（附件用）
│   │   ├── embedding.py          # Embedding 模型封装
│   │   ├── ingest.py             # 校园库重建 CLI
│   │   ├── indexer.py            # 单条增量入库（供消费者调用）
│   │   ├── consumer.py           # RabbitMQ 增量索引消费者
│   │   ├── doc_store.py          # 附件库：Milvus Lite（milvus.db）
│   │   ├── doc_ingest.py         # 附件入库流程（切块 → embedding → 写入 Milvus）
│   │   └── write_lock.py         # 索引写入互斥（防止消费者与手动 ingest 并发）
│   ├── memory/                   # 会话持久化 + 配额管理
│   │   ├── store.py              # MemoryStore：tb_chat_message 读写
│   │   ├── ownership.py          # ensure_owner：thread_id → user_id 归属校验
│   │   └── quota.py              # get_quota：每日 token 配额 + 上传次数配额
│   ├── storage/
│   │   └── oss.py                # 阿里云 OSS 封装（附件上传）
│   ├── schemas/                  # Pydantic 请求/响应模型
│   │   ├── chat.py               # ChatRequest / Attachment（/assistant/chat 专用）
│   │   └── responses.py          # ResponsesRequest / ResponsesEvent（/v1/responses 专用）
│   └── common/
│       ├── logger.py             # 结构化日志
│       ├── clock.py              # 显式北京时间（不走系统时区）
│       ├── security.py           # JWT 解析（从请求头提取 user_id）
│       ├── constants.py          # 常量（档位名、默认值）
│       └── exceptions.py         # 自定义异常体系
├── tests/                        # pytest 测试套件
├── data/                         # 向量索引文件（.gitignore，部署时挂载卷）
│   ├── rag/
│   │   ├── vectors.npy           # 校园库向量（numpy 数组）
│   │   ├── docs.json             # 校园库文档元数据（与 vectors.npy 行号对齐）
│   │   └── milvus.db             # 附件库（Milvus Lite 文件）
├── Dockerfile                    # 容器镜像构建
├── docker-compose.yml            # 部署编排（agent 单服务）
├── pyproject.toml                # 项目元数据 + 依赖（uv 管理）
├── uv.lock                       # 锁定的依赖版本
└── .env.example                  # 环境变量模板
```

---

## 2. Agent Loop 7 节点详解

LangGraph 状态图定义在 `app/agent/graph.py`。整体拓扑：

```
START → route ──→ model_router ──→ agent ←──┐
                  ↓ (有附件时)              │ 工具循环（上限 5 轮）
              image_agent ─────────────→ agent ←──┘
                                             ↓ (strip_calls 清洗)
                                        summarize → END
```

各节点职责：

| 节点 | 入口条件 | 核心动作 | 退出产物 |
|------|---------|---------|---------|
| **route** | 每轮必进 | 检查 `state.attachments` 是否非空 | 有附件 → `image_agent`；否则直接 → `model_router` |
| **image_agent** | route 判定有附件 | ① OSS URL → 视觉模型识别图片正文 / pypdfium2 抽 PDF 文本；② `_with_attachments()` 把原始正文当轮注入；③ `_summarize_attachment()` 生成 1-2 句摘要写进 `state.messages`（自然持久化） | 更新后的 state（attachments 摘要已入 history） |
| **model_router** | 每轮必进 | 从 `state.mode` 字符串查 SkillRegistry → 拿到对应 system prompt + 工具集合 | 组装好 LLM 调用参数，交给 agent |
| **agent** | 每轮必进 | 模型推理 + ToolNode 执行。如果模型返回 tool_calls，ToolNode 执行后把结果追加回 state.messages，再次调用模型，**最多循环 5 轮** | 最终 answer 文本 + 可能的 tool_calls 残留在 messages 中 |
| **summarize** | agent 完成后必进 | ① `strip_calls`：从 messages 中剥离 tool_calls / tool 消息（这些是中间产物，不该持久化）；② 检查 history 条数：超过 40 条时用 summarizer 模型压缩到 20 条（`CHAT_SUMMARY_KEEP`） | 干净的、长度受控的 history → 写回 checkpoint |

### 关键设计点

1. **image_agent 的摘要机制**：原始附件正文只在当轮注入（`_with_attachments`），不进 history。进 history 的是压缩后的摘要。否则每轮重发全文会随 LangGraph checkpoint 永久膨胀。
2. **工具循环硬上限 5 轮**：防止模型陷入工具调用死循环。超过后 ToolNode 不再执行，直接返回当前已累积的 answer。
3. **strip_calls 在 summarize 节点**：ToolNode 的中间产物（tool_calls 消息、ToolMessage）在对话结束时统一清洗掉，checkpoint 里只留干净的 HumanMessage → AIMessage 对。

---

## 3. 两套向量索引对比

| 维度 | 校园库 | 附件库 |
|------|--------|--------|
| **物理存储** | `vectors.npy` + `docs.json`（numpy 数组 + JSON 文档表） | `milvus.db`（Milvus Lite 单文件） |
| **建索引方式** | 离线 CLI：`python -m app.rag.ingest` | 按需写入：附件上传后由 `doc_ingest` 自动切块入库 |
| **检索算法** | 稠密向量（余弦）+ BM25 稀疏 → **RRF 融合** → 规则重排（活动按时间、问答按热度） | 单路向量 top-4，无融合、无重排 |
| **隔离粒度** | 全校共用，无过滤（所有用户共享同一知识库） | **按 `file_url` 过滤**：每次检索限定在单个附件范围内 |
| **数据来源** | MySQL 业务库（tb_activity status∈{1,2} + tb_question status=1） | 用户上传的 OSS 文件（PDF 切块） |
| **更新频率** | 手动全量重建 / RabbitMQ 增量单条 | 随附件上传实时入库，删除附件时按 file_url 清理 |
| **指纹机制** | `sha1(text)[:16]` — 命中则复用旧向量，0 次 embedding | 无（每条切块都是唯一的） |
| **写锁** | `write_lock.py`：防止消费者与手动 ingest 并发覆盖 | Milvus Lite 自带文件锁，单进程内不会冲突 |
| **检索入口** | `app/rag/retriever.py` → `VectorIndex.search()` | `app/rag/doc_store.py` → `DocStore.search()` |

### RRF（Reciprocal Rank Fusion）公式

校园库融合检索的核心：

```
score(doc) = Σ 1 / (k + rank_i)   # k=60，两组检索各自的排名倒数求和
```

- 稠密向量 top-15 候选 + BM25 top-15 候选 → 合并去重 → RRF 打分 → 取 top-10 送规则重排
- 纯向量检索可能漏掉关键词精确匹配的场景，BM25 补足；纯关键词又会丢失语义相似度

---

## 4. 附件处理策略

附件正文的生命周期遵循「**当轮注入全文、History 只存摘要**」原则。这是为了避免随 LangGraph checkpoint 每轮重发全文而浪费 token。

| 附件类型 | 当轮注入内容 | History 持久化内容 | 触发条件 |
|---------|------------|-------------------|---------|
| **图片** | 视觉模型原始识别正文（`vision.py` → glm-4.6v） | `_summarize_attachment` 压缩版（1-2 句描述） | `type.startswith("image/")` |
| **短 PDF（<2000 字）** | `pypdfium2` 直抽全文（约 0.01s） | `_summarize_attachment` 压缩版 | PDF 文本层可用且字符数 < 2000 |
| **长 PDF** | 附件库向量检索 top-4 块（从 Milvus Lite 按 file_url 过滤后检索） | **不进 history**。追问时靠 `doc_urls` 复用，重新触发向量检索 | PDF 文本层 ≥ 2000 字，或 `pypdfium2` 抽不到文本 → MinerU 回退 |

### 为什么长 PDF 不进 history？

长 PDF 切块后总 token 量可能超过模型上下文窗口。即使摘要，长 PDF 的有效信息分布在多个切块中，一句话摘要会丢失关键细节。因此：
- 当轮只给模型 top-4 最相关的切块
- history 里不留任何附件正文
- 后续追问时，`route` 节点再次检测到附件，会重新触发向量检索，拿到新的 top-4

---

## 5. 会话持久化

服务维护五层持久化存储，各司其职：

| 存储层 | 位置 | 内容 | 写入时机 | 读取用途 |
|-------|------|------|---------|---------|
| **LangGraph Checkpoint** | MySQL（`mysql_db_agent` 库） | `state.messages` 完整对话历史（含 tool_calls 中间产物，但 summarize 节点已 strip） | 每轮对话结束 | 续聊时恢复上下文 |
| **tb_chat_message** | MySQL（`mysql_db_agent` 库） | 原始消息（用户输入 + 助手回复的最终文本） | MemoryStore 在 agent 节点结束后写入 | 前端 UI 展示聊天记录 |
| **tb_upload_record** | MySQL（`mysql_db_agent` 库） | 附件上传记录 + `vision_tokens`（视觉模型消耗的 token，与对话 token 分开计费） | /assistant/upload 接口 | 管理员审计 + 配额统计 |
| **tb_user_quota** | MySQL（`mysql_db_agent` 库） | 每日 token 用量 + 剩余次数配额 | 每轮对话后累加 | /assistant/chat 入口的配额预检 |
| **tb_thread_ownership** | MySQL（`mysql_db_agent` 库） | thread_id → user_id 映射 | 首次创建会话时写入 | 多租户隔离：确保用户 A 不能读取用户 B 的会话 |

### 多租户隔离

所有通过 `/assistant/chat` 的请求都必须携带有效 JWT。入口逻辑：

1. `require_user_id(http_request)` 从 `Authorization: Bearer <jwt>` 中解析 `userId`
2. `ensure_owner(thread_id, user_key)` 校验该 thread_id 是否归属此用户
3. 不匹配直接 403

注意：`/v1/responses` 端点不做此校验，`user_key` 留空，checkpoint 写入但不写业务表。这个端点是给 Codex CLI / Cursor 等外部客户端用的。
