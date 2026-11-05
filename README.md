# CampusAgent

CampusHub 校园平台的智能助手服务。FastAPI + LangGraph 编排，用一个对话式 Agent 串起「二手商品查询 / 活动与社区检索 / 联网搜索 / 图片理解」四类能力。

不是独立的聊天机器人：它挂在 CampusHub 微服务集群后面，共用同一套 JWT 鉴权、同一个 MySQL、同一批业务表。

---

## 目录

- [能力](#能力)
- [技术栈](#技术栈)
- [架构](#架构)
- [快速开始](#快速开始)
- [关键设计](#关键设计)
- [踩坑记录](#踩坑记录)
- [已知限制](#已知限制)

---

## 能力

| 能力 | 实现方式 |
|---|---|
| 二手商品查询 | 模型自主调用 SQL 工具（关键词/价格上限/分类过滤） |
| 活动与社区问答 | 向量检索（本地 numpy 索引，1024 维） |
| 联网搜索 | Tavily，**仅在「随便聊聊」档位开放** |
| 图片理解 | glm-4.6v 识图转文字，拼进当轮提问 |
| 文档问答 | MinerU 云端解析 PDF/Word → 切块 → Milvus Lite；提问时只带最相关的 4 块 |
| 多轮对话 | MySQL checkpointer 持久化，超长自动摘要压缩 |
| 鉴权 | 与 campus-hub 共用 HS256 密钥，`sub` = userId |
| 配额 | 每人每日 token 上限 + 每日上传次数，超限直接拦截 |
| 会话隔离 | `tb_thread_owner` 校验归属，猜 thread_id 也读不到别人的对话 |

## 技术栈

| 层 | 选型 | 说明 |
|---|---|---|
| Web | FastAPI + uvicorn | SSE 流式返回，逐 token 推送 |
| 编排 | LangGraph (StateGraph) | 状态机 + checkpointer，节点职责单一 |
| 对话模型 | glm-5.3-flash | OpenAI 兼容协议；**强制思考，关不掉** |
| 视觉模型 | glm-4.6v | 同一个 GLM key，`image_url` 多模态块 |
| Embedding | 腾讯 TokenHub `kinfra-text-embedding-0.6b` | 1024 维；必须显式传 `encoding_format="float"` |
| 校园向量索引 | numpy 矩阵 + 余弦相似度 | 579 条固定数据，全量重建，不引入向量库 |
| 附件向量索引 | Milvus Lite (`milvus-lite` + `pymilvus`) | 进程内、落本地文件，按 `file_url` 隔离检索 |
| 文档解析 | MinerU 云端 API | 提交 URL → 轮询 → 取 zip 里的 `full.md` |
| 存储 | MySQL（aiomysql） | 4 个库：activity / community / secondhand / agent |
| 消息队列 | RabbitMQ (aio-pika) | 增量索引的单条同步 |
| 对象存储 | alibabacloud-oss-v2 | 图片上传（`oss2` 装不上，crcmod 要编译） |

## 架构

```
                         ┌──────────────────────────────────────────┐
  POST /assistant/chat   │  summarize  上下文压缩（超阈值才调模型）  │
  ──────────────────────>│      ↓                                   │
                         │  image_agent  图片 → glm-4.6v → 文字     │
                         │      ↓                                   │
                         │  route  判意图（只影响 prompt 和日志）   │
                         │      ↓                                   │
                         │  agent  ──┬─(要工具)─> tools ──┐        │
                         │           │                    │        │
                         │           └─(收尾)─> strip_calls ─> persist
                         └──────────────────────────────────────────┘
                                                   ↓
                                    tb_chat_message + tb_token_usage
```

三个「记忆」组件，职责不要混：

| 组件 | 存什么 | 管什么 |
|---|---|---|
| checkpointer | 状态快照（原样、无损） | 跑得下去 |
| summarize | 压缩后的 messages（有损、保语义） | 这轮发多少 |
| tb_chat_message | 可读的对话记录 | 人查得到 |

## 快速开始

### 1. 环境

```bash
uv sync                    # 依赖全部走 uv
```

然后在项目根目录创建 `.env`：**直接复制模板再改**，比对着文档手敲不容易漏。

```bash
cp .env.example .env
```

必填配置（缺任意一个，服务直接起不来）：

```ini
MYSQL_HOST=...
MYSQL_USER=...
MYSQL_PASSWORD=...
MYSQL_DB_ACTIVITY=campushub_activity
MYSQL_DB_COMMUNITY=campushub_community
MYSQL_DB_SECONDHAND=campushub_secondhand
MYSQL_DB_AGENT=campushub_agent
GLM_API_KEY=...
GLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
GLM_CHAT_MODEL=glm-5.3-flash
```

按需配置（不配则该能力降级，不影响启动）：

```ini
TENCENT_TOKENHUB_API_KEY=...        # 不配则 RAG 检索不可用
TENCENT_TOKENHUB_MODEL_NAME=kinfra-text-embedding-0.6b
TENCENT_TOKENHUB_BASE_URL=...
TAVILY_API_KEY=...                  # 不配则联网搜索返回空
GLM_VISION_MODEL=glm-4.6v           # 不配则识图功能不可用
OSS_ACCESS_KEY_ID=...               # 不配则上传接口 500
OSS_ACCESS_KEY_SECRET=...
OSS_BUCKET=...
MINERU_API_KEY=...                  # 不配则文档只上传不解析
MILVUS_LITE_PATH=/app/data/rag/milvus.db   # 附件向量库落盘位置
JWT_SECRET=...                      # 必须和 campus-hub 一致，否则验不出 userId
RABBITMQ_URL=amqp://user:pass@host:5672/
RABBITMQ_INDEX_QUEUE=campus.index.rebuild
DAILY_TOKEN_QUOTA=100000
DAILY_UPLOAD_LIMIT=5
```

### 2. 建索引（**部署必做**）

`data/` 在 `.gitignore` 里，仓库不含索引文件。克隆下来必须自己跑一次：

```bash
uv run python -m app.rag.ingest --limit 0     # 全量，复用旧向量
```

不跑的话检索直接返回空，模型会说「没找到」——不是 bug，是没数据。

附件向量库（Milvus Lite）**不用建**：第一次上传文档时自动创建，文件落在
`MILVUS_LITE_PATH`（默认 `data/rag/milvus.db`）。但它的父目录必须存在——
服务启动时会自动 `mkdir`，所以别把这个路径指到只读挂载上。

### 3. 启动

本地：

```bash
uv run python -m app.main          # 或 uv run uvicorn app.main:app --port 8000
uv run python -m app.rag.consumer  # 可选：增量索引消费者，独立进程
```

Docker：

```bash
cp .env.example .env && vim .env           # 填配置
docker compose build
docker compose run --rm agent python -m app.rag.ingest --limit 0   # 建索引（写进挂载卷）
docker compose up -d
docker compose logs -f agent
```

`./data` 是挂载卷，同时放 numpy 索引和 `milvus.db`。**不挂卷的话容器一重建，
附件索引就全丢了。**

健康检查：`GET /health` → `{"status": "ok"}`（只证明进程活着，不查任何外部依赖）

### 4. 测试

```bash
uv run pytest -q                   # 41 passed
uvx ruff check app tests           # 干净
```

测试全部用假模型（`GenericFakeChatModel`）+ 假数据，**不产生任何 API 费用**。

---

## 关键设计

以下五条是这个项目里真正想过的地方，每条都附「为什么不能反过来做」。

### 1. 工具集静态装配，绝不按对话内容裁剪

**做法**：`tools_for_mode(mode)` 只回答一件事——这轮要不要给联网工具。本地三个工具（二手查询 ×2 + RAG 检索）**永远全给**。

```python
_LOCAL_TOOLS = SECONDHAND_TOOLS + CAMPUS_SEARCH_TOOLS

def tools_for_mode(mode):
    return _ALL_TOOLS if mode == "casual" else _LOCAL_TOOLS
```

**之前是反的**：按意图（关键词规则）挑一组工具。结果用户问「有二手自行车吗」正常，追问「最便宜的是哪个」时——这句话没有任何关键词，被判成 `other`，工具集返回空列表。模型这一轮**连工具名都看不到**，只能拿上一轮那 5 条旧结果复述，答出来的是「那 5 条里最便宜的」，不是全库最便宜的。

**为什么不能靠「继承上一轮意图」补救**：一旦落进 `other` 就永久粘在最后一个识别成功的意图上——「谢谢」「今天天气怎么样」都会继承「二手」，越走越偏，而且日志里看不出来。

**为什么不能「猜得更准」补救**：任何基于用户输入的判定都会判错，关键是**判错的后果可不可恢复**。工具消失是不可恢复的（模型连补救入口都没有），而「多给几个工具」的代价只是每轮多 300 token。

> 参考 OpenAI Codex 的 `build_tool_router`：它的入参里根本没有用户消息，裁剪维度全是模型能力 / 权限 / 预算。工具太多时用 Deferred + Tool Search 渐进披露，也不是按意图砍。

### 2. RAG 索引按文本指纹去重 + 增量更新

**问题**：campus-hub 的造数脚本用 `ELT()` 拼标题正文，1000 条问答实际只有 34 种内容，一字不差。全量灌进索引后，这几十条同内容向量的相似度几乎相同，会霸占检索候选池（`top_k × 3 = 15` 条）的前几名，去重后只剩 1 条——`top_k=5` 实际只返回 1~2 条。

**做法**：每条文档存 `sha1(text)[:16]` 作为指纹。

| 用途 | 怎么用 |
|---|---|
| 去重 | 指纹相同只留一条，1737 → 579 |
| 复用向量 | 指纹没变就直接用旧向量，重跑全量 **0 次 embedding 调用** |
| 增量防重复 | 新文档若与库内同指纹，先删旧的再插，避免重复向量回流 |

**为什么用指纹不用 id**：id 标识「哪一行」，指纹标识「内容是什么」。造数脚本那个坑正是 1000 个 id 只有 34 种内容——按 id 看不出重复。

**增量链路**（RabbitMQ）：

```
业务表增删改 → 消息 {"source":"activity","id":123} → 消费者回查数据库
                                                      ├─ 查到且在架 → 删旧行 + 算新向量 + 追加（1 次调用）
                                                      └─ 查不到/已下架 → 只删旧行（0 次调用）
```

消息**只带 id、不带内容**，所以增/删/改三条路径合成一条，天然幂等（重复投递、乱序都不怕）。

### 3. 上下文压缩用摘要，不用截断

超过 40 条消息时，把早期对话交给模型压成一段摘要（`SummarizationMiddleware`，保留最近 20 条）。

**为什么不能按条截断**：截断会直接丢掉「用户想找什么、已经查到什么」，下一轮模型就断片。摘要是有损但保语义的，截断是无损但断语义的——后者更糟。

摘要提示词是自定义的中文版本：必须保留用户核心需求、已知条件、已查到的结果、未满足的问题。中间件写死的英文前缀 `Here is a summary of the conversation to date:` 会被就地替换成「【此前对话摘要】」。

### 4. 无图时必须显式清空附件状态

`image_agent` 每次都会跑（所有档位共用）。**没有新图时它必须返回 `attachment_text: ""`**，不返回就会踩坑：

checkpointer 保留上一轮的值，那张图的文字会拼进**之后每一轮**的提问。表现是「换了话题它还在说那张图」，而且视觉 token 会重复计入配额。

### 5. 文档切块检索，不整篇塞进 prompt

一份 50 页讲义解析出来约 3 万字。整篇塞进 prompt 就是 3 万 token，而日配额只有 10 万——**三轮就爆**，而且每轮都要重发一遍历史。

所以上传时就切块入库，提问时只带最相关的 4 块（约 1600 字 ≈ 1000 token），差 30 倍。三个配套决定：

| 决定 | 为什么不能反过来 |
|---|---|
| 切块保留 80 字重叠 | 答案经常正好落在切分点上，不重叠就会「明明有答案却检索不到」 |
| 提问时**只在这一份文件内**检索 | 混进全局索引的话，问「图书馆几点开门」会检索到某份讲义里的句子 |
| 检索不到就明说没找到 | 模型看不到全文，让它「自由发挥」等于允许它编造没读到的章节 |

附件块放在 Milvus Lite 里，而校园库仍然用 numpy——这不是重复建设：校园库是 579 条固定数据、只读、全量重建；附件块按文件增删、只对上传者可见，两者生命周期完全不同。

---

## 踩坑记录

| 坑 | 表现 | 解法 |
|---|---|---|
| **同一会话并发** | 同一 thread_id 并发 4 个请求，**全部返回 200**，但 checkpoint 只剩 1 问 1 答，前三轮对话静默消失。checkpointer 是 last-write-wins、没有版本冲突检测，4 个请求都读到旧状态再各自写回 | 应用层按 thread_id 串行（`app/agent/locks.py`），跨会话仍并行 |
| `aiomysql` × `pymysql>=1.2` | 连接随机报错（`escape_bytes_prefixed` 变成哨兵字符串） | pyproject 钉 `pymysql>=1.1,<1.2` |
| 建库字符集 | `Illegal mix of collations` | 统一 `utf8mb4_0900_ai_ci` |
| SSE 泄漏摘要正文 | 聊天框里突然冒出摘要文字 | 只转发 `metadata.langgraph_node == "agent"` 的 `on_chat_model_stream` |
| `Pipeline.execute()` | 出 `async with` 块后 `command_stack` 被 reset | 必须在块内执行 |
| glm-5.3-flash 思考 | `max_tokens` 给太小（如 20）时正文为空，全被推理吃掉 | 至少 1500 |
| 思考参数形状 | `thinking` 必须是对象，`type` 只认 `enabled`/`disabled` | `{"type":"enabled","level":"low"}` |
| 工具轮数用尽 | 模型第 N 轮仍要工具，`should_use_tools` 直接跳 `persist`，带 `tool_calls` 的消息落进 checkpointer → 下轮上游判 400，会话永久卡死 | 加 `strip_calls` 节点在收尾前用 `RemoveMessage` 替换掉 |
| 索引文件非原子写 | `np.save` 成功后写 json 失败 → 行数不匹配 → 检索返回错位的文档且**不报错** | 写临时文件 + `os.replace` 原子替换；加载时校验 `len(docs) == vectors.shape[0]` |
| **工具执行抛异常** | 数据库超时、embedding 失败这类业务异常会让**整张图崩掉**，用户拿到 500，模型连「查询失败」都说不出口 | LangGraph 默认只吞 `ToolInvocationError`，其余一律 raise。传入自定义 `handle_tool_errors`，降级成 `ToolMessage(status=error)`，原始堆栈只进日志 |
| **索引两个写入方互相覆盖** | 消费者和手动 `ingest` 同时跑时，各自读到同一份旧索引再整体写回，后写的盖掉前一个的改动（原子写救不了更新丢失） | `app/rag/write_lock.py` 用 MySQL `GET_LOCK` 锁住「读 → 改 → 落盘」整段，包含中间的 embedding 调用 |
| **消费者锁静默失效** | `GET_LOCK` 挂在那条长连接上，空闲超过 `wait_timeout`（默认 8h）连接被回收 → 锁跟着释放，进程还在消费，重复启动也拦不住 | 每 30 分钟用 `IS_USED_LOCK` 比对 `CONNECTION_ID`，丢了就重抢；不用重复 `GET_LOCK`（5.7+ 会累加持有计数） |
| **模块级 `asyncio.Semaphore`** | 在第一个事件循环上耗尽后，换一个 loop 再申请会**永久挂起且不报错** | 按当前 loop 惰性创建，发现 loop 变了就重建 |
| 附件 URL 未校验 | 传 `http://169.254.169.254/...` 也能过，等于把「让视觉模型抓任意地址」的口子开给调用方 | schema 层校验：只收公开 http/https，挡掉回环 / 私有网段 / 链路本地 / `.local` `.internal`；顺带把附件数限到 5（与前端 `ATTACHMENT_RULES.maxCount` 一致） |
| 上传先扣配额 | OSS 失败也消耗当天次数，网络抖几下当天就不能传了 | 改成「只读预检 → 上传 → 成功后扣次数」 |
| 多进程改索引 | 独立进程（ingest / 消费者）改完文件，API 进程的缓存还是旧的 | `retriever` 每次 stat 文件 mtime，变了就重载 |
| 前端 Pinia store | 改了 store 代码 HMR 换不掉已创建实例 | 关标签页重开 |
| 前端消息渲染 | 直接改 push 进去的原始对象，Vue 拦截不到 | 从数组取回代理引用再改 |
| `milvus-lite` API 与文档不符 | `get_load_state()` 在 pymilvus 3.x 返回 **dict**（`{'state': LoadState.Loaded}`）而非带 `.state` 的对象；`describe_collection()['fields'][1]` 也不能假设是向量字段 | 两种返回形状都兼容；按「哪个字段带 `dim`」找向量字段 |
| MinerU 把代码渲染成 LaTeX | `dp[0] = 0` 会变成 `$\mathsf { d p } [ 0 ] = 0$`，正文变长且不利于关键词匹配 | 保留（关掉 `enable_formula` 会连真公式一起丢），语义检索仍能命中 |
| Milvus Lite 刷 gRPC 噪音 | 每次建集合都打一条 `Method not implemented!`（ERROR），实际不影响功能 | 在 `setup_logging` 里给 `grpc` logger 加过滤器，只丢这一条 |

## 已知限制

- **并发**：同一会话的请求在应用层串行（进程内 `asyncio.Lock`）。uvicorn 开多 worker
  时不跨进程，需要换成 Redis 分布式锁；当前单 worker 部署够用。
- **首字延迟约 2.5~3s**。实测 `glm-5.3-flash` 和 `glm-4.5-air` **都强制思考**（流式响应里约 1s 就出现 `reasoning_content`），换模型降不下来。`type: disabled` 未验证成功。
- **索引只收 `status=1` 的记录**：已结束的活动不进索引（推荐一个已经结束的活动没意义）。这是产品决策，不是 bug。
- **摘要消耗的 token 不计入配额**（占比 < 1%，暂不处理）。
- **SSE 中途断开可能不落库**：`persist` 节点没跑到。
- **`tb_chat_message` 不存附件信息**，只存文本。
- **删会话不删 OSS 文件**：图片永久留在桶里。
- **删会话也不清附件向量块**：Milvus Lite 里那份文件的块会留着（按 `file_url` 隔离，
  搜不到别人的，但占空间）。要清理得先解决「会话 ↔ 附件」的关联关系。
- **Milvus Lite 是单进程库**：uvicorn 开多 worker 时每个进程各一份 `milvus.db`，
  写入会互相看不见。当前单 worker 部署够用，要多实例就换成真正的 Milvus 服务。
- **文档解析在上传时同步做**：一份 PDF 会让 `/upload` 多等 30~60 秒。换来的是
  提问时零等待、解析失败也不影响文件已经传上去这件事。
- **扫描件只拿到文字层**：MinerU 会 OCR，但正文里的图表、公式图片不在 markdown 里，
  只有文字。提示里会说明「原文有 N 张图未解析」，避免模型假装看懂了图。
- **Docker 构建未验证**：`Dockerfile` 还是旧的依赖快照。
