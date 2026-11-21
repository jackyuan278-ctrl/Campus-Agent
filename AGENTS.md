# AGENTS.md — Campus-Agent 仓库规范（给 AI 读的）

> Codex 叫这 "Repository Knowledge as System of Record"。
> 改代码之前先读一遍，别踩坑。写代码时遇到不确定的硬性约束，回来查这里。

---

## 项目是什么

**CampusHub 校园应用的智能问答后端** — 一个 uvicorn 单体同时管：

- FastAPI HTTP 接口（`/api/assistant/chat`、`/api/assistant/upload`）
- LangGraph 7 节点 Agent 编排（路由 → 图片识别 → 档位选择 → 模型推理 → 工具调用 → 清洗 → 持久化）
- 两套独立向量索引（校园库 + 附件库）

不是通用聊天机器人。所有回答都围绕"校园二手、校园活动、校园帖子、上传的 PDF/图片资料"这四类场景。

---

## 快速上手

```bash
# 起服务
uv run uvicorn app.main:app --reload --port 8000

# 跑测试
uv run pytest -q
# 单独跑
uv run pytest tests/test_chat_flow.py::test_sse_stream_frames -v

# Lint + format（提交前 pre-commit hook 自动跑）
uv run ruff check .
uv run ruff format .

# 重建校园索引
uv run python -m app.rag.ingest --limit 0       # 全量（指纹命中就 0 次 embedding）
uv run python -m app.rag.ingest --limit 50     # 只处理前 50 条（调试用）
```

---

## 代码地图

```
campus-agent/
├── app/
│   ├── main.py              ← FastAPI app 工厂 + lifespan
│   ├── config.py            ← 单层 Settings（.env 读入）
│   ├── api/v1/              ← HTTP handler（chat.py, upload.py）
│   ├── agent/               ← LangGraph Agent 核心 ⭐
│   │   ├── graph.py         ← 7 节点图定义 + System Prompts（所有档位的人设都在这）
│   │   ├── state.py         ← AgentState TypedDict
│   │   ├── skills/          ← SkillRegistry 注册中心（新增）
│   │   └── tools/           ← 工具包（每个包末尾都有 register()）
│   │       ├── secondhand.py   ← 二手查询工具
│   │       ├── activity.py     ← 校园活动检索工具
│   │       └── web.py          ← 联网搜索（只给 casual 档）
│   ├── llm/                 ← LLM 封装
│   │   ├── gateway.py       ← get_chat_model() / get_embedding_model() 工厂
│   │   ├── vision.py        ← 图片视觉识别（glm-4.6v）
│   │   ├── pdf_text.py      ← PDF 文本层直抽（pypdfium2，0.01s）
│   │   └── mineru.py        ← MinerU 回退（60s，只有文本层不可用时才用）
│   ├── rag/                 ← RAG 两套索引
│   │   ├── index.py         ← 校园库：vectors.npy + docs.json（稠密 + BM25 → RRF）
│   │   └── doc_store.py     ← 附件库：Milvus Lite（milvus.db），按 file_url 隔离
│   ├── memory/              ← 会话持久化 + 配额
│   ├── database.py          ← MySQL 连接（aiomysql）
│   └── common/              ← logger, clock, security, constants, exceptions
├── tests/                   ← pytest，11 个文件覆盖 agent/rag/api
├── data/                    ← 向量索引文件（.gitignore）
└── .env.example             ← 环境变量模板
```

---

## 硬性约束（别碰）

1. **不要擅自启动服务、不要用真实付费 key 做验证** — 测试全用 Fake model
2. **不要手改 `data/rag/*`** — vectors.npy 和 docs.json 靠行号对齐，错位不报错只答错。重建用 `uv run python -m app.rag.ingest`
3. **Milvus Lite 单进程独占** — 启动服务期间不要同时跑 ingest，会锁死 milvus.db
4. **tb_chat_message / tb_upload_record / tb_user_quota / tb_thread_ownership 四张表** — 仓库里没建表 SQL，部署脚本是 `scripts/deploy/` 里的（不在 git 上）
5. **不要在 tool 模块的 register() 外面硬拼工具列表** — 所有 mode→tools 映射都走 SkillRegistry
6. **不要把附件正文直接写进 HumanMessage** — 会随 checkpointer 永久留在 history，每轮重发浪费 token。应该：原始正文当轮注入（_with_attachments），摘要版本进 history（image_agent 里的 _summarize_attachment）
7. **不要把 vision_tokens 当普通 token 计入模型调用** — vision_tokens 是视觉识别的，和对话 token 分开算，持久化时写入 tb_upload_record.vision_tokens
8. **不要修改 `_handle_tool_error` 里的 fallback 逻辑** — 那是 ToolNode 层的兜底，每个 tool 自己已经有 try/except，除非你在加新工具

---

## 加新功能的流程

SkillRegistry 注册制，graph.py 不再硬编码：

```python
# 第一步：在 app/agent/tools/ 下新建文件（或在已有包里加函数）
# app/agent/tools/lostfound.py
from app.agent.skills import SkillRegistry
from langchain_core.tools import tool


@tool
def search_lost_items(keyword: str) -> str:
    """搜索失物招领物品。"""
    # 你的实现


LOSTFOUND_TOOLS = [search_lost_items]


# 第二步：声明自己服务哪些档位
def register(r: SkillRegistry) -> None:  # noqa: F821
    r.register_tools("market", LOSTFOUND_TOOLS)  # 二手估价档能用
    r.register_tools("chat", LOSTFOUND_TOOLS)  # 智能问答档能用
    # casual（随便聊聊）档不给——保持联网纯净


# 第三步：graph.py 底部加一行 register 调用（仅此一行）
# 在 graph.py 底部的 Skill 注册区块：
#     from app.agent.tools import lostfound  # noqa: E402
#     lostfound.register(registry)
```

**只改这三个地方** — graph.py 内部（tools_for_mode、ToolNode、各节点）一行都不用碰。

---

## Lint 规范

```bash
uv run ruff check .      # lint（E/W/F/I/B/UP，per-file-ignore 见 pyproject.toml [tool.ruff.lint.per-file-ignores]）
uv run ruff format .     # 格式化（double quote, space indent, 100 列）
pre-commit install      # 第一次装 hook，提交前自动跑 ruff check --fix + ruff-format
```

**不要手动加 noqa** — 除非你确认：
- `F821` 出现在字符串注解里（register 函数的 SkillRegistry）
- `E402` 出现在 graph.py 底部的 Skill 注册 import（避免循环 import）
- `E501` 完全不用加，format 自动折行

---

## 架构要点

### 两套向量索引（容易混）

| | 校园库 | 附件库 |
|---|---|---|
| 存储 | `vectors.npy` + `docs.json` | Milvus Lite (`milvus.db`) |
| 检索 | 稠密 + BM25 → RRF → 规则重排 | 单路向量 top4 |
| 隔离 | 全校共用 | `file_url` 过滤 |

### Agent Loop 7 节点（graph.py）

```
START → route → image_agent → model_router → agent ←→ tools
                  ↓ (图片摘要进 history)           ↓ (strip_calls)
                 └───────────────────────────── summarize → END
```

- **route**：根据 attachments 有无决定要不要走 image_agent
- **image_agent**：图片视觉识别 + 同步做 _summarize_attachment，摘要写进 state.messages（自然持久化）
- **model_router**：mode 字符串 → system prompt + tool 集（走 SkillRegistry）
- **agent**：模型推理 + ToolNode 执行（工具循环上限 5 轮）
- **summarize**：history 超 40 条 → 用 summarizer 模型压到 20 条（CHAT_SUMMARY_KEEP）

### 附件处理策略

| 附件类型 | 当轮注入 | History 持久化 |
|---|---|---|
| 图片 | 原始视觉识别正文（_with_attachments） | _summarize_attachment 压缩版（1-2 句） |
| 短 PDF（<2000 字） | 全文当轮注入 | _summarize_attachment 压缩版 |
| 长 PDF | 向量检索 top 4 块（当轮注入） | 不进 history，追问靠 doc_urls 复用重新检索 |

### 会话持久化

- LangGraph checkpoint（MySQL 后端）存 state.messages 完整对话
- tb_chat_message 存原始消息（用户端 UI 展示用）
- tb_upload_record 存附件上传记录 + vision_tokens
- tb_user_quota 存用户配额（每天 20 次上传）
- tb_thread_ownership 存 thread_id → user_id 映射（多租户隔离）
