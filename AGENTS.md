# AGENTS.md — Campus-Agent 仓库地图

> Codex Progressive Disclosure 风格：精简导航 + 指向深度文档。
> 改代码前先过一遍硬性约束；想知道架构细节、加新功能流程、两套向量索引对比 → 跳 docs/。

---

## 项目是什么

**CampusHub 校园应用的智能问答后端** — FastAPI + LangGraph 的 uvicorn 单体，提供两个 HTTP 接口：

- `POST /api/assistant/chat` — SSE 流式问答
- `POST /api/assistant/upload` — 附件（PDF / 图片）上传

核心能力：
- LangGraph 7 节点 Agent 编排：route → image_agent → model_router → agent ↔ tools → summarize
- 两套独立向量索引：校园库（稠密 + BM25 → RRF 重排）+ 附件库（Milvus Lite）
- 围绕**校园二手、校园活动、校园帖子、上传资料**四类场景的问答

这是一个校园垂直场景的问答系统，不是通用聊天机器人。不要引入与这四类无关的能力。

## 快速命令

```bash
# 起服务（开发模式，热重载）
uv run uvicorn app.main:app --reload --port 8000

# 跑测试（pytest，覆盖 agent/rag/api）
uv run pytest -q
uv run pytest tests/test_chat_flow.py::test_sse_stream_frames -v   # 单个用例

# Lint + format（提交前 pre-commit hook 自动跑）
uv run ruff check .
uv run ruff format .

# 重建校园索引（指纹命中则跳过 embedding，节省时间）
uv run python -m app.rag.ingest --limit 0       # 全量重建
uv run python -m app.rag.ingest --limit 50     # 只处理前 50 条（调试用）
```

## 硬性约束（别碰）

1. **不要擅自启动服务、不要用真实付费 key** — 所有测试走 Fake model（tests/conftest.py 里已有 fixture）
2. **不要手改 `data/rag/*`** — vectors.npy 与 docs.json 靠行号对齐，错位不报错只答错。重建必须用 ingest 命令
3. **Milvus Lite 单进程独占** — 启动服务期间不要同时跑 ingest，会锁死 milvus.db
4. **四张表无建表 SQL** — tb_chat_message / tb_upload_record / tb_user_quota / tb_thread_ownership 的 DDL 在部署脚本里（不在 git）
5. **工具注册统一走 SkillRegistry** — 禁止在 tool 模块的 register() 外面硬拼 mode→tools 映射
6. **附件正文不进 HumanMessage 持久化** — 原始正文当轮注入（_with_attachments），摘要版进 history（image_agent 里的 _summarize_attachment）
7. **vision_tokens 独立计费** — 视觉识别 token 与对话 token 分开算，持久化时写入 tb_upload_record.vision_tokens
8. **`_handle_tool_error` fallback 不动** — ToolNode 层兜底逻辑，每个 tool 自己已经有 try/except，除非加新工具

## Lint 规范

```bash
uv run ruff check .        # 规则集 E/W/F/I/B/UP，per-file-ignore 见 pyproject.toml [tool.ruff.lint]
uv run ruff format .       # double quote · space indent · 100 列
pre-commit install         # 首次安装 hook，提交前自动跑 ruff check --fix + ruff-format
```

禁止手动加 noqa，除非两种情况：
- `F821` — register 函数里 SkillRegistry 类型注解用字符串（避免循环 import）
- `E402` — graph.py 底部 Skill 注册区块的 import（延后导入避免循环）

`E501`（行超长）永远交给 format 自动折行，不要加 noqa。

## 深度文档索引（都在 docs/）

| 想知道 | 去哪里找 |
|---|---|
| 完整代码地图 / 目录职责 / 各模块入口文件 | `docs/ARCHITECTURE.md` § Code Map |
| Agent 7 节点图 / 数据流 / 每步职责 | `docs/ARCHITECTURE.md` § Graph |
| 两套向量索引对比（校园库 vs 附件库） | `docs/ARCHITECTURE.md` § RAG |
| 附件处理策略（图片 / 短 PDF / 长 PDF） | `docs/ARCHITECTURE.md` § Attachments |
| LangGraph checkpointer / 会话持久化 / 多租户隔离 | `docs/ARCHITECTURE.md` § Persistence |
| 加新工具 / 新 Skill 的注册流程（Step by Step） | `docs/CONTRIBUTING.md` § New Skill |
| 环境变量配置 / 部署 | `docs/DEPLOY.md` 或直接看仓库根目录 `.env.example` |

## 相关 Skill

这个仓库有一个配套的 Trae Skill（动态加载，优先查 Skill 再碰代码）：

- **campus-agent** — 架构地图、启动与索引命令、检索与文档链路、症状到位置的排错表。

找不到对应 Skill 或者 Skill 没覆盖，直接 grep 源码。
