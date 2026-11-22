---
name: campus-dev
description: 修改 campus-agent 代码、起服务、跑测试、改 prompt、加新工具时触发。触发词：改代码、起服务、跑测试、改 prompt、加工具、二手估价 prompt。排除：索引重建、SSE 格式改动、部署、容器重启。
---

# campus-dev

FastAPI + LangGraph 的校园问答单体服务，端口 8000。下面是日常开发的最短路径。

## 快速命令

```bash
uv sync                                        # 按 uv.lock 安装
cp .env.example .env                           # 必填 GLM_API_KEY / TENCENT_TOKENHUB_API_KEY / MYSQL_*
uv run uvicorn app.main:app --reload --port 8000
uv run pytest -q
uv run ruff check . && uv run ruff format .
pre-commit install                             # 提交前自动跑 ruff
```

## 代码地图

| 路径 | 职责 |
|---|---|
| `app/main.py` | lifespan（校验配置→建 Milvus→连 MySQL→编译图）、`/health` |
| `app/api/v1/chat.py` | `POST /assistant/chat`（SSE，`?stream=false` 返回 JSON）、消息 CRUD |
| `app/api/v1/upload.py` | `POST /assistant/upload`：落 OSS→解析→切块→入 Milvus |
| `app/agent/graph.py` | 7 节点图、档位 prompt、SkillRegistry 工具装配 |
| `app/agent/runtime.py` | `AIOMySQLSaver`、5s 探活、断连重建 |
| `app/agent/locks.py` | thread_id 级锁，防 last-write-wins |
| `app/agent/tools/` | `secondhand` / `activity` / `web`，各自带 `register()` |
| `app/rag/ingest.py` | 校园库全量构建 |
| `app/rag/chunker.py` | 切块：父标题注入、页码、块类型 |
| `app/rag/retriever.py` | 稠密 + BM25 双路→RRF→规则重排 |
| `app/memory/` | 历史表、日配额、会话归属 |
| `app/common/clock.py` | 北京时间真源（容器 UTC，别直接 `datetime.now()`） |

## Agent Loop 7 节点

```
summarize → image_agent → route → agent ⇄ tools → strip_calls → persist
```

summarize（>40 条压到 20）→ image_agent（图片转文字 + 附件块检索，正文只当轮注入不进 checkpointer）→ route（按档位装工具集）→ agent → ToolNode（最多 3 轮）→ strip_calls（超轮次逼收尾）→ persist。

## 加新工具的流程（SkillRegistry 注册制）

```python
# 1) app/agent/tools/lostfound.py
from app.agent.skills import SkillRegistry
from langchain_core.tools import tool

@tool
def search_lost(keyword: str) -> str: ...

def register(r: SkillRegistry) -> None:
    r.register_tools("market", [search_lost])
    r.register_tools("chat",   [search_lost])

# 2) graph.py 底部加一行 register 调用（仅此一行，内部不用碰）
from app.agent.tools import lostfound  # noqa: E402
lostfound.register(registry)
```

## 硬性约束（别碰）

1. 不要擅自启动服务、不要用真实付费 key 做验证（embedding/chat/视觉都计费）
2. 不要手改 `data/rag/*`，`vectors.npy` 和 `docs.json` 靠行号对齐
3. 不要在 tool 模块的 `register()` 外面硬拼工具列表
4. 不要把附件正文直接写进 HumanMessage，会随 checkpointer 永久留在 history
5. 不要改 `_handle_tool_error` 里的 fallback 逻辑
6. `E501` 完全不用加 noqa，format 自动折行；F821/E402 例外（见 AGENTS.md）
