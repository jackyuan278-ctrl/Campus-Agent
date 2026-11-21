# 运行与部署

## 本地

```bash
uv sync                                        # 依赖以 uv.lock 为准；改了 pyproject 必须 uv lock
cp .env.example .env                           # 填 GLM / TokenHub / MySQL
uv run uvicorn app.main:app --reload --port 8000
uv run pytest -q                               # 会连真实库（checkpointer 会写表），不擅自跑
```

本地想连线上库调试时注意：`ensure_owner` 会往 `tb_thread_owner` 写记录，`persist` 会写历史表 —— 用测试 `user_key` 才不会污染真实配额。

## 索引

| 目的 | 命令 | 花钱吗 |
|---|---|---|
| 重建校园索引（推荐） | `uv run python -m app.rag.ingest --limit 0` | **不花**：指纹命中就复用旧向量，只读库 + 重写两个文件 |
| 强制重算全部向量 | `uv run python -m app.rag.ingest --limit 0 --full` | 花：每条一次 embedding |
| 只看规模与新鲜度 | `uv run python .agents/skills/campus-agent/scripts/index_stats.py` | 不花 |
| 附件索引 | 走上传接口，`upload → MinerU/pypdfium2 → 切块 → Milvus` | 花：切块数 × embedding |

判断「重建后有没有生效」：日志里 `索引已保存：N 条`，N 应与库里有效活动 + 问答条数对得上（活动按 `status IN (1,2)` 且 `end_time` 未过）。

## 服务器

| 项 | 值 |
|---|---|
| 主机 | 腾讯云 ECS，`ubuntu@124.220.155.27`（用 `cww.pem` 登录） |
| 编排文件 | `/opt/campushub/deploy/docker-compose.yml`（服务名 `agent`） |
| 容器 | `CampusHub-agent`，宿主网络，端口 8000 |
| 数据卷 | `/opt/campushub/deploy/agent-data/rag` → 容器 `/app/data/rag` |

部署（服务器上执行）：

```bash
cd /opt/campushub/deploy
docker tag campushub/agent:1.0.0 campushub/agent:prev     # 留一份可回滚的镜像
docker compose build agent
docker compose up -d agent
docker logs --tail 40 CampusHub-agent
```

回滚：

```bash
docker tag campushub/agent:prev campushub/agent:1.0.0
docker compose up -d --force-recreate agent
```

**不要 `docker compose down -v`** —— `-v` 会把挂载卷一起删掉，索引得重建。

## 日志关键词

| 关键字 | 含义 |
|---|---|
| `状态图已就绪（checkpointer=MySQL/...）` | 启动成功，checkpointer 已连上 |
| `索引已加载：N 条，维度 1024` | **读文件**（load），毫秒级，不花钱 |
| `索引构建完成：N 条` / `索引已保存：N 条` | **跑过 ingest**（build），可能花钱 |
| `向量化完成：N 条` | embedding 调用完成 |
| `附件入索引：xxx.pdf（N 块）` | Milvus 写入成功 |
| `文档入索引：... 路由 / N 块 / 切分 400+80` | 文档链路完成 |
| `稠密 N / 关键词 N / 融合 N` | 两路召回候选数；关键词为 0 说明 BM25 没命中 |
| `检索命中 N 条，最高分 x` | 融合 + 重排后的最终结果 |
| `route：thread=... mode=... intent=... 附件=N` | 本次请求的路由判断 |
| `工具执行失败` | 工具异常已降级成话术（接口仍 200） |
| `疑似提示注入` | 命中检测规则，只告警不改内容 |

常用过滤：

```bash
docker logs --since 30m CampusHub-agent 2>&1 | grep -E '检索命中|稠密|文档入索引|工具执行失败'
```

## 部署后验收

1. `curl 127.0.0.1:8000/health` → `{"status":"ok"}`
2. 日志有「状态图已就绪」+「索引已加载：N 条」
3. 网关链路：`curl 127.0.0.1:8080/api/assistant/chat/messages?thread_id=probe` → `401`（agent 自己抛的，说明 `8080 → StripPrefix → 8000` 通）
4. 一条真实对话（计费，需本人同意）

## 容器与资源

- 内存上限 512MB，正常占用 180~280MB；解析大 PDF 时会涨
- `TZ=Asia/Shanghai` 由 compose 注入；**代码里仍然显式用 `ZoneInfo`**，不依赖环境变量
- Milvus Lite 单文件库，多 worker 会因 `LOCK` 失败；保持 `uvicorn` 单进程
