---
name: deploy-ecs
description: 部署、重启容器、回滚、排查线上问题时触发。触发词：部署、deploy、容器、重启、回滚、ECS、docker。排除：日常代码改动、索引重建。
---

# deploy-ecs

生产跑在腾讯云 ECS，容器名 `campus-agent`，端口 `8000`，由 CampusHub 网关把 `/api/assistant/**` 转发进来（StripPrefix 后交给本服务）。

## 部署编排（docker-compose.yml）

```yaml
services:
  agent:
    build: .
    image: campus-agent:latest
    container_name: campus-agent
    env_file: .env
    ports: "8000:8000"
    volumes: ./data:/app/data
    restart: unless-stopped
```

不依赖 MySQL/Redis/Milvus 容器——MySQL 用现成的 CampusHub 那套，附件向量库是 Milvus Lite（进程内跑）。

## 首次部署顺序

```bash
cp .env.example .env && vim .env       # 必填 MYSQL_HOST / GLM_API_KEY / TENCENT_TOKENHUB_API_KEY
docker compose build
docker compose run --rm agent python -m app.rag.ingest --limit 0   # 索引是必需文件
docker compose up -d
curl 127.0.0.1:8000/health            # {"status":"ok"}
```

## MySQL 连接（4 只读 + 1 读写）

- 仓库里**没有建表 SQL**，部署脚本不在 git 上
- 涉及的表：`tb_chat_message` / `tb_upload_record` / `tb_user_quota` / `tb_thread_ownership`
- 连接用 `aiomysql`（见 `app/database.py`）

## OSS bucket

- 附件上传后落 OSS，**bucket 必须公共读**（否则 MinerU 解析时 403）
- 上传链路：客户端直传 OSS → 拿 URL → `POST /assistant/upload` 传 URL → 服务端解析 → 切块 → 入 Milvus Lite

## Milvus Lite 生产注意

- 单进程独占（目录里有 `LOCK`），**不要开多 worker**
- `./data` 卷同时放着 numpy 索引 + Milvus Lite 的 `milvus.db`
- 不挂卷的话容器一重建，附件索引全部丢失

## 回滚 & 日志

```bash
docker compose logs -f --tail=200 agent          # 实时日志
docker compose stop agent && docker compose up -d # 重启
docker compose down && docker compose up -d      # 重建（会丢容器内临时数据，但 ./data 卷保留）
docker image history campus-agent:latest          # 看镜像构建层，定位哪个改动导致变大
```

## 常见线上问题

| 症状 | 排查 |
|---|---|
| 容器启动即退出 | `docker logs` 看第一行——lifespan 第一段就校验 env |
| 接口 401 | `JWT_SECRET` 与 CampusHub 网关不一致 |
| 重启后「索引没了」 | 确认 `./data:/app/data` 卷挂了 |
| 上传 500 | MinerU key 超时？OSS bucket 是不是公共读？ |
| 附件问答答错 | `docker exec -it campus-agent python -c "from app.rag.doc_store import doc_store; print(len(list(doc_store.client.query('file_url != \"\"'))))"` |

## Dockerfile 要点

- `python:3.13-slim`，uv==0.13.6（与本地 lock 语义一致）
- 先拷 `pyproject.toml` + `uv.lock` 跑 `uv sync --frozen`，依赖层命中缓存
- 装了 `libgomp1`（faiss-cpu 依赖 OpenMP）和 `tzdata`（日志时间对齐）
- Debian 源换成腾讯云内网镜像（国内服务器 apt 不会卡）
- `UV_PROJECT_ENVIRONMENT=/app/.venv`，`uvicorn` 直接在 PATH 里
