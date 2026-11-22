# 部署运维

> 本文件描述 Campus-Agent 的容器化部署架构、依赖外部服务、首次部署流程和日常运维操作。

---

## 1. 部署架构总览

```
┌─────────────────────────────────────────────────────────────┐
│ 腾讯云 ECS（容器名：campus-agent）                            │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Docker 容器                                         │    │
│  │  ├─ uvicorn (port 8000)                              │    │
│  │  ├─ LangGraph Agent                                  │    │
│  │  ├─ Milvus Lite (进程内，本地文件 milvus.db)          │    │
│  │  └─ RabbitMQ Consumer (可选，配了 RABBITMQ_URL 才起)  │    │
│  └─────────────────────────────────────────────────────┘    │
│          │                                                  │
│          │ volumes: ./data:/app/data                        │
│          │  ├── vectors.npy + docs.json（校园库）            │
│          │  └── milvus.db（附件库）                           │
└──────────┼──────────────────────────────────────────────────┘
           │ MySQL (aiomysql)
           ▼
┌───────────────────────────────────────────┐
│  MySQL 实例（与 campus-hub 共用）            │
│                                             │
│  只读业务库（agent 只查不写）：               │
│    ├─ MYSQL_DB_ACTIVITY   tb_activity       │
│    ├─ MYSQL_DB_COMMUNITY  tb_question       │
│    └─ MYSQL_DB_SECONDHAND  tb_item          │
│                                             │
│  读写业务库（agent 自己的库）：               │
│    └─ MYSQL_DB_AGENT                        │
│         ├─ tb_chat_message                  │
│         ├─ tb_upload_record                 │
│         ├─ tb_user_quota                    │
│         └─ tb_thread_ownership              │
└───────────────────────────────────────────┘

外部依赖：
  ├─ 阿里云 OSS（文件上传，bucket 必须公共读）
  ├─ GLM API（对话模型 / 视觉模型 / Embedding）
  ├─ MinerU（可选，PDF 解析回退）
  ├─ Tavily（可选，联网搜索，仅 casual 档用）
  └─ RabbitMQ（可选，增量索引消费者）
```

---

## 2. Dockerfile 与 docker-compose.yml

### Dockerfile 关键设计

位置：仓库根目录 `Dockerfile`

| 层 | 说明 |
|----|------|
| 基础镜像 | `python:3.13-slim` |
| Debian 源替换 | 默认 `deb.debian.org` 在国内服务器超时 → 换成腾讯云内网源 `mirrors.tencentyun.com` |
| 系统依赖 | `libgomp1`（faiss-cpu 链接了 OpenMP 运行时）+ `tzdata`（让日志时间对齐 Asia/Shanghai） |
| uv 安装 | `pip install uv==0.13.6`（与本地 lock 版本对齐） |
| 依赖安装 | `uv sync --frozen --no-dev --no-install-project`（先装依赖层，后续 app 代码变动不会让依赖缓存失效） |
| 工作目录 | `/app` |
| 卷目录 | `mkdir -p /app/data/rag`（Milvus Lite 不会自动建目录，首次传附件才报错） |
| 默认入口 | `uvicorn app.main:app --host 0.0.0.0 --port 8000` |

### docker-compose.yml 关键配置

位置：仓库根目录 `docker-compose.yml`

```yaml
services:
  agent:
    container_name: campus-agent
    build: .
    image: campus-agent:latest
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data           # 校园库 + 附件库都在这，不挂卷容器重建即丢
    restart: unless-stopped
    env_file: .env
    environment:
      - TZ=Asia/Shanghai
    healthcheck:
      test: ["CMD", "python", "-c",
             "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 20s
    logging:
      driver: json-file
      options:
        max-size: "20m"
        max-file: "3"
```

### 为什么 compose 里没有 MySQL / Redis / Milvus 容器？

- **MySQL**：与 campus-hub 前端服务共用同一套 MySQL 实例，agent 只是多开了一个库（`mysql_db_agent`）存放自己的 4 张业务表
- **Milvus Lite**：进程内向量库，不需要独立服务，就是一个本地文件 `milvus.db`
- **Redis**：会话持久化已改用 MySQL `tb_chat_message`，不再需要 Redis

---

## 3. 首次部署流程

```bash
# 1. 准备环境变量
cp .env.example .env
# 编辑 .env，填入生产环境的 MySQL 连接、GLM API key、OSS 配置等

# 2. 构建镜像
docker compose build

# 3. 重建校园索引（这步**必须**，不做就是空检索）
docker compose run --rm agent python -m app.rag.ingest --limit 0

# 4. 启动服务
docker compose up -d

# 5. 验证
curl http://localhost:8000/health
# 预期返回 {"status": "ok", ...}
```

### 必填环境变量（缺一不可，服务启动时会拒绝启动）

```
MYSQL_HOST
MYSQL_USER
MYSQL_PASSWORD
MYSQL_DB_ACTIVITY     # 只读，校园活动库
MYSQL_DB_COMMUNITY    # 只读，校园社区库
MYSQL_DB_SECONDHAND   # 只读，二手市场库
MYSQL_DB_AGENT        # 读写，agent 自己的业务库
GLM_API_KEY
GLM_BASE_URL
GLM_CHAT_MODEL
```

### OSS Bucket 公共读要求

上传到 OSS 的文件**必须公共读**。原因：
- `/assistant/upload` 返回的 URL 是 OSS 公网地址
- 视觉模型（glm-4.6v）和 MinerU 云服务需要**公网可访问**图片/PDF 才能下载解析
- 如果 bucket 是私有，图片上传成功但视觉模型拿不到 → OCR 直接失败

Bucket policy 示例：
```json
{
  "Version": "1",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": "*",
      "Action": ["oss:GetObject"],
      "Resource": ["acs:oss:*:*:<your-bucket>/*"]
    }
  ]
}
```

### MySQL 库表（agent 自己的）

仓库里**没有建表 SQL**，部署脚本在 `scripts/deploy/` 下（不在 git 上）。首次部署需要让 DBA 执行这些脚本，创建 4 张表：

| 表 | 说明 |
|----|------|
| `tb_chat_message` | 原始对话消息（前端 UI 展示） |
| `tb_upload_record` | 附件上传记录 + vision_tokens |
| `tb_user_quota` | 每日 token 用量 + 剩余配额 |
| `tb_thread_ownership` | thread_id → user_id 映射 |

---

## 4. 日常运维

### 重建校园索引（不停机也行，但强烈建议停服务）

```bash
docker compose down
docker compose run --rm agent python -m app.rag.ingest --limit 0 --full   # --full 强制全量重算
docker compose up -d
```

不停机直接跑会触发 Milvus Lite 文件锁警告（见 `docs/INDEXING.md` 第 4 节），虽然校园库重建本身不碰 Milvus Lite，但安全起见统一停服务。

### 查看日志

```bash
# 实时日志
docker compose logs -f agent

# 最近 200 行
docker compose logs --tail 200 agent

# 只看错误
docker compose logs agent 2>&1 | grep ERROR
```

日志轮转已配置：每个文件 20MB，最多保留 3 个。

### 进入容器调试

```bash
docker compose exec agent bash

# 容器内跑单条 ingest 验证
uv run python -m app.rag.ingest --limit 50
```

### 重启服务

```bash
docker compose restart agent
```

### 扩容 / 缩容

当前 compose 只起一个实例。如需多实例（水平扩容），注意：
1. **checkpoint 共享**：多实例指向同一个 MySQL（已满足）
2. **索引文件共享**：`./data` 是宿主机挂载的本地目录，多容器要共享同一个卷——考虑改成 NFS 或对象存储挂载
3. **Milvus Lite 不支持多进程读写**：多实例部署时需要单独起 Milvus Standalone 替换 Milvus Lite
4. **thread_lock 进程内锁**：多实例间无法互斥同一会话，可能出现 checkpointer last-write-wins 问题

当前推荐**单实例跑**。

---

## 5. 回滚流程

```bash
# 1. 拉回旧镜像 / 重建旧版本
git checkout <旧 commit>
docker compose build

# 2. 停服务、切镜像、起服务
docker compose down
docker compose up -d

# 3. 如果旧代码依赖的 MySQL schema 版本不同，需要手动迁移
# 脚本在 scripts/deploy/migration/
```

回滚相关脚本和 schema 迁移工具均位于 `scripts/deploy/` 目录（不在 git 仓库内，部署时从 CI 或 DBA 处获取）。

### 特别注意：索引文件回滚

校园库的 `vectors.npy` / `docs.json` 和 Milvus Lite 的 `milvus.db` 是**与代码版本耦合**的：
- 如果 embedding 模型换过，旧索引的向量维度可能不匹配
- 旧代码可能不认识新索引的 fingerprint 字段格式

回滚后**建议立即重建索引**：
```bash
docker compose run --rm agent python -m app.rag.ingest --limit 0 --full
```

---

## 6. 常见故障排查

| 现象 | 可能原因 | 排查 |
|------|---------|------|
| 启动时报「ModuleNotFoundError」 | pyproject.toml 里有新依赖但镜像还是旧的 | `docker compose build --no-cache` 强制重建 |
| `/health` 返回 503 | agent_runtime 未初始化 | 看容器日志有没有「Agent 运行时未初始化」 |
| 检索结果为空 | 校园库未重建，或 Milvus Lite 锁死 | `docker compose run --rm agent python -m app.rag.ingest --limit 50` 验证重建 |
| 附件上传后视觉识别失败 | OSS bucket 不是公共读 | 浏览器打开返回的 URL 看是否需要登录 |
| 索引重建时报 `ImportError: libgomp.so.1` | 容器内缺少 OpenMP 运行时 | Dockerfile 里的 apt-get install libgomp1 是否正确执行 |
| RabbitMQ consumer 日志持续报错 | 队列连接断了或队列不存在 | `docker compose logs agent 2>&1 | grep consumer` 看详细错误 |
| 回答被截断或重复 | LangGraph checkpoint 被并发覆盖 | 检查 thread_lock 是否生效（单实例不会有这个问题） |

---

## 7. 环境变量速查表

| 变量 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `MYSQL_HOST` | ✅ | — | MySQL 主机 |
| `MYSQL_PORT` | ❌ | 3306 | MySQL 端口 |
| `MYSQL_USER` | ✅ | — | MySQL 用户 |
| `MYSQL_PASSWORD` | ✅ | — | MySQL 密码 |
| `MYSQL_DB_ACTIVITY` | ✅ | — | 活动库（只读） |
| `MYSQL_DB_COMMUNITY` | ✅ | — | 社区库（只读） |
| `MYSQL_DB_SECONDHAND` | ✅ | — | 二手库（只读） |
| `MYSQL_DB_AGENT` | ✅ | — | agent 自库（读写） |
| `GLM_API_KEY` | ✅ | — | 智谱 GLM API key |
| `GLM_BASE_URL` | ✅ | — | 智谱 API base URL |
| `GLM_CHAT_MODEL` | ✅ | — | 对话模型名 |
| `GLM_VISION_MODEL` | ❌ | glm-4.6v | 视觉模型名 |
| `TENCENT_TOKENHUB_API_KEY` | ❌ | — | Embedding API key（不配则校园库无法重建） |
| `MILVUS_LITE_PATH` | ❌ | `data/rag/milvus.db` | 附件库路径 |
| `OSS_ACCESS_KEY_ID` | ❌ | — | OSS 上传用 |
| `OSS_ACCESS_KEY_SECRET` | ❌ | — | OSS 上传用 |
| `OSS_BUCKET` | ❌ | — | OSS bucket（必须公共读） |
| `OSS_ENDPOINT` | ❌ | oss-cn-beijing.aliyuncs.com | OSS endpoint |
| `MINERU_API_KEY` | ❌ | — | PDF 解析回退（不配则长 PDF 上传仅当文本层抽不到时报错） |
| `TAVILY_API_KEY` | ❌ | — | 联网搜索（仅 casual 档） |
| `JWT_SECRET` | ❌ | — | 与 campus-hub 共用的 HS256 密钥 |
| `DAILY_TOKEN_QUOTA` | ❌ | 100000 | 每日 token 上限 |
| `DAILY_UPLOAD_LIMIT` | ❌ | 5 | 每人每天最多上传次数 |
| `RABBITMQ_URL` | ❌ | — | 增量索引消费者（不配则消费者不启动） |
| `TZ` | ❌ | Asia/Shanghai | 日志时间对齐（需镜像装了 tzdata） |
