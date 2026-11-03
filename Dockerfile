# Dockerfile：独立部署 Agent 服务
# 依赖以 pyproject.toml + uv.lock 为唯一来源（不再使用 requirements.txt）
FROM python:3.13-slim

# 官方 uv 二进制（含 uvx）；版本钉死与本地 uv 0.13.6 一致，避免镜像里解析出不同 lock 语义
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# 先只拷依赖清单，让依赖层能命中缓存
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
