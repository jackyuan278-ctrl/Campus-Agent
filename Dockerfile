# Dockerfile：独立部署 Agent 服务
# 依赖以 pyproject.toml + uv.lock 为唯一来源（不再使用 requirements.txt）
FROM python:3.13-slim

# 包索引源：默认官方源，国内构建用 --build-arg 覆盖（compose 里已配清华源）
ARG UV_DEFAULT_INDEX=https://pypi.org/simple
ENV UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX}

# faiss-cpu（Milvus Lite 的底层依赖）链接了 OpenMP 运行时 libgomp.so.1，
# 而 python:slim 镜像里不带 GCC 运行时——不装的话 `import faiss` 直接
# ImportError: libgomp.so.1: cannot open shared object file。
#
# 同时装 tzdata：镜像默认没有 /usr/share/zoneinfo，`TZ=Asia/Shanghai` 会被静默
# 忽略、date 仍显示 UTC。代码里已经用显式时区（app.common.clock），这里只是让
# 日志时间也对齐——排查跨天问题时差 8 小时很容易把人带偏。
#
# 先把 Debian 源换成国内镜像：默认的 deb.debian.org 在国内服务器上会卡住
# （实测 apt-get update 三分钟没有任何进展）。腾讯云内网源最快且不走公网流量，
# 非腾讯云机器可改成 mirrors.cloud.tencent.com。
RUN set -eux; \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
        sed -i 's|deb.debian.org|mirrors.tencentyun.com|g; s|security.debian.org|mirrors.tencentyun.com|g' \
            /etc/apt/sources.list.d/debian.sources; \
    fi; \
    if [ -f /etc/apt/sources.list ]; then \
        sed -i 's|deb.debian.org|mirrors.tencentyun.com|g; s|security.debian.org|mirrors.tencentyun.com|g' \
            /etc/apt/sources.list; \
    fi; \
    apt-get update; \
    apt-get install -y --no-install-recommends libgomp1 tzdata; \
    rm -rf /var/lib/apt/lists/*

ENV TZ=Asia/Shanghai

# uv 版本与本地 0.13.6 对齐，保证 lock 语义一致。
# 不再用 ghcr.io 的二进制镜像：国内机器拉 ghcr 经常卡死（服务器上实测卡在
# 这一步十几分钟零进展），改成 pip 从同一个索引源装同版本，行为一样。
RUN pip install --no-cache-dir "uv==0.13.6"

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# 先只拷依赖清单，让依赖层能命中缓存
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app

# 附件向量库要写本地文件，而 Milvus Lite 不会自动建目录：
# 目录不存在时它会在「第一次有人传文档」才报错。这里先建好，
# 同时给 docker-compose 的卷挂载提供落点。
RUN mkdir -p /app/data/rag

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
