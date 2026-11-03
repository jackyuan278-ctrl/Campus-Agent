"""应用入口：FastAPI 实例 + 生命周期 + 全局异常处理 + 路由挂载。"""
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import chat
from app.common.exceptions import AppError, ConfigError
from app.common.logger import setup_logging
from app.config import get_settings
from app.database import dispose_all

logger = logging.getLogger(__name__)

# 初始化日志配置（必须在其它模块打日志之前执行）
setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时校验配置，退出时释放连接池。"""
    settings = get_settings()

    missing = settings.validate_required()
    if missing:
        # 直接抛错让进程起不来：错配置跑起来的服务比不启动更难排查
        raise ConfigError(
            f"缺少必填配置：{', '.join(missing)}（请在项目根目录 .env 中补齐）"
        )
    settings.warn_suspicious()

    logger.info(
        "服务启动：host=%s port=%s mysql=%s llm_model=%s",
        settings.app_host,
        settings.app_port,
        settings.mysql_host,
        settings.llm_model,
    )
    try:
        yield
    finally:
        dispose_all()
        logger.info("服务已关闭")


app = FastAPI(
    title="CampusHub Agent API",
    description="校园智能助手（活动/社区 RAG + 二手商品工具调用）",
    version="0.1.0",
    lifespan=lifespan,
)

# 1. 配置跨域资源共享 (CORS)
#    allow_credentials 由配置推导：来源为 * 时强制 false，避免“通配 + 凭证”被原样回显
_cors = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors.cors_allow_origins,
    allow_credentials=_cors.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 2. 全局异常处理：业务异常转成统一 JSON，避免堆栈直接抛给前端
@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    logger.error("%s %s 失败：%s", request.method, request.url.path, exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.code, "message": exc.message},
    )


# 3. 挂载路由
app.include_router(chat.router, prefix="/assistant", tags=["助手"])


@app.get("/health", tags=["系统"])
def health():
    """存活探针：只证明进程活着，不查任何外部依赖。"""
    return {"status": "ok"}


if __name__ == "__main__":
    _settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=_settings.app_host,
        port=_settings.app_port,
        reload=True,
    )
