"""配置：load_dotenv 读取项目根目录 .env（与 PythonProject 一致的用法）。

设计原则：
- 只做「读 + 校验」，不在这里连任何外部服务
- 必填项不给默认值：宁可启动时报错，也别静默连到别人的库上
- 校验时机交给 main.py 的 lifespan，import 阶段不 fail-fast（方便单测）

用法：
    from app.config import get_settings
    settings = get_settings()
"""
import logging
import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv

from app.common.exceptions import ConfigError

logger = logging.getLogger(__name__)

# 项目根目录：本文件位于 app/config.py，向上一级即项目根
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", verbose=False)

# 必填项：缺任意一个，服务就不该启动（值只从 .env 来，代码里不给默认值）
REQUIRED_KEYS: tuple[str, ...] = (
    "MYSQL_HOST",
    "MYSQL_USER",
    "MYSQL_PASSWORD",
    "MYSQL_DB_ACTIVITY",
    "MYSQL_DB_COMMUNITY",
    "MYSQL_DB_SECONDHAND",
    "MYSQL_DB_AGENT",
    "GLM_API_KEY",
    "GLM_BASE_URL",
    "GLM_CHAT_MODEL",
)


def _env(name: str, default: str = "") -> str:
    """读取环境变量，去空白；空串视为未设置。"""
    return (os.getenv(name) or "").strip() or default


def _env_int(name: str, default: int) -> int:
    """读取整数型环境变量。非法值直接报错，不做静默回落。"""
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"环境变量 {name} 必须是整数，当前值：{raw!r}") from None


class Settings:
    """全局配置。字段对应 .env 中的环境变量。"""

    def __init__(self) -> None:
        # ===== CampusHub MySQL（只读） =====
        self.mysql_host: str = _env("MYSQL_HOST")
        self.mysql_port: int = _env_int("MYSQL_PORT", 3306)
        self.mysql_user: str = _env("MYSQL_USER")
        self.mysql_password: str = _env("MYSQL_PASSWORD")
        self.mysql_db_activity: str = _env("MYSQL_DB_ACTIVITY")
        self.mysql_db_community: str = _env("MYSQL_DB_COMMUNITY")
        self.mysql_db_secondhand: str = _env("MYSQL_DB_SECONDHAND")
        # Agent 自己的库：聊天记录 / 上下文
        self.mysql_db_agent: str = _env("MYSQL_DB_AGENT")

        # ===== 对话模型（GLM，OpenAI 兼容协议） =====
        self.llm_api_key: str = _env("GLM_API_KEY")
        self.llm_base_url: str = _env("GLM_BASE_URL")
        # 变量名沿用桌面 env.txt 的 GLM_CHAT_MODEL，避免两处配置对不上
        self.llm_model: str = _env("GLM_CHAT_MODEL", "glm-5.3-flash")

        # ===== Embedding（腾讯 TokenHub，1024 维） =====
        self.embedding_api_key: str = _env("TENCENT_TOKENHUB_API_KEY")
        self.embedding_model: str = _env("TENCENT_TOKENHUB_MODEL_NAME")
        self.embedding_base_url: str = _env("TENCENT_TOKENHUB_BASE_URL")

        # ===== 向量库（Milvus） =====
        self.milvus_uri: str = _env("MILVUS_URI")

        # ===== 联网搜索（Tavily） =====
        self.tavily_api_key: str = _env("TAVILY_API_KEY")

        # ===== 视觉模型（GLM-4.6V：OCR / 看图，复用同一个 GLM key） =====
        self.vision_model: str = _env("GLM_VISION_MODEL", "glm-4.6v")

        # ===== 文件上传（阿里云 OSS） =====
        self.oss_access_key_id: str = _env("OSS_ACCESS_KEY_ID")
        self.oss_access_key_secret: str = _env("OSS_ACCESS_KEY_SECRET")
        self.oss_bucket: str = _env("OSS_BUCKET")
        self.oss_endpoint: str = _env("OSS_ENDPOINT", "oss-cn-beijing.aliyuncs.com")
        self.oss_region: str = _env("OSS_REGION", "cn-beijing")

        # ===== 用量配额 =====
        # 和 campus-hub 共用同一个 HS256 密钥，才能验出 userId
        self.jwt_secret: str = _env("JWT_SECRET", "")
        self.daily_token_quota: int = _env_int("DAILY_TOKEN_QUOTA", 100_000)
        # 每人每天最多上传几次（一次可以是一个文件）
        self.daily_upload_limit: int = _env_int("DAILY_UPLOAD_LIMIT", 5)

        # ===== 消息队列（RabbitMQ，用于增量索引） =====
        # campus-hub 的增删改发消息到这里，消费者收到后重算单条向量。
        # 不配也能跑（消费者不启动），只是索引只能靠手动跑 ingest 更新。
        self.rabbitmq_url: str = _env("RABBITMQ_URL")
        self.rabbitmq_index_queue: str = _env("RABBITMQ_INDEX_QUEUE", "campus.index.rebuild")

        # ===== 会话记忆（已改用 MySQL tb_chat_message，不再依赖 Redis） =====

        # ===== 服务自身参数（不涉及外部资源，给默认值） =====
        self.app_host: str = _env("APP_HOST", "127.0.0.1")
        self.app_port: int = _env_int("APP_PORT", 8000)

        # ===== CORS =====
        # 逗号分隔；默认 * 表示放开所有来源（仅适用于无凭证的公开读接口）
        self.cors_allow_origins: list[str] = [
            origin.strip()
            for origin in _env("CORS_ALLOW_ORIGINS", "*").split(",")
            if origin.strip()
        ]

    # ---------- 校验 ----------

    def validate_required(self) -> list[str]:
        """返回缺失的必填项名称，空列表表示通过。"""
        return [key for key in REQUIRED_KEYS if not _env(key)]

    def warn_suspicious(self) -> None:
        """可疑但不致命的配置，只告警不拦截。"""
        if "4v" in self.llm_model.lower():
            logger.warning(
                "GLM_CHAT_MODEL=%s 是视觉模型，不支持 function calling；"
                "做意图识别/工具调用建议换成 glm-4-flash 等文本模型",
                self.llm_model,
            )

    # ---------- 可用性探测（供后续模块判断某能力是否就绪） ----------

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key and self.llm_base_url and self.llm_model)

    @property
    def embedding_configured(self) -> bool:
        return bool(self.embedding_api_key and self.embedding_model and self.embedding_base_url)

    @property
    def oss_configured(self) -> bool:
        """OSS 四要素齐全才算能上传（bucket/endpoint 有默认值，主要看密钥）。"""
        return bool(
            self.oss_access_key_id
            and self.oss_access_key_secret
            and self.oss_bucket
            and self.oss_endpoint
        )

    @property
    def milvus_configured(self) -> bool:
        return bool(self.milvus_uri)

    @property
    def rabbitmq_configured(self) -> bool:
        """配了连接串才启动索引消费者。"""
        return bool(self.rabbitmq_url)

    # ---------- CORS ----------

    @property
    def cors_allow_credentials(self) -> bool:
        """是否允许跨域携带凭证。

        浏览器规范禁止 Access-Control-Allow-Origin: * 与 credentials 同时生效；
        Starlette 在两者并存时会把请求 Origin 原样回显，等于对全网开放带凭证访问。
        所以只在显式配置了来源白名单时才允许携带凭证。
        """
        return "*" not in self.cors_allow_origins

    # ---------- 连接串 ----------

    @property
    def mysql_url(self) -> str:
        """SQLAlchemy 连接串（不含库名，多库切换用）。

        driver 用 aiomysql（异步），与 FastAPI / Redis 的异步链路保持一致。
        账号密码做 URL 编码，防止密码里的 @ : / 等字符破坏连接串。
        """
        return (
            f"mysql+aiomysql://{quote_plus(self.mysql_user)}:{quote_plus(self.mysql_password)}"
            f"@{self.mysql_host}:{self.mysql_port}"
        )

    def db_url(self, db_name: str) -> str:
        return f"{self.mysql_url}/{db_name}?charset=utf8mb4"


@lru_cache
def get_settings() -> Settings:
    """全局单例（进程内只读一次 .env；改了 .env 需重启进程）。"""
    return Settings()
