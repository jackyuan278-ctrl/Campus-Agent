"""用户身份解析：验 campus-hub（Java 端）签发的 JWT。

对齐点（来自 campus-common 的 JwtTool / campus-gateway 的 AuthGlobalFilter）：
- 算法 HS256，密钥是 `campus.jwt.secret`（默认 ${JWT_SECRET:campushub-jwt-secret-placeholder-32bytes-min}）
- 载荷里 `sub` 是 userId（数字，解析后按字符串用）
- Authorization 头里 "Bearer " 前缀可有可无，大小写不敏感

为什么在这里验签而不是让 campus-hub 开个接口：
    多一次 HTTP 往返只为了拿一个 userId，不划算；密钥是共享的，本地验签更直接。
    代价是两边密钥必须一致——不一致时所有 token 都会验签失败，所以下面会打 WARNING。
"""
import logging

import jwt

from app.config import get_settings

logger = logging.getLogger(__name__)

_BEARER_PREFIX = "bearer "


def parse_user_id(authorization: str | None) -> str | None:
    """从 Authorization 头取 userId；任何异常都返回 None（当作匿名）。"""
    if not authorization:
        return None

    token = authorization.strip()
    if token.lower().startswith(_BEARER_PREFIX):
        token = token[len(_BEARER_PREFIX):].strip()
    if not token:
        return None

    secret = get_settings().jwt_secret
    if not secret:
        logger.debug("未配置 JWT_SECRET，跳过验签（全部按匿名处理）")
        return None

    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        logger.info("JWT 已过期，按匿名处理")
        return None
    except jwt.PyJWTError as exc:
        # 最常见的两种：密钥和 campus-hub 不一致、token 被截断
        logger.warning(
            "JWT 验签失败（%s）。若所有请求都这样，检查 .env 的 JWT_SECRET 是否与 campus-hub 一致",
            type(exc).__name__,
        )
        return None

    subject = payload.get("sub")
    return str(subject) if subject else None


def resolve_user_key(authorization: str | None, thread_id: str) -> str:
    """配额统计用的用户标识。

    有合法 token → user:<userId>（跨线程、跨设备都算一个人）
    否则 → anon:<threadId>（仅用于本地调试；线上接口都要求登录）
    """
    user_id = parse_user_id(authorization)
    if user_id:
        return f"user:{user_id}"
    return f"anon:{thread_id}"


def require_user_id(request) -> str:
    """要求登录：解析不出 userId 就抛 401（对齐 campus-hub 网关的行为）。

    入参是 FastAPI 的 Request，直接从 Authorization 头取，避免每个路由都重复写一段。
    """
    from app.common.exceptions import UnauthorizedError

    user_id = parse_user_id(request.headers.get("authorization"))
    if not user_id:
        raise UnauthorizedError()
    return user_id
