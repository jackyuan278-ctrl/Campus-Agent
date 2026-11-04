"""阿里云 OSS 上传封装。

几个约束来自现实：
- SDK（alibabacloud-oss-v2）是同步的，这里用 `asyncio.to_thread` 包一层，
  否则一次上传（几百毫秒到几秒）会把事件循环堵住，同进程其它请求全部排队。
- 返回的公网 URL 拼法与 campus-hub 的 OssUtil 保持一致：
  `https://{bucket}.{endpoint}/{key}`，这样前端两边的图片地址格式统一。
- 白名单/体积在后端**再拦一次**：前端限制只是体验，绕过它太容易。
"""
import asyncio
import logging
import uuid
from datetime import datetime

import alibabacloud_oss_v2 as oss

from app.common.exceptions import AppError
from app.config import get_settings

logger = logging.getLogger(__name__)

OBJECT_PREFIX = "campushub/agent/"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10MB，与前端 ATTACHMENT_RULES 对齐

# 允许的 MIME → 后缀。key 是 content-type，value 是保底后缀（文件名没后缀时用）
ALLOWED_TYPES: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "application/pdf": ".pdf",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "text/plain": ".txt",
    "text/markdown": ".md",
}

_client: oss.Client | None = None


class UploadError(AppError):
    """上传失败（参数非法或 OSS 报错）。"""

    code = 3002
    status_code = 400
    message = "文件上传失败"


def _get_client() -> oss.Client:
    """懒加载：import 时不读配置、不建客户端。"""
    global _client
    if _client is None:
        settings = get_settings()
        config = oss.config.load_default()
        config.region = settings.oss_region
        config.endpoint = settings.oss_endpoint
        config.credentials_provider = oss.credentials.StaticCredentialsProvider(
            access_key_id=settings.oss_access_key_id,
            access_key_secret=settings.oss_access_key_secret,
        )
        _client = oss.Client(config)
    return _client


def _resolve_suffix(filename: str, content_type: str) -> str:
    """优先用文件名的后缀，拿不到再按 content-type 兜底。"""
    dot = filename.rfind(".")
    if 0 <= dot < len(filename) - 1:
        suffix = filename[dot:].lower()
        if suffix in set(ALLOWED_TYPES.values()):
            return suffix
    return ALLOWED_TYPES[content_type]


def _build_url(key: str) -> str:
    settings = get_settings()
    endpoint = settings.oss_endpoint.strip().removeprefix("https://").removeprefix("http://").rstrip("/")
    return f"https://{settings.oss_bucket}.{endpoint}/{key}"


async def upload_bytes(data: bytes, filename: str, content_type: str) -> dict[str, str | int]:
    """把内存里的字节传上 OSS，返回 {url, key, size}。

    抛 UploadError 而不是裸异常：OSS 的异常信息里可能带 endpoint、bucket
    甚至签名片段，不该原样返给前端。
    """
    settings = get_settings()
    if not settings.oss_configured:
        raise UploadError("文件上传未启用：请在 .env 配置 OSS_ACCESS_KEY_ID / OSS_ACCESS_KEY_SECRET")
    if not data:
        raise UploadError("文件内容为空")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError(f"文件不能超过 {MAX_UPLOAD_BYTES // 1024 // 1024}MB")

    normalized_type = (content_type or "").split(";")[0].strip().lower()
    if normalized_type not in ALLOWED_TYPES:
        # 微信 / 部分客户端上传的图片 MIME 为空，靠扩展名兜底
        suffix = filename[filename.rfind("."):].lower() if "." in filename else ""
        guessed = next((k for k, v in ALLOWED_TYPES.items() if v == suffix), "")
        if guessed:
            normalized_type = guessed
        else:
            raise UploadError("不支持的文件类型，仅支持图片 / PDF / Word / 文本")

    key = f"{OBJECT_PREFIX}{datetime.now():%Y/%m}/{uuid.uuid4().hex}{_resolve_suffix(filename, normalized_type)}"
    request = oss.PutObjectRequest(
        bucket=settings.oss_bucket,
        key=key,
        body=data,
        content_type=normalized_type,
    )
    try:
        await asyncio.to_thread(_get_client().put_object, request)
    except Exception as exc:
        logger.exception("OSS 上传失败：key=%s", key)
        raise UploadError("文件上传失败，请稍后重试") from exc

    url = _build_url(key)
    logger.info("上传成功：%s（%d 字节）", url, len(data))
    return {"url": url, "key": key, "size": len(data)}
