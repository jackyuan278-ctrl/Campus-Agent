"""附件上传接口。

为什么 campus-agent 自己做上传而不是复用 campus-hub 的：
    campus-hub 只有头像/封面/商品图三个上传口（各自绑业务），且不新增它的功能；
    agent 需要的是「任意附件」通道，返回的 URL 后续要喂给 MinerU 和视觉模型。

前端约定：表单字段名必须是 `file`，返回 `{"data": {"url", "key", "size"}}`。

顺序很重要：**预检额度 → 读文件 → 传 OSS → 成功后扣额度**。
反过来（先扣再传）会让上传失败的用户白白损失一次当天额度。

文档类附件（PDF/Word/txt/md）在 OSS 落地之后**同步解析**：
    用户上传时本来就在等，提问时期待秒回；反过来把 30~60 秒的解析放到
    对话里，用户就要对着空聊天框干等，而且失败会直接导致提问失败。
    解析失败不影响上传成功——文件已经在 OSS 里，退回去没有意义。
"""
import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, File, Request, UploadFile

from app.common.security import require_user_id
from app.config import get_settings
from app.llm.mineru import MineruError
from app.memory.quota import check_upload_quota, consume_upload
from app.rag.doc_ingest import index_document, is_document
from app.storage.oss import MAX_UPLOAD_BYTES, UploadError, upload_bytes

router = APIRouter()
logger = logging.getLogger(__name__)

# 同时最多解析几份文档。MinerU 那边有并发/额度限制，一次传 5 个文件时
# 不能让它们一起来；排队总比被对方限流好。
_PARSE_CONCURRENCY = 2
_parse_sem = None
_parse_sem_loop = None


def _parse_semaphore() -> asyncio.Semaphore:
    """按当前事件循环取信号量（模块级实例换 loop 会永久挂起，同 graph/vision）。"""
    global _parse_sem, _parse_sem_loop
    loop = asyncio.get_running_loop()
    if _parse_sem is None or _parse_sem_loop is not loop:
        _parse_sem = asyncio.Semaphore(_PARSE_CONCURRENCY)
        _parse_sem_loop = loop
    return _parse_sem


async def _parse_document(url: str, filename: str, content_type: str) -> dict:
    """解析并索引，任何失败都降级成状态字段，绝不让上传失败。"""
    if not get_settings().mineru_configured:
        return {"status": "skipped", "reason": "未配置文档解析"}
    if not is_document(content_type, filename):
        return {"status": "skipped", "reason": "不是文档类型，走图片识别"}

    try:
        async with _parse_semaphore():
            info = await index_document(url, filename)
    except MineruError as exc:
        # 这类异常是我们自己包装过的，message 可以直接给用户看
        logger.warning("文档解析失败：%s（%s）", filename, exc)
        return {"status": "failed", "reason": str(exc)}
    except Exception:
        logger.exception("文档解析出现未预期错误：%s", filename)
        return {"status": "failed", "reason": "解析服务异常，可稍后重新上传"}
    return {"status": "ok", **info}


@router.post("/upload", summary="上传附件（图片 / PDF / Word / 文本）")
async def upload(
    http_request: Request,
    file: Annotated[UploadFile, File(description="表单字段名 file")],
):
    """接收文件 → 传 OSS → （文档则解析入索引）→ 返回公网 URL。

    需要登录（token 无效一律 401），每人每天有上传次数上限——
    上传是花真金白银的 OSS 存储，不能敞开。
    """
    user_id = require_user_id(http_request)
    user_key = f"user:{user_id}"

    # 只读预检：额度满了就别白读一遍文件
    await check_upload_quota(user_key)

    # 多读 1 字节用来判断是否超限：直接把超大文件整个读进内存再拒绝，
    # 等于让一次恶意上传吃掉全部内存。
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError(f"文件不能超过 {MAX_UPLOAD_BYTES // 1024 // 1024}MB")

    filename = file.filename or "upload"
    content_type = file.content_type or ""
    result = await upload_bytes(data, filename=filename, content_type=content_type)

    # OSS 写成功之后才扣额度：上传失败不该消耗用户当天的次数
    await consume_upload(user_key)

    parsed = await _parse_document(result["url"], filename, content_type)
    return {"data": {**result, "parsed": parsed}}
