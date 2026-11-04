"""附件上传接口。

为什么 campus-agent 自己做上传而不是复用 campus-hub 的：
    campus-hub 只有头像/封面/商品图三个上传口（各自绑业务），且不允许改它的代码；
    agent 需要的是"任意附件"通道，返回的 URL 后续要喂给 MinerU 和视觉模型。

前端约定：表单字段名必须是 `file`，返回 `{"data": {"url", "key", "size"}}`。
"""
import logging
from typing import Annotated

from fastapi import APIRouter, File, Request, UploadFile

from app.common.security import require_user_id
from app.memory.quota import consume_upload
from app.storage.oss import MAX_UPLOAD_BYTES, UploadError, upload_bytes

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload", summary="上传附件（图片 / PDF / Word / 文本）")
async def upload(
    http_request: Request,
    file: Annotated[UploadFile, File(description="表单字段名 file")],
):
    """接收文件 → 传 OSS → 返回公网 URL。

    需要登录（token 无效一律 401），每人每天有上传次数上限——
    上传是花真金白银的 OSS 存储，不能敞开。
    """
    user_id = require_user_id(http_request)
    user_key = f"user:{user_id}"
    await consume_upload(user_key)
    """接收文件 → 传 OSS → 返回公网 URL。

    多读 1 字节用来判断是否超限：直接把超大文件整个读进内存再拒绝，
    等于让一次恶意上传吃掉全部内存。
    """
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError(f"文件不能超过 {MAX_UPLOAD_BYTES // 1024 // 1024}MB")

    result = await upload_bytes(
        data,
        filename=file.filename or "upload",
        content_type=file.content_type or "",
    )
    return {"data": result}
