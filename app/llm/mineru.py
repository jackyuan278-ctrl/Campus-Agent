"""MinerU 云端解析：PDF / Word → markdown。

三步异步（MinerU 那边是任务队列，不是同步接口）：

    POST /api/v4/extract/task          {"url": "<公网可下载的地址>"}  → task_id
    GET  /api/v4/extract/task/{id}                                     → state / full_zip_url
    GET  full_zip_url                                                  → zip 里取 full.md

为什么是「给它 URL」而不是「把文件传给它」：
    精准解析 API 的设计就是 MinerU 自己来下载。所以 OSS 对象必须公网可读
    （当前 bucket 是 public-read-write，能直接读）。这也意味着我们不用把
    10MB 的文件再上传一遍。

为什么只取 full.md、不存 zip：
    zip 里还有 images/ 和一堆 json，对「让模型读到文字」没有帮助；图片引用
    抽出来之后在 markdown 里也是坏链（指向 zip 内部相对路径），所以统一剔除，
    并在正文开头说明「原文有 N 张图，未解析」，避免模型以为看懂了图表。
"""
import asyncio
import io
import logging
import re
import zipfile

import httpx

from app.common.exceptions import AppError
from app.config import get_settings

logger = logging.getLogger(__name__)

# 轮询间隔：MinerU 一份 20 页 PDF 通常 20~60 秒，3 秒一问既及时又不打爆对方
_POLL_INTERVAL = 3.0

# markdown 里的图片引用：![alt](images/xxx.jpg)
_IMAGE_REF = re.compile(r"!\[[^\]]*\]\([^)]*\)")


class MineruError(AppError):
    """文档解析失败。"""

    code = 2003
    status_code = 502
    message = "文档解析失败"


_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """懒加载：import 时不建连接池。"""
    global _client
    if _client is None:
        settings = get_settings()
        _client = httpx.AsyncClient(
            base_url=settings.mineru_base_url,
            headers={
                "Authorization": f"Bearer {settings.mineru_api_key}",
                "Content-Type": "application/json",
                "Accept": "*/*",
            },
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=10.0),
        )
    return _client


async def aclose() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _unwrap(payload: dict, action: str) -> dict:
    """拆掉 MinerU 的 {code, data, msg} 外壳，code != 0 一律算失败。"""
    if payload.get("code") not in (0, "0"):
        raise MineruError(f"{action}失败：{payload.get('msg') or payload.get('code')}")
    data = payload.get("data")
    return data if isinstance(data, dict) else {}


async def _submit(url: str) -> str:
    """提交解析任务，返回 task_id。"""
    settings = get_settings()
    resp = await _get_client().post(
        "/api/v4/extract/task",
        json={"url": url, "model_version": settings.mineru_model},
    )
    resp.raise_for_status()
    data = _unwrap(resp.json(), "提交解析任务")
    task_id = str(data.get("task_id") or "")
    if not task_id:
        raise MineruError("提交解析任务失败：没有返回 task_id")
    logger.info("MinerU 任务已提交：%s", task_id)
    return task_id


async def _wait(task_id: str) -> str:
    """轮询到 done，返回 full_zip_url。超时或 failed 抛 MineruError。"""
    settings = get_settings()
    deadline = asyncio.get_running_loop().time() + settings.mineru_timeout
    while True:
        resp = await _get_client().get(f"/api/v4/extract/task/{task_id}")
        resp.raise_for_status()
        data = _unwrap(resp.json(), "查询解析结果")
        state = str(data.get("state") or "")

        if state == "done":
            zip_url = str(data.get("full_zip_url") or "")
            if not zip_url:
                raise MineruError("解析完成但没有返回下载地址")
            return zip_url
        if state == "failed":
            raise MineruError(f"解析失败：{data.get('err_msg') or '未返回原因'}")

        if asyncio.get_running_loop().time() >= deadline:
            raise MineruError(f"解析超时（{settings.mineru_timeout}s，当前状态 {state or '未知'}）")
        await asyncio.sleep(_POLL_INTERVAL)


def _extract_markdown(raw: bytes) -> tuple[str, int]:
    """从 zip 里取出 full.md，剔掉图片引用。返回 (正文, 剔除的图片张数)。"""
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise MineruError("解析结果不是合法的 zip") from exc

    with archive:
        names = archive.namelist()
        # 优先根目录的 full.md；找不到就退而求其次取任意 .md（不同 model_version 布局略有差异）
        target = next((n for n in names if n.endswith("full.md")), None)
        if target is None:
            target = next((n for n in names if n.lower().endswith(".md")), None)
        if target is None:
            raise MineruError("解析结果里没有 markdown 文件")

        text = archive.read(target).decode("utf-8", errors="replace")

    images = len(_IMAGE_REF.findall(text))
    text = _IMAGE_REF.sub("", text)
    # 剔掉图片后会留下大量空行，压成最多一个空行，省 token
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, images


async def parse_document(url: str) -> tuple[str, int]:
    """把一份文档解析成 markdown。返回 (正文, 图片张数)。

    失败一律抛 MineruError——调用方（上传接口）应把它降级成「上传成功但没解析」，
    而不是让整次上传失败：文件已经在 OSS 里了，退回去没有意义。
    """
    settings = get_settings()
    if not settings.mineru_configured:
        raise MineruError("未配置 MINERU_API_KEY，无法解析文档")

    task_id = await _submit(url)
    zip_url = await _wait(task_id)

    resp = await _get_client().get(zip_url)
    resp.raise_for_status()
    text, images = _extract_markdown(resp.content)
    logger.info("文档解析完成：%d 字，%d 张图未解析", len(text), images)
    return text, images
