"""活动 / 社区检索工具：把 RAG 检索包成模型可调用的 function calling 工具。

底层仍是 `app.rag.retriever.retrieve`（numpy 余弦 + 去重），这里只加两层：
1. 「工具说明书」——让模型自己决定什么时候搜、搜什么时间窗；
2. **结构化过滤**——时间窗和「已结束」由代码判断，不交给模型。

为什么时间窗让模型填、而不是正则解析用户这句话：
    「这周」「最近」「周末」这类词，正则一旦猜错是**静默错误**（查出一批上个月的活动，
    看起来完全正常）；而模型填参数，错也错在日志里，能查。function calling 本来就是
    拿来做参数抽取的，没必要为了省一次参数填写去自己写中文时间正则。
"""
import logging
from datetime import datetime
from typing import Any

from langchain_core.tools import tool

from app.common import clock
from app.rag.retriever import retrieve

logger = logging.getLogger(__name__)


def _time_filter(time_range: str, now: datetime, location: str | None = None):
    """构造候选谓词：活动没结束 + （可选）开始时间落在窗口内 + （可选）地点匹配。

    两条规则的优先级不同：
        「已结束」是硬过滤，任何档位都生效——推荐一个办完的活动只会让用户觉得答错；
        「时间窗 / 地点」只在模型显式填了参数时生效。
    """

    def _where(doc: dict[str, Any]) -> bool:
        # 问答帖没有活动时间窗，不参与时间过滤（它们本来就不是「这周」的主题，
        # 排不上来是因为相似度不够，不是被误杀）
        if doc.get("source") != "activity":
            return True

        end_time = clock.parse_dt(doc.get("end_time"))
        if end_time is not None and end_time < now:
            return False

        if location:
            # 地点没有独立字段，只在 extra 里（"地点：图书馆一楼展厅｜开始时间：…"）。
            # 用包含匹配而不是相等：用户说「图书馆」，库里存的是「图书馆一楼展厅」。
            if location not in str(doc.get("extra") or ""):
                return False

        start, end = clock.range_of(time_range, now)
        if start is None and end is None:
            return True

        start_time = clock.parse_dt(doc.get("start_time"))
        if start_time is None:
            # 时间字段缺失时放行：宁可多给一条让模型自己判断，也不要静默丢掉。
            # 静默丢数据的后果比多一条噪音严重得多——用户会得到「没有活动」这个结论。
            return True
        return start <= start_time < end

    return _where


# 摘要截断长度。`text` 是给向量用的拼接串，`content` 最多 500 字；
# 全给模型 5 条就 3000 字，其中一半信息在 prompt 里从没被用到。
_SUMMARY_CHARS = 200


def _trim(doc: dict[str, Any]) -> dict[str, Any]:
    """只回模型真正要用的字段（去掉 text / fingerprint / content 全文）。"""
    return {
        "source": doc.get("source"),
        "id": doc.get("id"),
        "title": doc.get("title"),
        "summary": str(doc.get("content") or "")[:_SUMMARY_CHARS],
        "extra": doc.get("extra"),              # 地点 + 开始时间（人可读）
        "start_time": doc.get("start_time"),
        "signup_end_time": doc.get("signup_end_time"),
        "score": doc.get("score"),
    }


@tool
async def search_campus_activities(
    query: str,
    time_range: str = "all",
    location: str = "",
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """检索校园活动与社区帖子（稠密向量 + 关键词两路召回后融合）。

    用户问「有什么活动」「讲座」「比赛」「社团招新」「帖子讨论」时使用。

    time_range 只能填这四个值之一：
      all   不限时间（默认）
      today 今天
      week  本周剩余——用户说「这周」「最近」「周末」「这几天」时填这个
      month 未来一个月——用户说「这个月」「接下来」时填这个

    location 可选：用户提到地点时填（如「图书馆」「体育馆」），不填就是不限。

    已结束的活动不会返回。返回空列表代表确实没有，照实说即可，不要编造。
    """
    if time_range not in clock.TIME_RANGES:
        # 模型偶尔会自己发明档位（比如 "tomorrow"）。降级成 all 而不是抛异常：
        # 抛异常会浪费一轮工具调用，而用户看到的只是「没按时间筛」，还能救
        logger.warning("模型传入未知 time_range=%r，按 all 处理", time_range)
        time_range = "all"

    now = clock.now_cn()
    hits = await retrieve(
        query,
        top_k=top_k,
        where=_time_filter(time_range, now, location.strip() or None),
    )
    logger.info(
        "活动检索：time_range=%s location=%s 命中 %d 条（now=%s）",
        time_range, location or "-", len(hits), clock.describe_now(now),
    )
    return [_trim(doc) for doc in hits]


CAMPUS_SEARCH_TOOLS = [search_campus_activities]
