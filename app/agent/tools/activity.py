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


def _time_filter(time_range: str, now: datetime):
    """构造候选谓词：活动没结束 + （可选）开始时间落在窗口内。

    两条规则的优先级不同：
        「已结束」是硬过滤，任何档位都生效——推荐一个办完的活动只会让用户觉得答错；
        「时间窗」只在模型显式填了档位时生效。
    """

    def _where(doc: dict[str, Any]) -> bool:
        # 问答帖没有活动时间窗，不参与时间过滤（它们本来就不是「这周」的主题，
        # 排不上来是因为相似度不够，不是被误杀）
        if doc.get("source") != "activity":
            return True

        end_time = clock.parse_dt(doc.get("end_time"))
        if end_time is not None and end_time < now:
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


@tool
async def search_campus_activities(
    query: str,
    time_range: str = "all",
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """检索校园活动与社区帖子（向量检索）。

    用户问「有什么活动」「讲座」「比赛」「社团招新」「帖子讨论」时使用。

    time_range 只能填这四个值之一：
      all   不限时间（默认）
      today 今天
      week  本周剩余——用户说「这周」「最近」「周末」「这几天」时填这个
      month 未来一个月——用户说「这个月」「接下来」时填这个

    已结束的活动不会返回。返回空列表代表确实没有，照实说即可，不要编造。
    """
    if time_range not in clock.TIME_RANGES:
        # 模型偶尔会自己发明档位（比如 "tomorrow"）。降级成 all 而不是抛异常：
        # 抛异常会浪费一轮工具调用，而用户看到的只是「没按时间筛」，还能救
        logger.warning("模型传入未知 time_range=%r，按 all 处理", time_range)
        time_range = "all"

    now = clock.now_cn()
    results = await retrieve(query, top_k=top_k, where=_time_filter(time_range, now))
    logger.info(
        "活动检索：time_range=%s 命中 %d 条（now=%s）",
        time_range, len(results), clock.describe_now(now),
    )
    return results


CAMPUS_SEARCH_TOOLS = [search_campus_activities]
