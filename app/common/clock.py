"""统一时间源：所有「现在几点」都必须走这里。

为什么单独一个模块：
    三个时间源，只有 agent 是错的。实测（2026-11-07）：
        宿主机  17:11 Asia/Shanghai
        MySQL   17:11 CST（@@system_time_zone=CST）
        agent 容器 09:11 UTC（TZ 为空）
    Python 里直接 `datetime.now()` 拿到的是 UTC。它不会报错，也不会看起来异常，
    但 CST 00:00~08:00 这八个小时里「今天」是**前一天**——查出来的活动看着都正常，
    只是少了当天那批、多了昨天那批。这类错只能靠不写 `datetime.now()` 来避免。

为什么不用 datetime.utcnow() + 8：那是"看起来对"的做法，一次忘记加就前功尽弃，
    而且没法在代码里一眼看出这个地方是本地时区还是 UTC。

为什么带 fallback：python:slim 镜像里没有 /usr/share/zoneinfo，ZoneInfo 会抛
    ZoneInfoNotFoundError。中国 1991 年后不实行夏令时，固定 +08:00 与 Asia/Shanghai
    完全等价，所以退化成固定偏移是安全的，不需要给镜像装 tzdata。
"""
import logging
from datetime import datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

try:
    TZ_CN = ZoneInfo("Asia/Shanghai")
except Exception:  # noqa: BLE001 - 缺 tzdata 时退化成固定偏移，不影响正确性
    logger.warning("缺少 Asia/Shanghai 时区数据，退化为固定 +08:00（中国无夏令时，等价）")
    TZ_CN = timezone(timedelta(hours=8), "CST")

_WEEKDAYS = "一二三四五六日"

# 支持的时间窗。工具参数只认这几个值，其余一律按 all 处理。
TIME_RANGES = ("all", "today", "week", "month")


def now_cn() -> datetime:
    """当前时间（带 +08:00 时区信息）。"""
    return datetime.now(TZ_CN)


def today_start(now: datetime | None = None) -> datetime:
    """今天 00:00（带时区）。"""
    base = now or now_cn()
    return datetime.combine(base.date(), time.min, tzinfo=TZ_CN)


def describe_now(now: datetime | None = None) -> str:
    """给人看/给模型看的一句话时间，如「2026-11-07 周六 17:15」。

    模型不知道今天几号（训练截止在前），所以每轮都要把这句话注入 system prompt。
    """
    base = now or now_cn()
    return f"{base:%Y-%m-%d} 周{_WEEKDAYS[base.weekday()]} {base:%H:%M}"


def range_of(time_range: str, now: datetime | None = None) -> tuple[datetime | None, datetime | None]:
    """把档位名换算成 [start, end) 左闭右开区间；None 表示该侧不设限。

    语义选择（决定了用户看到什么）：
        today  今天 00:00 → 明天 00:00
        week   今天 00:00 → 下周一 00:00   ← 「这周」= 本周剩余，不是「过去 7 天」
        month  今天 00:00 → 今天 +30 天
        all    不设限

    为什么 week 的下界是「今天」而不是「本周一」：用户问「这周有什么活动」时想看的是
    还能去的，把周一到周三已经办完的捞回来只是噪音。已结束的活动另有 end_time 过滤。
    """
    base = now or now_cn()
    start = today_start(base)
    tomorrow = start + timedelta(days=1)

    if time_range == "today":
        return start, tomorrow
    if time_range == "week":
        # weekday(): 周一=0 → 到下周一还有 (7 - weekday) 天
        return start, start + timedelta(days=7 - base.weekday())
    if time_range == "month":
        return start, start + timedelta(days=30)
    return None, None


def parse_dt(value: Any) -> datetime | None:
    """把索引里的时间字段解析成带时区的 datetime；解析不了返回 None。

    索引里存的是 `isoformat(sep=" ")`（不带时区，语义就是北京时间），
    所以这里补上 TZ_CN，才能和 now_cn() 比较——不带时区的 datetime 和
    带时区的 datetime 直接比较会抛 TypeError。
    """
    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=TZ_CN)
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        logger.debug("时间字段解析失败：%r", value)
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=TZ_CN)
