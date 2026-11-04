"""Agent 工具层：CampusHub 查询能力封装。

表结构参考（campushub_secondhand.tb_item）：
    id, seller_id, name, price, image, category_id,
    condition_desc, description, trade_location, status,
    create_time, update_time
（status: 1在售 0下架 3已售）

约定：
- 全部异步，Engine 来自 app.database（aiomysql）
- 返回值必须能 JSON 序列化：Decimal -> float，datetime -> str
- 排序字段只走白名单，绝不把外部输入拼进 SQL
"""
from typing import Any

from langchain_core.tools import tool
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.database import get_secondhand_engine

# 只暴露「在售」商品：1在售 0下架 3已售
STATUS_ON_SALE = 1

# 单页上限，防止一次拉全表
MAX_PAGE_SIZE = 50

# 排序白名单：ORDER BY 不支持参数绑定，只能这么防注入
SORT_MAP: dict[str, str] = {
    "price_asc": "price ASC",
    "price_desc": "price DESC",
    "time_desc": "create_time DESC",
}

# 需要转成字符串的时间字段（datetime 不能直接进 JSON / prompt）
_TIME_FIELDS = ("create_time", "update_time")
# 需要转成 float 的金额字段（MySQL DECIMAL 取出来是 Decimal）
_FLOAT_FIELDS = ("price",)


def _normalize_paging(page: int, page_size: int) -> tuple[int, int]:
    """收敛分页参数：page 最小 1，page_size 限制在 1~MAX_PAGE_SIZE。"""
    return max(1, page), min(max(1, page_size), MAX_PAGE_SIZE)


def _to_dict(row: Any) -> dict[str, Any]:
    """把结果行转成可 JSON 序列化的 dict。"""
    data = dict(row)
    for field in _FLOAT_FIELDS:
        if data.get(field) is not None:
            data[field] = float(data[field])
    for field in _TIME_FIELDS:
        if data.get(field) is not None:
            data[field] = str(data[field])
    return data


async def search_secondhand_items(
    keyword: str | None = None,
    category_id: int | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    condition: str | None = None,
    sort_by: str = "price_asc",
    page: int = 1,
    page_size: int = 20,
) -> list[dict[str, Any]]:
    """搜索二手商品（结构化条件查询）。"""
    page, page_size = _normalize_paging(page, page_size)
    engine: AsyncEngine = get_secondhand_engine()

    # 动态条件：一律用 :name 占位，值走 params
    where_parts: list[str] = ["status = :status"]
    params: dict[str, Any] = {"status": STATUS_ON_SALE}

    if keyword:
        where_parts.append("name LIKE :keyword")
        params["keyword"] = f"%{keyword}%"
    if category_id is not None:
        where_parts.append("category_id = :category_id")
        params["category_id"] = category_id
    if min_price is not None:
        where_parts.append("price >= :min_price")
        params["min_price"] = min_price
    if max_price is not None:
        where_parts.append("price <= :max_price")
        params["max_price"] = max_price
    if condition:
        # 模糊匹配：用户说「九成新」，库里可能存「9成新」
        where_parts.append("condition_desc LIKE :condition")
        params["condition"] = f"%{condition}%"

    # ORDER BY 的值只可能来自 SORT_MAP，安全
    order = SORT_MAP.get(sort_by, SORT_MAP["price_asc"])

    sql = (
        "SELECT id, name, price, category_id, condition_desc, image, status, create_time "
        "FROM tb_item "
        f"WHERE {' AND '.join(where_parts)} "
        f"ORDER BY {order} "
        "LIMIT :page_size OFFSET :offset"
    )
    params["page_size"] = page_size
    params["offset"] = (page - 1) * page_size

    async with engine.connect() as conn:
        result = await conn.execute(text(sql), params)
        rows = result.mappings().all()

    return [_to_dict(row) for row in rows]


async def get_secondhand_detail(item_id: int) -> dict[str, Any] | None:
    """查询单个商品详情（主键查询）。"""
    engine: AsyncEngine = get_secondhand_engine()
    sql = text(
        """
        SELECT id, seller_id, name, price, image, category_id,
               condition_desc, description, trade_location, status,
               create_time, update_time
        FROM tb_item
        WHERE id = :item_id
        """
    )
    async with engine.connect() as conn:
        result = await conn.execute(sql, {"item_id": item_id})
        row = result.mappings().first()

    return None if row is None else _to_dict(row)


# ===== LangChain 工具封装 =====
# 核心函数保持纯 Python（便于单测和复用），这里只包一层给图/模型使用。
# docstring 会被当成工具的 description 送给模型，所以要写清楚用途。


@tool
async def search_secondhand_items_tool(
    keyword: str | None = None,
    max_price: float | None = None,
    category_id: int | None = None,
    page_size: int = 5,
) -> list[dict[str, Any]]:
    """搜索校园在售二手商品。按关键词模糊匹配名称，可按价格上限和分类过滤。"""
    return await search_secondhand_items(
        keyword=keyword,
        max_price=max_price,
        category_id=category_id,
        page_size=page_size,
    )


@tool
async def get_secondhand_detail_tool(item_id: int) -> dict[str, Any] | None:
    """按商品 ID 查询二手商品详情（含描述、交易地点、卖家）。"""
    return await get_secondhand_detail(item_id)


SECONDHAND_TOOLS = [search_secondhand_items_tool, get_secondhand_detail_tool]
