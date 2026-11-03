"""Agent 工具层：CampusHub 查询能力封装。

表结构参考（campushub_secondhand.tb_item）：
    id, seller_id, name, price, image, category_id,
    condition_desc, description, trade_location, status,
    create_time, update_time
    （status: 1在售 0下架 3已售）
"""
from typing import Any


def search_secondhand_items(
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
    ...


def get_secondhand_detail(item_id: int) -> dict[str, Any] | None:
    """查询单个商品详情（主键查询）。"""
    ...
