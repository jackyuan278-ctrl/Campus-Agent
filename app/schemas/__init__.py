"""请求/响应模型包。

只放 Pydantic 模型（进出接口的数据结构），不放业务逻辑。
"""
from app.schemas.chat import ChatRequest

__all__ = ["ChatRequest"]
