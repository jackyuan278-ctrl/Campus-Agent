"""业务异常：统一由 main.py 的全局 handler 转成 JSON 响应。

约定：
- 所有可预期异常都继承 AppError，不要在路由里裸 raise Exception
- 子类用类属性定义默认的 code / status_code / message，实例化时可覆盖
"""


class AppError(Exception):
    """可预期异常的基类。"""

    code: int = 500        # 业务错误码（响应体里的 code）
    status_code: int = 500  # HTTP 状态码
    message: str = "服务内部错误"

    def __init__(
        self,
        message: str | None = None,
        *,
        code: int | None = None,
        status_code: int | None = None,
    ) -> None:
        if message:
            self.message = message
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code
        super().__init__(self.message)


class ConfigError(AppError):
    """配置缺失或非法。启动阶段抛出，宁可起不来也别带着错配置跑。"""

    code = 1001
    status_code = 500
    message = "配置错误"
