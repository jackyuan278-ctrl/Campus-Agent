"""日志配置。

约定：各模块用 `logging.getLogger(__name__)` 取自己的 logger，
这样日志格式里的 %(name)s 能看出是哪个模块打的。
本模块只负责一次性配置根 logger。
"""
import logging
import sys

# 配置日志格式：时间 - 级别 - 模块 - 消息
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


class _DropKnownNoise(logging.Filter):
    """丢掉第三方库已知无害的噪音日志。

    目前只有一条：Milvus Lite 的 gRPC 服务端没实现 AllocTimestamp，客户端连上来
    探测时会打一句 "Method not implemented!"（ERROR 级别）。功能完全正常，
    但每次建集合都刷一条 ERROR，会淹掉真正的错误。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        noisy = (
            record.name.startswith("grpc")
            and "Method not implemented" in record.getMessage()
        )
        return not noisy


def setup_logging() -> None:
    """初始化根 logger。重复调用不会叠加 handler。"""
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),  # 输出到控制台
            # logging.FileHandler("app.log"),   # 需要落盘时开启
        ],
        force=False,
    )
    logging.getLogger("grpc").addFilter(_DropKnownNoise())
