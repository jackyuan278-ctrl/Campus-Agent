"""日志配置。

约定：各模块用 `logging.getLogger(__name__)` 取自己的 logger，
这样日志格式里的 %(name)s 能看出是哪个模块打的。
本模块只负责一次性配置根 logger。
"""
import logging
import sys

# 配置日志格式：时间 - 级别 - 模块 - 消息
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


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
