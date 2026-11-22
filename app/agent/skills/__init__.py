"""SkillRegistry — 让工具集与 graph.py 解耦。

设计原则：
- registry 只管"mode → 哪些工具可用" + "ToolNode 需要的全量工具清单"
- prompt、安全约束、时间注入等共用逻辑仍由 graph.py 持有
- 每个 tool 模块只负责声明"我能服务哪些 mode"，graph.py 底部统一调用 register()

用法（graph.py 底部）：
    from app.agent.skills import registry
    from app.agent.tools import secondhand, activity, web
    secondhand.register(registry)
    activity.register(registry)
    web.register(registry)

加一个新功能（如失物招领）：
    1. 新建 app/agent/tools/lostfound.py，写工具 + register()
    2. graph.py 底部加一行 lostfound.register(registry)
    → graph.py 内部工具分配代码一行不动
"""

import logging
import time
from typing import Any

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class ToolRouter:
    """工具级熔断：某个工具连续失败 N 次后临时跳过。

    最简内存实现（单进程够用），按 tool_name 维度独立计数。
    - failure_threshold: 连续失败多少次后熔断（默认 3）
    - cooldown_seconds: 熔断冷却秒数（默认 60，到期后自动半开）
    - 成功会降计数（半开恢复），不会等到冷却完才恢复
    - 熔断中的工具不会在 filtered list 里出现，模型自然跳过
    """

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: int = 60) -> None:
        self.failure_counts: dict[str, int] = {}
        self.circuit_open_since: dict[str, float] = {}  # tool_name → 熔断开始时间戳
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds

    def is_available(self, tool_name: str) -> bool:
        """查这个工具现在能不能给模型用。熔断中的跳过。"""
        count = self.failure_counts.get(tool_name, 0)
        if count < self.failure_threshold:
            return True
        # 已熔断，但看看冷却到了没
        opened_at = self.circuit_open_since.get(tool_name)
        if opened_at is not None and (time.time() - opened_at) >= self.cooldown_seconds:
            # 冷却到期，半开：清零，允许试一次
            logger.warning("工具 %s 冷却到期，半开尝试", tool_name)
            self.failure_counts[tool_name] = 0
            self.circuit_open_since.pop(tool_name, None)
            return True
        return False

    def record_failure(self, tool_name: str) -> None:
        """记录一次失败。达到阈值时触发熔断。"""
        count = self.failure_counts.get(tool_name, 0) + 1
        self.failure_counts[tool_name] = count
        if count >= self.failure_threshold and tool_name not in self.circuit_open_since:
            logger.warning(
                "工具 %s 连续失败 %d 次，进入熔断（冷却 %ds）",
                tool_name,
                count,
                self.cooldown_seconds,
            )
            self.circuit_open_since[tool_name] = time.time()

    def record_success(self, tool_name: str) -> None:
        """成功会逐步降计数（半开恢复）。不会完全清零计数，只往 0 靠。"""
        if tool_name in self.failure_counts:
            self.failure_counts[tool_name] = max(0, self.failure_counts[tool_name] - 1)

    def filter_available(self, tools: list[BaseTool]) -> list[BaseTool]:
        """过滤掉熔断中的工具，返回可用清单。"""
        available = [t for t in tools if self.is_available(t.name)]
        removed = [t.name for t in tools if not self.is_available(t.name)]
        if removed:
            logger.warning("以下工具因熔断被跳过：%s", removed)
        return available

    def reset(self, tool_name: str | None = None) -> None:
        """手动重置熔断状态（测试用或运维用）。不传名则全部重置。"""
        if tool_name is None:
            self.failure_counts.clear()
            self.circuit_open_since.clear()
        else:
            self.failure_counts.pop(tool_name, None)
            self.circuit_open_since.pop(tool_name, None)


class SkillRegistry:
    """mode → 工具集 的注册中心。

    同一个工具可以注册给多个 mode；registry 会按 mode 合并，
    并维护一份按 name 去重的全量清单供 ToolNode 使用。
    """

    def __init__(self) -> None:
        # mode → [tool, ...]  按注册顺序合并
        self._mode_tools: dict[str, list[BaseTool]] = {}
        # tool_name → tool  去重用
        self._all_tools: dict[str, BaseTool] = {}
        # mode → prompt suffix  档位人设（可选，不注册时 graph.py 用兜底 prompt）
        self._mode_prompts: dict[str, str] = {}

    def register_tools(self, mode: str, tools: list[BaseTool]) -> None:
        """给某个 mode 追加一组工具。

        同名工具跨 mode 只保留一份（ToolNode 执行时按 name 反查）。
        """
        bucket = self._mode_tools.setdefault(mode, [])
        for tool in tools:
            if tool not in bucket:
                bucket.append(tool)
            self._all_tools[tool.name] = tool

    def register_prompt(self, mode: str, prompt: str) -> None:
        """给某个 mode 注册人设 prompt suffix（工具包自己的档位人设放在这里注册）。"""
        self._mode_prompts[mode] = prompt

    def get_prompt(self, mode: str | None) -> str | None:
        """取某个 mode 的 prompt suffix。未注册返回 None（graph.py 会 fallback）。"""
        if not mode:
            return None
        return self._mode_prompts.get(mode)

    def get_tools_for_mode(
        self, mode: str | None, fallback: list[BaseTool] | None = None
    ) -> list[BaseTool]:
        """取某个 mode 下的工具集。

        mode 未注册时返回 fallback（graph.py 用来给未知档位兜底）。
        返回的是副本，外部改不动内部状态。
        """
        if not mode:
            return list(fallback or [])
        tools = self._mode_tools.get(mode)
        if tools is None:
            return list(fallback or [])
        return list(tools)

    def get_all_tools(self) -> list[BaseTool]:
        """ToolNode 需要的全量工具清单（按 name 去重）。"""
        return list(self._all_tools.values())

    def list_modes(self) -> list[str]:
        return list(self._mode_tools.keys())

    def __repr__(self) -> str:
        return (
            f"SkillRegistry(modes={list(self._mode_tools.keys())}, "
            f"total_tools={len(self._all_tools)})"
        )


# 全局单例（graph.py 底部统一填充）
registry = SkillRegistry()
# 全局熔断路由器（单进程内存）
tool_router = ToolRouter()
