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

from typing import Any

from langchain_core.tools import BaseTool


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

    def register_tools(self, mode: str, tools: list[BaseTool]) -> None:
        """给某个 mode 追加一组工具。

        同名工具跨 mode 只保留一份（ToolNode 执行时按 name 反查）。
        """
        bucket = self._mode_tools.setdefault(mode, [])
        for tool in tools:
            if tool not in bucket:
                bucket.append(tool)
            self._all_tools[tool.name] = tool

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
