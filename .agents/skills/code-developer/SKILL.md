---
name: code-developer
description: 用户要改代码、解释代码、debug、重构建议时触发。触发词：改代码、debug、重构、解释代码、代码 review。排除：业务功能实现（那是另一个任务）。
---

# code-developer

给用户提供代码阅读、debug 诊断、重构建议的 skill。读本地仓库文件、跑测试、静态检查，**不负责写新业务功能**。

## 快速用法

```python
# 读文件 + 静态分析
summary = await explain_file(path="app/agent/graph.py")

# 跑单测并定位失败
result = await run_tests(path="tests/agent/", keyword="test_route")

# 生成重构建议
tips = await suggest_refactor(path="app/agent/tools/secondhand.py")
```

## 工具清单

| tool_name | 功能 | 数据源 |
|---|---|---|
| `explain_file` | 函数级摘要 + 依赖关系梳理 | 本地文件 + AST |
| `run_tests` | 调 pytest 跑指定路径/关键词的测试 | 本地 pytest |
| `suggest_refactor` | 给出可落地的重构建议（圈复杂度、重复代码等） | ruff + radon |
| `search_code` | ripgrep 跨文件正则搜索 | 本地文件 |
| `git_blame_lookup` | 看某行代码最后一次修改者和提交 | git |

## 硬性约束（别碰）

1. **绝不自动改代码**——只输出 diff 预览，由用户确认后再应用
2. 不运行危险 shell 命令（`rm -rf /`、`sudo`、写系统目录）
3. `run_tests` 默认带 `-x` 遇错即停，不要跑全量测试消耗太久
4. 解释外部依赖（第三方库）时只引用官方文档链接，不瞎编内部实现

## Mock 测试

```bash
# 不需要真实仓库也能跑通 explain_file
MOCK_MODE=true uv run python -c "
from app.skills.code_developer import explain_file
print(explain_file('app/agent/graph.py'))
"
# mock 会返回预置的 3 节点图解释
```
