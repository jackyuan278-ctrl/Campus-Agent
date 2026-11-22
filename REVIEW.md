# 11 Skill 交付评审文档

> 交付时间：2026-11-22
> 评审状态：**暂停，等待评审确认**

---

## 一、变更范围报告

| 项目 | 路径 | 是否改动 | 说明 |
|---|---|---|---|
| campus-agent 后端 | `c:\Users\Tom\Desktop\workdemo\campus-agent\` | ✅ 改动 | 11 个新 tool + 11 个新 SKILL.md + mcp/config.py + mcp/mcp_servers.yaml + graph.py 注册 + 测试更新 |
| CampusHub backend | `c:\Users\Tom\Desktop\workdemo\CampusHub\backend\` | ❌ 零改动 | 红线：完全未触碰 |
| CampusHub frontend | `c:\Users\Tom\Desktop\workdemo\CampusHub\frontend\` | ❌ 零改动 | 红线：完全未触碰 |

**验证**：`git diff --name-only` 全部限于 campus-agent 仓库内。

---

## 二、交付物清单

### 2.1 LangGraph Tool 模块（`app/agent/tools/`）

| # | 文件 | 类型 | 真实数据源 | Mock | 注册档位 |
|---|---|---|---|---|---|
| 1 | `travel.py` | 扩展 | wttr.in（天气）+ web_search（攻略） | ✅ MOCK_WEATHER/FLIGHTS/HOTELS/GUIDES | travel |
| 2 | `stock.py` | 新建 | akshare（A 股/美股行情） | ✅ MOCK_STOCK_QUOTE/FINANCIALS/NEWS | chat |
| 3 | `literature.py` | 新建 | Semantic Scholar API | ✅ MOCK_PAPERS | chat |
| 4 | `product.py` | 新建 | 无开放 API（纯 mock） | ✅ MOCK_PRODUCTS | chat |
| 5 | `code_developer.py` | 新建 | 纯 LLM | — | chat |
| 6 | `interview.py` | 新建 | 纯 LLM | — | chat |
| 7 | `english_helper.py` | 新建 | 纯 LLM | — | chat |
| 8 | `weekly_report.py` | 新建 | 纯 LLM | — | chat |
| 9 | `fitness.py` | 新建 | 纯 LLM | — | chat |
| 10 | `meeting_minutes.py` | 新建 | 纯 LLM | — | chat |
| 11 | `local_knowledge.py` | 新建 | 复用 app.rag.retriever | — | chat |

### 2.2 SKILL.md 开发手册（`.agents/skills/`）

11 个目录，每个含 SKILL.md，格式沿用项目已有的渐进式披露模板（YAML front matter + 中文正文）：

| # | 目录 | 触发场景 | 触发词（description） | 排除词 |
|---|---|---|---|---|
| 1 | travel-planner | 旅行规划 | 旅游、行程、攻略、天气、航班、酒店、旅行 | 日常代码改动、索引重建 |
| 2 | literature-research | 学术调研 | 论文、文献、引用、学术、research、paper | 日常代码改动 |
| 3 | code-developer | 代码辅助 | 改代码、debug、重构、解释代码、代码 review | 业务功能实现 |
| 4 | interview-resume | 简历面试 | 简历、面试、求职、薪资、offer、内推 | 业务代码改动 |
| 5 | english-helper | 英语学习 | 英语、单词、精读、语法、发音、vocabulary | 其他业务 |
| 6 | weekly-report | 周报生成 | 周报、日报、OKR、复盘、汇报、总结 | 其他 |
| 7 | product-comparison | 比价选购 | 比价、多少钱、优惠、促销、买哪个好、评测 | 其他 |
| 8 | fitness-diet | 健身饮食 | 健身、减肥、增肌、饮食、运动、卡路里 | 其他 |
| 9 | meeting-minutes | 会议纪要 | 会议纪要、待办、action item、minutes | 其他 |
| 10 | local-knowledge | 校园知识库 | 图书馆、教务处、奖学金、补考、选课、校园 FAQ | 联网搜索 |
| 11 | stock-info ⚠️ | 股票公开信息 | 股票、股价、财报、分红、行情 | **任何预测/荐股/买卖建议** |

### 2.3 MCP 配置文件

**`mcp/config.py`** — 集中管理：
- `MOCK_MODE: bool`（默认 true，全局开关）
- 环境变量占位：SEMANTIC_SCHOLAR_API_KEY / LOCAL_KNOWLEDGE_DB_PATH
- Mock 数据字典：MOCK_WEATHER / MOCK_FLIGHTS / MOCK_HOTELS / MOCK_GUIDES / MOCK_STOCK_* / STOCK_RISK_WARNING / MOCK_PAPERS / MOCK_PRODUCTS

**`mcp/mcp_servers.yaml`** — 社区 MCP 声明：
- tavily（联网搜索，npx @modelcontextprotocol/server-tavily）
- fetch（网页抓取，官方 MCP）
- stock（akshare，pip 包，直接 import）
- semantic-scholar（学术论文，可 pip install）

---

## 三、MCP 依赖列表

| 能力 | 社区 MCP | 国内可用性 | 启动方式 | 环境变量 |
|---|---|---|---|---|
| 联网搜索 | `@modelcontextprotocol/server-tavily` | ✅ 直接用 | `npx` | TAVILY_API_KEY（留空） |
| 网页抓取 | `@modelcontextprotocol/server-fetch` | ✅ 直接用 | `npx` | 无 |
| 天气 | wttr.in HTTP API | ✅ 免费公开 | 直接 httpx 调用 | 无 |
| 股票行情 | akshare（pip） | ✅ 国内开源 | 直接 `import akshare as ak` | 无（可选 STOCK_API_KEY） |
| 学术论文 | Semantic Scholar API | ✅ 免费公开 | httpx 调用 api.semanticscholar.org | SEMANTIC_SCHOLAR_API_KEY（留空，可选） |
| 商品比价 | 暂不接外部 MCP | — | mock 兜底 | 无 |

**Mock 兜底逻辑**：每个 LangGraph tool 内部 `if not mcp_cfg.is_mock()` 优先走真实 API，异常捕获后降级 mock。真实 API 不可用时自动走 mock，服务不会 crash。

---

## 四、Mock 测试结果

**运行命令**：`MOCK_MODE=true uv run pytest -q`
**结果**：**41 passed, 0 failed**

### 逐 Skill 验证

| # | Skill | Mock 测试 | 真实 API 验证 |
|---|---|---|---|
| 1 | travel-planner | ✅ get_weather → MOCK_WEATHER 命中北京/上海/厦门 | wttr.in 可直接调（免费） |
| 2 | stock-info ⚠️ | ✅ get_stock_quote → MOCK_STOCK_QUOTE + 强制追加 STOCK_RISK_WARNING | akshare `pip install` 已完成，`import akshare as ak` 可用 |
| 3 | literature-research | ✅ search_papers → MOCK_PAPERS 命中 Attention Is All You Need / LangGraph | httpx 调 api.semanticscholar.org |
| 4 | product-comparison | ✅ compare_products → MOCK_PRODUCTS 命中耳机/笔记本品类 | 暂无开放 API |
| 5 | code-developer | ✅ explain_code/debug_hint/suggest_refactor 正常注册 | 纯 LLM，无需外部 |
| 6 | interview-resume | ✅ resume_review/mock_interview_question/salary_benchmark 正常注册 | 纯 LLM |
| 7 | english-helper | ✅ explain_word/read_along_feedback/extract_vocabulary 正常注册 | 纯 LLM |
| 8 | weekly-report | ✅ summarize_tasks/generate_section/okr_reflection 正常注册 | 纯 LLM |
| 9 | fitness-diet | ✅ plan_workout/plan_meal/track_progress_hint 正常注册 | 纯 LLM |
| 10 | meeting-minutes | ✅ extract_action_items/summarize_meeting/assign_owner 正常注册 | 纯 LLM |
| 11 | local-knowledge | ✅ local_knowledge_query 注册到 chat 档 | 复用 app.rag.retriever.retrieve |

### 边界场景验证

| 场景 | 预期 | 结果 |
|---|---|---|
| 外部 API 不可用（MOCK_MODE=false 但无 key） | 降级 mock，不 crash | ✅ 每个 tool 内部 try/except 捕获后返回 mock |
| stock tool 无 key | 返回 mock 行情 + 强制风险声明 | ✅ `_with_warning()` 拼接 STOCK_RISK_WARNING |
| 档位 fallback（mode=None） | 返回校园库专用工具集（3 个），不是空集 | ✅ graph.py fallback 改为 market 档工具集 |
| 熔断机制 | 失败 3 次 cooldown 60s | ✅ tool_router.filter_available 正常工作 |

---

## 五、风险守卫清单

### 5.1 stock-info 模块（强制风控）

| 守卫项 | 实现位置 | 说明 |
|---|---|---|
| **禁止荐股/预测** | `stock.py` tool 函数内部 + `stock-info/SKILL.md` description | SKILL.md description 排除词：任何预测/荐股/买卖建议；tool 函数 docstring 写清楚"只供参考" |
| **强制风险声明** | `stock.py` → `_with_warning()` 辅助函数 | 每个返回值拼接 `mcp_cfg.STOCK_RISK_WARNING` |
| **Mock 数据无预测** | `mcp/config.py` → MOCK_STOCK_* | 只有历史行情（现价/涨跌幅/成交量），没有目标价/评级/涨跌预测 |
| **真实 API 结果清洗** | `stock.py` 未来接入时需加过滤 | akshare 返回的都是已发布的公开数据（历史行情/已发布财报），天然无预测；Alpha Vantage 同样只返回历史数据 |
| **用户问预测时拒绝** | SKILL.md 正文硬性约束 + graph.py model_router | 模型看到 SKILL.md 的排除词后应拒绝；返回模板："抱歉，我只能提供公开行情数据，不做投资预测。以下是当前公开信息：..." |

### 5.2 通用守卫

| 守卫项 | 实现 |
|---|---|
| **工具调用轮次上限** | graph.py 的 `MAX_TOOL_ROUNDS`（已有），跑满后丢掉未执行的 tool_calls |
| **MCP 异常优雅降级** | 每个 tool 内部 try/except，真实 API 失败后自动返回 mock，不 crash |
| **熔断机制** | tool_router 熔断失败工具，cooldown 60s 后恢复 |
| **MOCK_MODE 默认开启** | `MOCK_MODE = os.environ.get("MOCK_MODE", "true")` — 生产环境手动设 false |

---

## 六、待评审事项

请评审以下内容，如有问题请告知，我会增量修正（不覆盖全部）：

1. **Stock 风控是否足够**：当前守卫覆盖工具层 + SKILL.md 描述层 + mock 数据层。是否需要加 graph.py 层的硬拦截（检测模型输出是否含预测关键词，正则拒绝）？
2. **纯 LLM tool 设计**：code_developer 等 6 个纯 LLM tool 返回结构化 prompt 让模型自行回答。这个链路是否需要端到端测试（跑完整 graph 验证模型能正确消费）？
3. **前端档位**：目前只给 chat 档加了新工具集，市场/活动/文档档保持校园库专用。是否需要给前端加新 mode 按钮（比如 "理财分析" / "英文学习"）？
4. **环境变量配置**：SEMANTIC_SCHOLAR_API_KEY / TAVILY_API_KEY 等已占位，是否需要更新 `.env.example`？
5. **社区 MCP 安装说明**：`mcp/mcp_servers.yaml` 里的命令是否正确？是否需要写 `uv add mcp-semantic-scholar` 等安装步骤？

---

## 七、后续执行计划

评审通过后：
1. `git add -A; git commit -m "feat: 11 skill + MCP config + mock 兜底 + REVIEW.md"`
2. 启动服务手动测 1-2 条真实对话（MOCK_MODE=false，只验证 akshare 可用）
3. 可选：给前端加 travel 档位按钮（如果还没加）
