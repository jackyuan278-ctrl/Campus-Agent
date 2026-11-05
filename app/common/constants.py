"""全局常量（约定：模块级全大写命名，供各模块 import 复用）。

按需往里加，避免在业务代码里散落硬编码。

分组建议：
    - 意图常量：INTENT_XXX
    - 分页/限流：MAX_PAGE_SIZE 等
    - 会话：上下文条数等
"""

# 查询历史消息（GET /assistant/chat/messages）的默认条数。
# 注意：这是「给人看」的分页默认值，不影响模型上下文，也不影响存储。
CHAT_HISTORY_LIMIT = 20

# 上下文压缩（SummarizationMiddleware）：
# 消息累计到 TRIGGER 条时，把早期对话压缩成一段摘要，保留最近 KEEP 条原文。
# 为什么是摘要而不是截断：截断会直接丢掉「用户想找什么、已经查到什么」，
# 模型下一轮就断片；摘要有损但保语义。
CHAT_SUMMARY_TRIGGER = 40
CHAT_SUMMARY_KEEP = 20

# 工具调用轮数上限（一轮 = 模型要工具 → 执行 → 模型再想）。
# 防止模型反复要工具陷入死循环；到上限后不再给它工具，逼它直接回答。
MAX_TOOL_ROUNDS = 3

# 单轮最多读几张图（每张 = 一次视觉模型调用，多了又慢又贵）。
# 与前端 ATTACHMENT_RULES.maxCount 对齐：前端允许传 5 个附件，这里也得能读 5 张，
# 不然第 4、5 张会被静默丢掉，用户完全不知道。
MAX_IMAGES = 5

# ---------- 文档解析与切块 ----------
# 一份 50 页讲义约 3 万字，整篇塞进 prompt 是 3 万 token——日配额 10 万，三轮就爆。
# 所以切块后只带最相关的几块进去。
DOC_CHUNK_SIZE = 400      # 每块目标字数
DOC_CHUNK_OVERLAP = 80    # 相邻块重叠，避免答案正好被切断
# 每轮带几块进 prompt：4 块 ≈ 1600 字 ≈ 1000 token
DOC_TOP_K = 4
