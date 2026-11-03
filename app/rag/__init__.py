"""RAG 检索层：活动 + 社区问答的向量化检索（占位）。

后续实现：
1. ingest：从活动/社区表读数据 → 切分 → BGE 向量化 → Milvus 入库
2. retriever：语义检索 + 回源校验（拿 item_id 回 MySQL 确认）
3. rerank：重排

当前先占位，把包结构搭好。
"""
