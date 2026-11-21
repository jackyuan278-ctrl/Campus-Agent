#!/usr/bin/env python3
"""只读：打印校园索引的规模、维度、新鲜度与字段覆盖情况。

用法（仓库根目录）：
    uv run python .agents/skills/campus-agent/scripts/index_stats.py

不连库、不调 embedding、不写文件。
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
RAG = ROOT / "data" / "rag"


def human(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


def main() -> int:
    docs_path = RAG / "docs.json"
    vec_path = RAG / "vectors.npy"
    if not docs_path.exists():
        print(f"[!] 没有 {docs_path}，先跑：uv run python -m app.rag.ingest --limit 0")
        return 1

    docs = json.loads(docs_path.read_text(encoding="utf-8"))
    print(
        f"docs.json     {len(docs)} 条，{docs_path.stat().st_size / 1024:.0f} KB，更新于 {human(docs_path.stat().st_mtime)}"
    )

    if vec_path.exists():
        try:
            import numpy as np

            arr = np.load(vec_path, mmap_mode="r")
            print(
                f"vectors.npy   {arr.shape[0]} × {arr.shape[1]}，{vec_path.stat().st_size / 1048576:.2f} MB，更新于 {human(vec_path.stat().st_mtime)}"
            )
            if arr.shape[0] != len(docs):
                print(
                    f"[!] 行数不一致（向量 {arr.shape[0]} / 文档 {len(docs)}）—— 索引已错位，必须重建"
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[!] vectors.npy 读取失败：{exc}")
    else:
        print("[!] 没有 vectors.npy")

    sources = Counter(d.get("source") for d in docs)
    print("来源分布      " + "，".join(f"{k}={v}" for k, v in sources.items()))

    acts = [d for d in docs if d.get("source") == "activity"]
    if acts:
        with_time = sum(1 for d in acts if d.get("start_time"))
        with_end = sum(1 for d in acts if d.get("end_time"))
        print(f"活动时间字段   start_time {with_time}/{len(acts)}，end_time {with_end}/{len(acts)}")
        if with_time == 0:
            print("[!] 索引没有 start_time —— 时间过滤会静默失效，跑一次 ingest（0 次 embedding）")

    fps = [d.get("fingerprint") for d in docs]
    dup = len(fps) - len(set(fps))
    if dup:
        print(f"[!] 指纹重复 {dup} 条 —— 疑似旧索引或去重失效")

    milvus = RAG / "milvus.db"
    if milvus.exists():
        size = sum(p.stat().st_size for p in milvus.rglob("*") if p.is_file())
        print(f"附件库        {milvus}，{size / 1024:.0f} KB（块数需用 MilvusClient 查，进程独占）")
    else:
        print("附件库        未创建（还没成功解析过文档）")

    print(f"仓库根        {ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
