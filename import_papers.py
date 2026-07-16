#!/usr/bin/env python3
"""知伴论文向量化 — 批量导入 postgraduate-paper/papers/"""
import sys
import time
from pathlib import Path

# Add sidecar to path
_zhiban_root = Path("/Users/xiaodu/zhiban-webui")
sys.path.insert(0, str(_zhiban_root))
sys.path.insert(0, str(_zhiban_root / "sidecar"))

from sidecar.rag.vector_store import vector_store
from sidecar.rag.embeddings import embedding_engine

source_dir = Path("/Users/xiaodu/postgraduate-paper/papers").resolve()
print(f"源目录: {source_dir}")
pdfs = sorted(source_dir.glob("*.pdf"))
print(f"论文数: {len(pdfs)}")

start = time.time()
print("正在加载嵌入模型...")
if not embedding_engine.is_available:
    embedding_engine.load()
print(f"模型就绪: {embedding_engine.model_name}")

print("开始构建向量索引...\n")
vector_store.build_index(force=True, source_dir=source_dir)

elapsed = round(time.time() - start, 1)
chunks = vector_store.chunk_count
print(f"\n=== 完成 ===")
print(f"向量块: {chunks}")
print(f"耗时: {elapsed}s ({round(elapsed/60, 1)}min)")
