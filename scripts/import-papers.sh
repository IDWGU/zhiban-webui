#!/usr/bin/env bash
# 知伴 — 一键导入论文脚本
# 用法: ./scripts/import-papers.sh /path/to/papers/
# 将目录下的 PDF/TXT/MD 文件索引到 ChromaDB 向量库

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

PAPERS_DIR="${1:-}"

if [ -z "$PAPERS_DIR" ]; then
  echo "用法: $0 <论文目录路径>"
  echo "示例: $0 ~/Downloads/papers/"
  echo "       $0 ./my-papers/"
  echo
  echo "支持的格式: PDF, DOCX, TXT, MD"
  exit 1
fi

if [ ! -d "$PAPERS_DIR" ]; then
  echo "错误: 目录不存在 — $PAPERS_DIR"
  exit 1
fi

echo "论文目录: $(cd "$PAPERS_DIR" && pwd)"
echo ""

# Activate virtual environment
VENV="$SCRIPT_DIR/sidecar/.venv"
if [ -d "$VENV/bin" ]; then
  source "$VENV/bin/activate"
  echo "venv: $VENV"
else
  echo "warning: .venv not found at $VENV, using system python3"
fi

export PYTHONPATH="$SCRIPT_DIR/sidecar:$PYTHONPATH"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "PYTHONPATH: $PYTHONPATH"
echo "HF_ENDPOINT: $HF_ENDPOINT"
echo ""
echo "正在构建向量索引..."
echo ""

python3 -c "
from pathlib import Path
import sys, time

sys.path.insert(0, 'sidecar')
from sidecar.rag.vector_store import vector_store
from sidecar.rag.embeddings import embedding_engine

source_dir = Path('$PAPERS_DIR').resolve()
print(f'源目录: {source_dir}')
print(f'文件列表:')
for f in sorted(source_dir.glob('*')):
    if f.suffix.lower() in ('.pdf', '.docx', '.txt', '.md'):
        print(f'  {f.name}')

print()
start = time.time()

if not embedding_engine.is_available:
    print('正在加载 BGE-M3 嵌入模型...')
    embedding_engine.load()

vector_store.build_index(force=True, source_dir=source_dir)

elapsed = round(time.time() - start, 1)
print(f'完成! 向量库共 {vector_store.chunk_count} 个向量块')
print(f'耗时: {elapsed}s')
"

echo ""
echo "索引完成。重启知伴应用后即可在 AI 问答中检索新论文。"
