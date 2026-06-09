#!/bin/bash
# 构建知伴 Sidecar 分发目录 (portable Python + 模型 + 源码)
# 用法: bash scripts/build-sidecar.sh           # 完整打包（含模型）
#       SKIP_MODELS=1 bash scripts/build-sidecar.sh  # 仅 Python + 源码，不含模型
#       SMOKE_TEST=1 bash scripts/build-sidecar.sh   # 打包后自动冒烟测试
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SIDECAR_SRC="$PROJECT_DIR/sidecar"
DIST_DIR="$PROJECT_DIR/sidecar-dist"
PYTHON_VER="3.14.4"
PYTHON_BUILD="20260408"
PYTHON_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_BUILD}/cpython-${PYTHON_VER}+${PYTHON_BUILD}-aarch64-apple-darwin-install_only.tar.gz"

# ===== 模型来源路径 =====
LLM_MODEL_SRC="${LLM_MODEL_SRC:-$HOME/.lmstudio/models/Jackrong/Qwopus3.5-2B-v3-GGUF}"
LLM_MODEL_FILE="Qwopus3.5-2B-v3-Q4_K_M.gguf"
SKIP_MODELS="${SKIP_MODELS:-0}"
SMOKE_TEST="${SMOKE_TEST:-0}"

echo "=== 知伴 Sidecar 打包 (portable Python) ==="

# ===== 步骤1: 获取 portable Python =====
echo ""
echo "--- [1/6] Portable Python ---"
PYTHON_CACHE="$HOME/.cache/zhiban/python-${PYTHON_VER}-${PYTHON_BUILD}.tar.gz"

if [ ! -f "$PYTHON_CACHE" ]; then
  echo "  下载 portable Python (${PYTHON_VER})..."
  mkdir -p "$(dirname "$PYTHON_CACHE")"
  curl -L -o "$PYTHON_CACHE" "$PYTHON_URL" 2>&1 | tail -1
  echo "  缓存: $PYTHON_CACHE ($(du -sh "$PYTHON_CACHE" | cut -f1))"
else
  echo "  使用缓存: $PYTHON_CACHE"
fi

echo "  提取 Python..."
rm -rf "$DIST_DIR/python"
mkdir -p "$DIST_DIR/python-tmp"
tar xzf "$PYTHON_CACHE" -C "$DIST_DIR/python-tmp/"
mv "$DIST_DIR/python-tmp/python" "$DIST_DIR/python"
rm -rf "$DIST_DIR/python-tmp"

# 删除不需要的文件
rm -rf "$DIST_DIR/python/include" "$DIST_DIR/python/share" 2>/dev/null || true
echo "  ✅ Python $(du -sh "$DIST_DIR/python" | cut -f1)"

# ===== 步骤2: 安装 pip 依赖 =====
echo ""
echo "--- [2/6] Pip 依赖 ---"
PYBIN="$DIST_DIR/python/bin/python3.14"

# 确保 pip 可用
"$PYBIN" -m ensurepip --upgrade 2>/dev/null || true

echo "  安装核心依赖..."
"$PYBIN" -m pip install --quiet \
  fastapi "uvicorn[standard]" websockets httpx openai \
  python-dotenv pydantic pyyaml networkx chromadb \
  sentence-transformers \
  pyobjc-framework-Vision pyobjc-framework-Quartz pyobjc-framework-ScreenCaptureKit \
  numpy psutil peft \
  llama-cpp-python PyMuPDF python-docx mlx-lm python-multipart 2>&1 | tail -3

echo "  验证关键模块..."
"$PYBIN" -c "
for m in ['fastapi','uvicorn','websockets','httpx','chromadb','sentence_transformers','numpy','psutil']:
    __import__(m)
print('  ✅ 全部模块 OK')
"

echo "  Python 最终大小: $(du -sh "$DIST_DIR/python" | cut -f1)"

# ===== 步骤3: 复制 sidecar 源码 =====
echo ""
echo "--- [3/6] Sidecar 源码 ---"
rm -rf "$DIST_DIR/sidecar-src"
mkdir -p "$DIST_DIR/sidecar-src"
cp -R "$SIDECAR_SRC" "$DIST_DIR/sidecar-src/sidecar"

# 清理源码中的非必要文件
find "$DIST_DIR/sidecar-src" \( -name "*_test.py" -o -name "verify.py" -o -name "acceptance.py" \) -delete 2>/dev/null
find "$DIST_DIR/sidecar-src" -name "*.pyc" -delete 2>/dev/null
find "$DIST_DIR/sidecar-src" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$DIST_DIR/sidecar-src" -name ".DS_Store" -delete 2>/dev/null
find "$DIST_DIR/sidecar-src" -type l ! -exec test -e {} \; -delete 2>/dev/null
rm -f "$DIST_DIR/sidecar-src/sidecar/.env"
rm -rf "$DIST_DIR/sidecar-src/.cache" "$DIST_DIR/sidecar-src/.chroma" "$DIST_DIR/sidecar-src/.conversations" 2>/dev/null || true

echo "  ✅ 源码 ($(du -sh "$DIST_DIR/sidecar-src" | cut -f1))"

# ===== 步骤4: 复制模型 =====
echo ""
echo "--- [4/6] 模型 ---"
mkdir -p "$DIST_DIR/models/llm" "$DIST_DIR/models/translation"

if [ "$SKIP_MODELS" != "1" ]; then
  if [ -f "$LLM_MODEL_SRC/$LLM_MODEL_FILE" ]; then
    cp "$LLM_MODEL_SRC/$LLM_MODEL_FILE" "$DIST_DIR/models/llm/"
    echo "  ✅ LLM: $LLM_MODEL_FILE ($(du -sh "$DIST_DIR/models/llm/$LLM_MODEL_FILE" | cut -f1))"
  else
    echo "  ⚠️  LLM 模型未找到: $LLM_MODEL_SRC/$LLM_MODEL_FILE"
  fi

  # 翻译模型
  TRANS_SRC="${TRANSLATION_SRC:-$HOME/postgraduate-paper/models/hy-mt2-gguf}"
  TRANS_FILE="Hy-MT2-1.8B-Q4_K_M.gguf"
  if [ -f "$TRANS_SRC/$TRANS_FILE" ]; then
    cp "$TRANS_SRC/$TRANS_FILE" "$DIST_DIR/models/translation/"
    echo "  ✅ 翻译: $TRANS_FILE"
  else
    echo "  ⚠️  翻译模型未找到"
  fi

  # Embedding 模型 (从 HuggingFace 缓存)
  EMBEDDING_SRC="$HOME/.cache/huggingface/hub/models--jinaai--jina-embeddings-v5-text-nano"
  if [ -d "$EMBEDDING_SRC" ]; then
    cp -r "$EMBEDDING_SRC" "$DIST_DIR/models/"
    echo "  ✅ Embedding: jina-embeddings-v5-text-nano"
  else
    echo "  ⚠️  Embedding 模型未缓存 (首次启动自动下载)"
  fi
else
  echo "  (SKIP_MODELS=1, 跳过模型)"
fi

echo "  models: $(du -sh "$DIST_DIR/models" 2>/dev/null | cut -f1)"

# ===== 步骤5: 创建配置 + 启动脚本 =====
echo ""
echo "--- [5/6] 配置 & 启动脚本 ---"

# sidecar.json — LLM 模型路径 (放在 .conversations/ 下，config.py 读取的位置)
mkdir -p "$DIST_DIR/sidecar-src/.conversations"
cat > "$DIST_DIR/sidecar-src/.conversations/sidecar.json" << JSONEOF
{"LLM_MODEL_PATH":"models/llm/${LLM_MODEL_FILE}","TRANSLATION_MODEL_PATH":"models/translation/Hy-MT2-1.8B-Q4_K_M.gguf","SIDECAR_DEBUG":false,"LLM_BASE_URL":"__local__","LLM_FLASH_ATTN":true,"LLM_USE_MMAP":true,"LLM_N_BATCH":2048,"LLM_N_UBATCH":1024}
JSONEOF

# start-sidecar.sh (极简版, portable Python 不需要 rpath fix)
cat > "$DIST_DIR/start-sidecar.sh" << 'SCRIPTEOF'
#!/bin/bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
PYBIN="$DIR/python/bin/python3.14"
log() { echo "[sidecar] $(date '+%H:%M:%S') $*" >&2; }
log "启动 (arm64 portable Python)"
export PYTHONPATH="$DIR/sidecar-src:$DIR/python/lib/python3.14/site-packages"
export SIDECAR_NO_RELOAD=1
export MODEL_CACHE="$DIR/models"
export GGML_METAL_PATH_RESOURCES="$DIR"
export GGML_METAL_NDEBUG=1
export LLM_FLASH_ATTN="${LLM_FLASH_ATTN:-1}"
export LLM_USE_MMAP="${LLM_USE_MMAP:-1}"
export LLM_N_BATCH="${LLM_N_BATCH:-2048}"
export LLM_N_UBATCH="${LLM_N_UBATCH:-1024}"
log "PYTHONPATH=$PYTHONPATH"
log "MODEL_CACHE=$MODEL_CACHE"
exec "$PYBIN" -u -P -m sidecar.webui_launcher "$@"
SCRIPTEOF
chmod +x "$DIST_DIR/start-sidecar.sh"

echo "  ✅ sidecar.json + start-sidecar.sh"

# ===== 步骤6: 最终清理 =====
echo ""
echo "--- [6/6] 最终清理 ---"

# 确保无 .env 泄露
find "$DIST_DIR" -name ".env" -not -name ".env.example" -delete 2>/dev/null || true

echo "  ✅ 清理完成"
echo ""
echo "=== 打包完成 ==="
echo "  $DIST_DIR/ ($(du -sh "$DIST_DIR" | cut -f1))"
echo "  ├── python/       $(du -sh "$DIST_DIR/python" 2>/dev/null | cut -f1)"
echo "  ├── sidecar-src/  $(du -sh "$DIST_DIR/sidecar-src" 2>/dev/null | cut -f1)"
echo "  ├── models/       $(du -sh "$DIST_DIR/models" 2>/dev/null | cut -f1)"
echo "  └── start-sidecar.sh"
echo ""
echo "  下一步: npm run build:dist   # 构建完整 DMG"

# ===== 可选: 冒烟测试 =====
if [ "$SMOKE_TEST" = "1" ]; then
  echo ""
  echo "=== 冒烟测试 ==="
  bash "$SCRIPT_DIR/smoke-test-dmg.sh" "$DIST_DIR/sidecar-dist" 2>&1 || true
fi
