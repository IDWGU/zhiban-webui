#!/usr/bin/env bash
# 知伴模型下载器 — 国内镜像轮换 + 超时自动切换
# 用法: source scripts/download-models.sh && download_all_models [model_key]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# 镜像列表（按优先级排序）
MIRRORS=(
    "https://hf-mirror.com"
    "https://huggingface.co"
    "https://huggingface.modelscope.cn"
    "https://aliendao.cn"
)

# 下载单个模型文件，依次尝试所有镜像
# 参数: repo file dest_dir [size_hint]
download_with_mirrors() {
    local repo="$1"
    local file="$2"
    local dest_dir="$3"
    local size_hint="${4:-未知}"

    mkdir -p "$dest_dir"
    local dest_file="${dest_dir}/${file}"

    # 检查是否已存在且完整（文件 > 1MB 视为有效）
    if [ -f "$dest_file" ]; then
        local existing_size
        existing_size=$(stat -f%z "$dest_file" 2>/dev/null || stat -c%s "$dest_file" 2>/dev/null || echo 0)
        if [ "$existing_size" -gt 1048576 ]; then
            echo -e "   ${GREEN}✅ 已存在${NC} (${existing_size} bytes)"
            return 0
        else
            echo -e "   ${YELLOW}⚠ 文件不完整，重新下载${NC}"
            rm -f "$dest_file"
        fi
    fi

    echo -e "   📦 ${CYAN}${file}${NC} (${size_hint})"

    local mirror_idx=1
    for mirror in "${MIRRORS[@]}"; do
        local url="${mirror}/${repo}/resolve/main/${file}"
        echo -ne "      [${mirror_idx}/${#MIRRORS[@]}] ${mirror} ... "

        # curl: -L 跟随重定向, -C - 断点续传, --connect-timeout 连接超时
        if curl -L --connect-timeout 15 --max-time 3600 -C - \
            --progress-bar \
            -o "${dest_file}.part" \
            "$url" 2>&1; then

            # 校验文件大小
            local dl_size
            dl_size=$(stat -f%z "${dest_file}.part" 2>/dev/null || stat -c%s "${dest_file}.part" 2>/dev/null || echo 0)
            if [ "$dl_size" -gt 1048576 ]; then
                mv "${dest_file}.part" "$dest_file"
                echo -e "${GREEN}✅ 完成${NC} (${dl_size} bytes)"
                return 0
            else
                echo -e "${RED}❌ 文件过小${NC}"
                rm -f "${dest_file}.part"
            fi
        else
            echo -e "${RED}❌ 失败${NC}"
            rm -f "${dest_file}.part"
        fi

        mirror_idx=$((mirror_idx + 1))
        sleep 1
    done

    echo -e "   ${RED}🚫 所有镜像均下载失败: ${file}${NC}"
    return 1
}

# 下载 Jina 嵌入模型（通过 Python + sentence-transformers）
download_jina_embedding() {
    echo -e "   📦 ${CYAN}jina-embeddings-v5-text-nano${NC} (~0.5GB)"

    local jina_cache="${PROJECT_DIR}/models/embedding/models--jinaai--jina-embeddings-v5-text-nano"
    if [ -d "$jina_cache/snapshots" ]; then
        local snap_count
        snap_count=$(find "$jina_cache/snapshots" -name "*.safetensors" 2>/dev/null | wc -l)
        if [ "$snap_count" -gt 0 ]; then
            echo -e "   ${GREEN}✅ 已存在${NC}"
            return 0
        fi
    fi

    # 尝试每个镜像
    local mirror_idx=1
    for mirror in "${MIRRORS[@]}"; do
        echo -ne "      [${mirror_idx}/${#MIRRORS[@]}] ${mirror} ... "

        HF_ENDPOINT="$mirror" python3 -c "
import os, sys
os.environ['HF_ENDPOINT'] = '$mirror'
try:
    from huggingface_hub import snapshot_download
    snapshot_download(
        'jinaai/jina-embeddings-v5-text-nano',
        cache_dir='${PROJECT_DIR}/models/embedding',
        resume_download=True,
        max_workers=4,
    )
    print('OK')
except Exception as e:
    print(f'FAIL: {e}')
    sys.exit(1)
" 2>&1 | tail -3

        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo -e "${GREEN}✅ 完成${NC}"
            return 0
        fi
        echo -e "${RED}❌ 失败${NC}"
        mirror_idx=$((mirror_idx + 1))
        sleep 1
    done

    echo -e "   ${RED}🚫 Jina 嵌入模型下载失败${NC}"
    return 1
}

# 检查至少有一个 LLM 模型
has_llm_model() {
    local llm_dir="${PROJECT_DIR}/models/llm"
    if [ -d "$llm_dir" ]; then
        local gguf_count
        gguf_count=$(find "$llm_dir" -name "*.gguf" ! -name "mmproj*" 2>/dev/null | wc -l)
        if [ "$gguf_count" -gt 0 ]; then
            return 0
        fi
    fi
    return 1
}

# 下载所有必需模型
download_all_models() {
    echo ""
    echo -e "${CYAN}══════════════════════════════════════════${NC}"
    echo -e "${CYAN}  模型检测与下载${NC}"
    echo -e "${CYAN}══════════════════════════════════════════${NC}"

    local missing_required=0

    # === 翻译模型（必需）===
    echo ""
    echo -e "${YELLOW}[翻译模型]${NC} Hy-MT2-1.8B — 论文翻译专用"
    if [ -f "${PROJECT_DIR}/models/translation/Hy-MT2-1.8B-Q4_K_M.gguf" ]; then
        echo -e "   ${GREEN}✅ 已存在${NC}"
    else
        download_with_mirrors \
            "tencent/Hy-MT2-1.8B-GGUF" \
            "Hy-MT2-1.8B-Q4_K_M.gguf" \
            "${PROJECT_DIR}/models/translation" \
            "~1.1GB" || missing_required=1
    fi

    # === LLM 模型（按需下载）===
    echo ""
    echo -e "${YELLOW}[对话模型]${NC} 选择一个即可（推荐 Qwopus3.5-9B）"
    if has_llm_model; then
        echo -e "   ${GREEN}✅ 已存在 LLM 模型${NC}"
    else
        echo -e "   ${CYAN}请选择要下载的模型:${NC}"
        echo -e "   ${CYAN}  [1]${NC} Qwopus3.5-9B-v3  (~5.4GB) ${GREEN}★推荐${NC}"
        echo -e "   ${CYAN}  [2]${NC} Qwopus3.5-4B-v3  (~2.6GB) 轻量"
        echo -e "   ${CYAN}  [3]${NC} 跳过，我自己有模型文件"
        echo -ne "   ${CYAN}输入 [1/2/3]:${NC} "

        read -r choice
        case "$choice" in
            1)
                download_with_mirrors \
                    "Jackrong/Qwopus3.5-9B-v3-GGUF" \
                    "Qwopus3.5-9B-v3.Q4_K_M.gguf" \
                    "${PROJECT_DIR}/models/llm" \
                    "~5.4GB" || true
                ;;
            2)
                download_with_mirrors \
                    "Jackrong/Qwopus3.5-4B-v3-GGUF" \
                    "Qwen3.5-4B.Q4_K_M.gguf" \
                    "${PROJECT_DIR}/models/llm" \
                    "~2.6GB" || true
                ;;
            *)
                echo -e "   ${YELLOW}已跳过。可将 GGUF 模型放入 models/llm/ 目录${NC}"
                ;;
        esac
    fi

    # === 嵌入模型（必需）===
    echo ""
    echo -e "${YELLOW}[嵌入模型]${NC} Jina Embedding v5 Nano — 向量检索"
    download_jina_embedding || missing_required=1

    echo ""
    if [ "$missing_required" -eq 0 ]; then
        echo -e "${GREEN}✅ 所有必需模型就绪${NC}"
    else
        echo -e "${RED}⚠ 部分必需模型缺失，功能可能受限${NC}"
    fi
    echo ""

    return $missing_required
}

# 如果直接执行此脚本
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    download_all_models
fi
