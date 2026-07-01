#!/usr/bin/env bash
# ============================================================
#  知伴 (ZhiBan) — AI 论文伴读系统 一键启动脚本
#  环境检测 → 安装依赖 → 下载模型 → 启动服务 → 打开浏览器
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

banner() {
    echo ""
    echo -e "${CYAN}  ╔══════════════════════════════════════╗${NC}"
    echo -e "${CYAN}  ║${NC}     🀄 ${BOLD}知伴 (ZhiBan)${NC}                    ${CYAN}║${NC}"
    echo -e "${CYAN}  ║${NC}     AI 论文伴读系统                  ${CYAN}║${NC}"
    echo -e "${CYAN}  ║${NC}     v0.2.0 — WebUI 一键版            ${CYAN}║${NC}"
    echo -e "${CYAN}  ╚══════════════════════════════════════╝${NC}"
    echo ""
}

# ============================================================
# Step 1: 平台检测
# ============================================================
check_platform() {
    echo -e "${YELLOW}[1/5]${NC} 检测系统环境..."

    local os_name
    os_name=$(uname -s)
    if [ "$os_name" != "Darwin" ]; then
        echo -e "${RED}❌ 知伴目前仅支持 macOS${NC}"
        exit 1
    fi

    local arch
    arch=$(uname -m)
    if [ "$arch" != "arm64" ]; then
        echo -e "${RED}❌ 知伴需要 Apple Silicon (M1/M2/M3/M4) Mac${NC}"
        echo -e "${RED}   Intel Mac 暂不支持本地模型推理${NC}"
        exit 1
    fi

    echo -e "   ${GREEN}✅${NC} macOS arm64 — 已就绪"
}

# ============================================================
# Step 2: 查找 Python 3.12+
# ============================================================
find_python() {
    echo "" >&2
    echo -e "${YELLOW}[2/5]${NC} 查找 Python 3.12+..." >&2

    local candidates=(
        "$SCRIPT_DIR/sidecar/.venv/bin/python3"
        "$(which python3.14 2>/dev/null || true)"
        "$(which python3.13 2>/dev/null || true)"
        "$(which python3.12 2>/dev/null || true)"
        "$(which python3 2>/dev/null || true)"
    )

    for py in "${candidates[@]}"; do
        if [ -n "$py" ] && [ -x "$py" ]; then
            local ver
            ver=$("$py" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0")
            local major
            major=$(echo "$ver" | cut -d. -f1)
            local minor
            minor=$(echo "$ver" | cut -d. -f2)

            if [ "$major" -ge 3 ] && [ "$minor" -ge 12 ]; then
                echo -e "   ${GREEN}✅${NC} Python $ver — $py" >&2
                echo "$py"
                return
            fi
        fi
    done

    # 找不到合适版本
    echo -e "${RED}❌ 未找到 Python 3.12+${NC}" >&2
    echo "" >&2
    echo -e "${YELLOW}请安装 Python 3.12 或更新版本:${NC}" >&2
    echo "" >&2
    echo -e "  方法1 (推荐): ${CYAN}brew install python@3.14${NC}" >&2
    echo -e "  方法2: 从 ${CYAN}https://www.python.org/downloads/${NC} 下载" >&2
    echo "" >&2
    echo -e "安装后重新运行此脚本即可。" >&2
    exit 1
}

# ============================================================
# Step 3: 安装 Python 依赖
# ============================================================
install_deps() {
    echo ""
    echo -e "${YELLOW}[3/5]${NC} 检查 Python 依赖..."

    local py="$1"

    # 快速检查核心包
    local missing=false
    for pkg in fastapi uvicorn psutil websockets httpx openai chromadb sentence_transformers; do
        if ! "$py" -c "import $pkg" 2>/dev/null; then
            missing=true
            break
        fi
    done

    if [ "$missing" = false ]; then
        echo -e "   ${GREEN}✅${NC} 依赖已就绪"
        return
    fi

    echo -e "   ${YELLOW}正在安装依赖 (清华镜像)...${NC}"
    echo ""

    "$py" -m pip install -r "$SCRIPT_DIR/requirements.txt" \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        --trusted-host pypi.tuna.tsinghua.edu.cn \
        2>&1 | tail -5

    # 验证安装
    if "$py" -c "import fastapi, uvicorn" 2>/dev/null; then
        echo -e "   ${GREEN}✅${NC} 依赖安装完成"
    else
        echo -e "${RED}❌ 依赖安装失败，请检查网络后重试${NC}"
        exit 1
    fi
}

# ============================================================
# Step 4: 下载模型
# ============================================================
download_models() {
    echo ""
    echo -e "${YELLOW}[4/5]${NC} 检测模型文件..."

    # 如果配置了 API Key 或显式跳过，则跳过模型下载
    if [ "${ZHIBAN_SKIP_MODELS:-}" = "1" ]; then
        echo -e "   ${YELLOW}⏭️${NC} ZHIBAN_SKIP_MODELS=1，跳过模型下载"
        return
    fi
    if [ -f "$SCRIPT_DIR/.env" ] && grep -q '^LLM_API_KEY=.' "$SCRIPT_DIR/.env" 2>/dev/null; then
        echo -e "   ${YELLOW}⏭️${NC} 检测到 LLM_API_KEY，使用 API 模式，跳过模型下载"
        return
    fi

    # 确保目录存在
    mkdir -p "$SCRIPT_DIR/models/llm"
    mkdir -p "$SCRIPT_DIR/models/translation"
    mkdir -p "$SCRIPT_DIR/models/embedding"

    source "$SCRIPT_DIR/scripts/download-models.sh"
    download_all_models || true  # 模型缺失不阻止启动（可能用户自己放）
}

# ============================================================
# Step 5: 启动服务 + 打开浏览器
# ============================================================
launch() {
    echo ""
    echo -e "${YELLOW}[5/5]${NC} 启动知伴服务..."

    local py="$1"

    # 设置环境变量
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    export ZHIBAN_PROJECT_DIR="$SCRIPT_DIR"

    # 加载 .env 配置（API Key、LLM_BASE_URL 等）
    if [ -f "$SCRIPT_DIR/.env" ]; then
        set -a
        source "$SCRIPT_DIR/.env"
        set +a
    fi

    # 数据目录
    mkdir -p "$SCRIPT_DIR/.conversations"
    mkdir -p "$SCRIPT_DIR/brain/paper-texts"
    mkdir -p "$SCRIPT_DIR/brain/paper-reading"
    mkdir -p "$SCRIPT_DIR/brain/library"

    echo ""
    echo -e "${GREEN}══════════════════════════════════════════${NC}"
    echo -e "${GREEN}  知伴已启动！${NC}"
    echo -e "${GREEN}══════════════════════════════════════════${NC}"
    echo ""
    echo -e "   ${CYAN}👉 浏览器访问:${NC} ${BOLD}http://localhost:18921${NC}"
    echo ""
    echo -e "   按 ${BOLD}Ctrl+C${NC} 停止服务"
    echo ""

    # 延迟打开浏览器（等服务器就绪）
    (
        sleep 3
        open "http://localhost:18921" 2>/dev/null || true
    ) &

    # 启动 Python 服务
    cd "$SCRIPT_DIR"
    exec "$py" scripts/serve.py
}

# ============================================================
# Main
# ============================================================
banner
check_platform
PYTHON=$(find_python)
install_deps "$PYTHON"
download_models
launch "$PYTHON"
