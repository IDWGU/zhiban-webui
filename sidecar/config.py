"""知伴 Sidecar — 全局配置"""

import os
import sys
from pathlib import Path

# Debug 模式：SIDECAR_DEBUG=1 时启用调试日志
DEBUG = os.getenv("SIDECAR_DEBUG", "0") == "1"

# In PyInstaller bundle, base paths are relative to the executable
if getattr(sys, 'frozen', False):
    _EXE_DIR = Path(sys.executable).parent
else:
    _EXE_DIR = Path(__file__).resolve().parent.parent  # zhiban/
_BUNDLED = _EXE_DIR / "models" / "bundled"

# 打包版：Electron 通过 ZHIBAN_RESOURCES 传入 app bundle Resources 路径
_RESOURCES = os.getenv("ZHIBAN_RESOURCES", "")
if _RESOURCES:
    _SIDECAR_DIST = Path(_RESOURCES) / "sidecar-dist"
    if _SIDECAR_DIST.exists():
        _BUNDLED = _SIDECAR_DIST / "models"

# ===== 路径 =====
KNOWLEDGE_BASE = Path(os.getenv("KNOWLEDGE_BASE", _EXE_DIR / "brain"))
PAPER_TEXTS = KNOWLEDGE_BASE / "paper-texts"
PAPER_READING = KNOWLEDGE_BASE / "paper-reading"
PAPER_LIBRARY = KNOWLEDGE_BASE / "library"
KNOWLEDGE_GRAPH = KNOWLEDGE_BASE / "knowledge-graph.yaml"

# ChromaDB: dev → .chroma/ ; packaged → ~/Library/Application Support/ZhiBan/chroma/
if getattr(sys, 'frozen', False):
    _default_chroma = Path.home() / "Library" / "Application Support" / "ZhiBan" / "chroma"
else:
    _default_chroma = _EXE_DIR / ".chroma"
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_DIR", str(_default_chroma)))

# 模型缓存：优先使用内置捆绑模型，否则回退到 .cache/models/
MODEL_CACHE = Path(os.getenv("MODEL_CACHE", str(_BUNDLED if _BUNDLED.exists() else _EXE_DIR / ".cache" / "models")))
MODEL_CACHE.mkdir(parents=True, exist_ok=True)

# ===== Server =====
WS_HOST = os.getenv("WS_HOST", "0.0.0.0")  # 0.0.0.0 兼容 IPv4/IPv6 混合环境（macOS Electron 可能走 IPv6）
WS_PORT = int(os.getenv("WS_PORT", "18921"))

# ===== RAG =====
# jina-embeddings-v5-text-nano: 768-dim, ~0.5GB, Apple Silicon MPS 友好, 当前默认
# BAAI/bge-m3: 1024-dim, ~2.2GB, 中文检索强, 备选
# KaLM-Embedding-V2.5: HFL 中文优化备选
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "jinaai/jina-embeddings-v5-text-nano")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "10"))
GRAPH_HOPS = int(os.getenv("GRAPH_HOPS", "2"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "5"))

# ===== LLM =====
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai_compatible")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
_raw_llm_base = os.getenv("LLM_BASE_URL", "__local__")
# 安全兜底：ANTHROPIC_BASE_URL (含 /anthropic 后缀) 可能从父进程泄漏，
# 其 Anthropic Messages API 格式与 OpenAICompatibleProvider 的 Chat Completions 不兼容。
# 请求会被发到 .../anthropic/v1/chat/completions → 401。
if _raw_llm_base.endswith("/anthropic"):
    _raw_llm_base = _raw_llm_base.rsplit("/anthropic", 1)[0]
LLM_BASE_URL = _raw_llm_base
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-pro")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "8192"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.05"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.8"))
LLM_MAX_CONTEXT = int(os.getenv("LLM_MAX_CONTEXT", "0"))  # 0=自动检测(GGUF元数据), >0=强制上限
LLM_EXTRA_HEADERS = os.getenv("LLM_EXTRA_HEADERS", "{}")
LLM_EXTRA_BODY = os.getenv("LLM_EXTRA_BODY", "{}")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))

# 反重复参数（小模型循环问题的核心防线）
LLM_REPEAT_PENALTY = float(os.getenv("LLM_REPEAT_PENALTY", "1.2"))  # 1.0=无惩罚, 1.05-1.15 有效抑制小模型重复
LLM_FREQUENCY_PENALTY = float(os.getenv("LLM_FREQUENCY_PENALTY", "0.3"))  # API 模式, -2.0~2.0
LLM_PRESENCE_PENALTY = float(os.getenv("LLM_PRESENCE_PENALTY", "0.4"))  # API 模式, -2.0~2.0
LLM_TOP_K = int(os.getenv("LLM_TOP_K", "40"))  # 0=禁用, 小模型推荐 40-50
LLM_STOP_TOKENS = os.getenv("LLM_STOP_TOKENS", "<|im_end|>,<|endoftext|>")  # 逗号分隔的停止词, Qwen ChatML 格式

# 本地 LLM 模型路径（GGUF 文件或 MLX 模型目录）
# 打包版自动检测内置模型，命令行设置 LLM_MODEL_PATH= 可覆盖
def _resolve_llm_model_path() -> str:
    env_val = os.getenv("LLM_MODEL_PATH", "")
    if env_val:
        return env_val
    # 优先检测内置捆绑模型
    for search_dir in [_BUNDLED / "llm", _EXE_DIR / "models" / "llm"]:
        if search_dir.exists():
            ggufs = sorted([f for f in search_dir.rglob("*.gguf") if not f.name.startswith("mmproj-")])
            if ggufs:
                return str(ggufs[0])
    return ""

LLM_MODEL_PATH = _resolve_llm_model_path()

# 翻译专用模型（固定 Hy-MT2-1.8B，本地模式下自动使用）
# 优先检测内置捆绑，不存在时回退到伴读模型（model_manager.has_translation_model 会处理）
TRANSLATION_MODEL_PATH = os.getenv(
    "TRANSLATION_MODEL_PATH",
    str(_BUNDLED / "translation" / "Hy-MT2-1.8B-Q4_K_M.gguf")
)

# ===== LLM 加载参数（llama.cpp 构造参数，修改后需重新加载模型）=====
LLM_FLASH_ATTN = os.getenv("LLM_FLASH_ATTN", "true").lower() == "true"
LLM_USE_MMAP = os.getenv("LLM_USE_MMAP", "true").lower() == "true"
LLM_N_BATCH = int(os.getenv("LLM_N_BATCH", "2048"))
LLM_N_UBATCH = int(os.getenv("LLM_N_UBATCH", "1024"))

# Backward-compatible aliases (deprecated)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", LLM_API_KEY)
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", LLM_BASE_URL)
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", LLM_MODEL)

# 会话持久化
# - frozen (PyInstaller) / packaged (Electron app bundle): 用可写用户目录
# - dev: 用项目目录
if getattr(sys, 'frozen', False) or _RESOURCES:
    _data_dir = Path.home() / "Library" / "Application Support" / "ZhiBan"
    _data_dir.mkdir(parents=True, exist_ok=True)
else:
    _data_dir = _EXE_DIR
CONVERSATIONS_DB = Path(os.getenv("CONVERSATIONS_DB",
    str(_data_dir / ".conversations" / "zhiban.db")))
# 优先用捆绑的 sidecar.json（含 LLM_BASE_URL=__local__），不存在时才用数据目录
if _RESOURCES:
    _bundled_settings = _SIDECAR_DIST / "sidecar.json"
    _local_settings = _data_dir / ".conversations" / "sidecar.json"
    # 首次启动：复制捆绑配置到可写位置
    if _bundled_settings.exists() and not _local_settings.exists():
        import shutil as _shutil
        _local_settings.parent.mkdir(parents=True, exist_ok=True)
        _shutil.copy2(_bundled_settings, _local_settings)
        print(f"   [config] Copied bundled sidecar.json to {_local_settings}")
    SIDECAR_SETTINGS = _local_settings
else:
    SIDECAR_SETTINGS = _data_dir / ".conversations" / "sidecar.json"
