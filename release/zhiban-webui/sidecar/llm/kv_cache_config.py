"""KV Cache 配置 — 各层阈值 + 模型 context window 自动检测

n_ctx 策略（按优先级）：
1. LLM_MAX_CONTEXT 环境变量 > 0 → 直接使用
2. GGUF 元数据自动检测 n_ctx_train → min(检测值, 安全上限)
3. 无法检测 → 使用 DEFAULT_N_CTX
"""

import os
import struct
from pathlib import Path

# ===== 上下文缓冲阈值 =====
# ===== 上下文缓冲阈值 =====
COMPRESS_BUFFER = 2000  # 剩余不足此值时触发压缩
WARN_BUFFER = 3000      # 剩余不足此值时警告
BLOCK_BUFFER = 1000     # 剩余不足此值时拒绝新消息（硬截断）

# 压缩触发比例：active_slice tokens > n_ctx * ratio → 启动压缩
COMPRESS_TRIGGER_RATIO = 0.8

# 物理清理阈值
MAX_MESSAGES_BEFORE_CLEANUP = 500
MAX_ARCHIVED_ENTRIES = 20

# KV Cache 精度
KV_CACHE_DTYPE = "f16"

# ===== 上下文窗口配置 =====
DEFAULT_N_CTX = 65536      # 默认 64K
MAX_N_CTX = 262144          # 硬上限 256K
DEFAULT_N_GPU_LAYERS = -1  # 全部层放 GPU（Apple Silicon）
DEFAULT_N_THREADS = 0      # 0 = 自动

# 4B Q4_K_M 模型 (2.5G) + fp16 KV cache (≈90KB/token) 的内存估算:
#   32K ctx:  2.5 + 2.9 = 5.4 GB
#   64K ctx:  2.5 + 5.8 = 8.3 GB
#   128K ctx: 2.5 + 11.5 = 14 GB
#   256K ctx: 2.5 + 23 = 25.5 GB
_CTX_TIERS = [
    (64.0, 262144),  # 64GB+  → 256K
    (32.0, 131072),  # 32GB+  → 128K
    (16.0, 65536),   # 16GB+  → 64K
    (0.0,  32768),   # <16GB  → 32K
]


def _detect_available_memory_gb() -> float:
    """获取可用于 LLM 推理的内存/显存 (GB)。

    检测顺序: NVIDIA CUDA → AMD ROCm → Apple Silicon 统一内存 → 系统 RAM.
    独显设备上 VRAM 是瓶颈；Apple Silicon 上统一内存即显存。
    """
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            # ROCm (AMD) 也走 cuda API，通过 HIP 层兼容
            gpu_name = torch.cuda.get_device_name(0) or ""
            backend = "ROCm" if any(x in gpu_name.lower() for x in ("radeon", "amd", "gfx")) else "CUDA"
            import logging
            logging.getLogger("zhiban.kv_cache").info(
                "%s GPU 显存: %.1f GB (%s)", backend, vram, gpu_name
            )
            return vram
    except Exception:
        pass

    # Apple Silicon: 统一内存架构，系统 RAM = 显存
    try:
        import platform
        if platform.system() == "Darwin" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            ram = _detect_system_ram_gb()
            import logging
            logging.getLogger("zhiban.kv_cache").info(
                "Apple Silicon 统一内存: %.1f GB", ram
            )
            return ram
    except Exception:
        pass

    return _detect_system_ram_gb()


def _detect_system_ram_gb() -> float:
    """获取系统总内存 (GB)。跨平台兼容。"""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass
    try:
        import platform
        if platform.system() == "Darwin":
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
            )
            return int(result.stdout.strip()) / (1024 ** 3)
    except Exception:
        pass
    return 16.0


def _effective_max_ctx() -> int:
    """根据可用内存/显存动态计算安全的最大 n_ctx。

    独显 (NVIDIA/AMD): 用 VRAM 判断
    Apple Silicon: 用统一内存判断
    CPU-only: 用系统 RAM 判断
    """
    mem = _detect_available_memory_gb()
    for threshold, ctx in _CTX_TIERS:
        if mem >= threshold:
            return ctx
    return _CTX_TIERS[-1][1]


def _read_gguf_context_length(model_path: str | Path) -> int | None:
    """轻量读取 GGUF 文件中的 llama.context_length 元数据。

    直接解析文件头部，不加载模型。GGUF 格式：
      magic(4) + version(4) + tensor_count(8) + metadata_kv_count(8)
      → 然后遍历 KV pairs，找 key="llama.context_length"
    """
    try:
        with open(model_path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return None
            version = struct.unpack("<I", f.read(4))[0]
            if version < 1 or version > 3:
                return None
            # skip tensor_count (8 bytes)
            f.read(8)
            kv_count = struct.unpack("<Q", f.read(8))[0]

            for _ in range(kv_count):
                key_len = struct.unpack("<Q", f.read(8))[0]
                key = f.read(key_len).decode("utf-8", errors="replace")
                value_type = struct.unpack("<I", f.read(4))[0]

                if key == "llama.context_length":
                    if value_type == 5:    # INT32
                        return struct.unpack("<i", f.read(4))[0]
                    elif value_type == 6:  # UINT32
                        return struct.unpack("<I", f.read(4))[0]
                    elif value_type == 10:  # FLOAT32
                        return int(struct.unpack("<f", f.read(4))[0])
                    else:
                        return None
                else:
                    # Skip value based on type
                    _skip_gguf_value(f, value_type)
            return None
    except Exception:
        return None


def _skip_gguf_value(f, value_type: int) -> None:
    """跳过 GGUF value 字段。类型码参考 GGUF 规范 v3。"""
    if value_type == 0:    # UINT8
        f.read(1)
    elif value_type == 1:  # INT8
        f.read(1)
    elif value_type == 2:  # UINT16
        f.read(2)
    elif value_type == 3:  # INT16
        f.read(2)
    elif value_type == 4:  # UINT32
        f.read(4)
    elif value_type == 5:  # INT32
        f.read(4)
    elif value_type == 6:  # FLOAT32
        f.read(4)
    elif value_type == 7:  # BOOL
        f.read(1)
    elif value_type == 8:  # STRING
        strlen = struct.unpack("<Q", f.read(8))[0]
        f.read(strlen)
    elif value_type == 9:  # ARRAY
        arr_type = struct.unpack("<I", f.read(4))[0]
        arr_len = struct.unpack("<Q", f.read(8))[0]
        for _ in range(arr_len):
            _skip_gguf_value(f, arr_type)
    elif value_type == 10:  # UINT64
        f.read(8)
    elif value_type == 11:  # INT64
        f.read(8)
    elif value_type == 12:  # FLOAT64
        f.read(8)
    else:
        f.read(8)


def _read_gguf_context_length(model_path: str | Path) -> int | None:
    """轻量读取 GGUF 文件中的 llama.context_length 元数据。

    直接解析文件头部，不加载模型。
    类型码参考 GGUF 规范 v3:
      4=UINT32, 5=INT32, 6=FLOAT32
    """
    try:
        with open(model_path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return None
            version = struct.unpack("<I", f.read(4))[0]
            if version < 1 or version > 3:
                return None
            f.read(8)  # skip tensor_count
            kv_count = struct.unpack("<Q", f.read(8))[0]

            for _ in range(kv_count):
                key_len = struct.unpack("<Q", f.read(8))[0]
                key = f.read(key_len).decode("utf-8", errors="replace")
                value_type = struct.unpack("<I", f.read(4))[0]

                if key == "llama.context_length":
                    if value_type == 4:   # UINT32
                        return struct.unpack("<I", f.read(4))[0]
                    elif value_type == 5:  # INT32
                        val = struct.unpack("<i", f.read(4))[0]
                        return max(val, 0)
                    elif value_type == 6:  # FLOAT32
                        return int(struct.unpack("<f", f.read(4))[0])
                    return None
                _skip_gguf_value(f, value_type)
            return None
    except Exception:
        return None


def resolve_n_ctx(model_path: str | None = None) -> int:
    """解析实际使用的 context 大小。

    优先级:
    1. LLM_MAX_CONTEXT > 0 → 直接使用，超过硬件安全上限时警告并限制
    2. GGUF 元数据检测 n_ctx_train → min(n_ctx_train, effective_max)
    3. 都失败 → min(DEFAULT_N_CTX, effective_max)

    effective_max 根据系统内存自动调整: <12GB → 32K, >=12GB → 64K.
    """
    effective_max = _effective_max_ctx()

    env_val = os.getenv("LLM_MAX_CONTEXT", "0")
    if env_val.strip() and int(env_val) > 0:
        user_val = int(env_val)
        if user_val > effective_max:
            import logging
            logging.getLogger("zhiban.kv_cache").warning(
                "LLM_MAX_CONTEXT=%d 超过系统内存安全上限 %d，已限制",
                user_val, effective_max,
            )
        return min(user_val, effective_max)

    if model_path:
        detected = _read_gguf_context_length(model_path)
        if detected and detected > 0:
            return min(detected, effective_max)

    return min(DEFAULT_N_CTX, effective_max)
