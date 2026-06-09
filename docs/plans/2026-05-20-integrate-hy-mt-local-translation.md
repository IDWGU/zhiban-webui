# HY-MT1.5-7B 本地翻译模型集成计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将知伴的文献翻译从 DeepSeek API 切换为本地部署的 HY-MT1.5 模型，零 API 成本、离线可用、质量持平或超越云端方案。

**Architecture:** 新增 `system_capability.py` 自动检测硬件配置并决定最佳模型/量化等级；新增 `llm/local_translation_engine.py` 封装 llama-cpp-python 对 GGUF 模型的推理；修改 `translator.py` 根据检测结果选择本地或云端引擎；模型下载支持国内镜像优先。

**Tech Stack:**
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) — Python bindings for llama.cpp，支持 Metal (MPS) 加速
- HY-MT1.5-7B GGUF (Q4_K_M) — 腾讯 WMT25 冠军翻译模型，量化后 ~4.5GB
- HY-MT1.5-1.8B GGUF (Q4_K_M) — 轻量版，量化后 ~1.2GB
- HuggingFace Hub / HF-Mirror / ModelScope — 模型文件托管（国内镜像优先）
- psutil — 系统内存检测

**硬件要求分级:**
- ⭐ 推荐: Apple Silicon + ≥16GB RAM → HY-MT1.5-7B, GPU 全部层
- ✅ 可行: Intel Mac / ≥8GB RAM → HY-MT1.5-1.8B, CPU only
- ❌ 不支持: <8GB RAM → 自动 fallback 到云端 API

---

### Task 0: 系统能力检测

**Files:**
- Create: `sidecar/system_capability.py`

**设计思路:**
在启动时自动检测当前机器的硬件能力，决定：
1. 是否能够运行本地翻译模型
2. 如果能，选择哪个模型变体（7B / 1.8B）
3. 推理参数（GPU 层数、并发数、线程数）

**Step 1: 核心实现**

`sidecar/system_capability.py`:

```python
"""系统硬件能力检测 — 自动决定本地推理配置"""

import os
import logging
from enum import Enum
from dataclasses import dataclass


class CapabilityLevel(Enum):
    UNSUPPORTED = "unsupported"   # <8GB RAM，无法运行本地模型
    LOW = "low"                   # 8-16GB → 1.8B, CPU
    MEDIUM = "medium"             # 16-24GB → 7B, CPU+GPU
    HIGH = "high"                 # 24GB+ → 7B, GPU all layers


@dataclass
class SystemCapability:
    level: CapabilityLevel
    total_ram_gb: float
    is_apple_silicon: bool
    model_variant: str            # "HY-MT1.5-7B" | "HY-MT1.5-1.8B" | ""
    quant: str                    # "Q4_K_M"
    n_gpu_layers: int             # -1=all, 0=none, N=partial
    concurrency: int               # 并发翻译数
    can_run_local: bool
    reason: str                   # 供前端展示的说明文本


def detect_capability() -> SystemCapability:
    """检测系统配置，返回分级结果"""
    total_ram_gb = _get_total_ram_gb()
    is_apple_silicon = _is_apple_silicon()

    if total_ram_gb < 8:
        return SystemCapability(
            level=CapabilityLevel.UNSUPPORTED,
            total_ram_gb=total_ram_gb,
            is_apple_silicon=is_apple_silicon,
            model_variant="",
            quant="",
            n_gpu_layers=0,
            concurrency=0,
            can_run_local=False,
            reason=f"系统内存 {total_ram_gb:.0f}GB < 8GB，无法运行本地翻译模型，自动切换到云端 API",
        )

    if total_ram_gb < 16:
        # 8-16GB: 1.8B 模型，CPU only
        return SystemCapability(
            level=CapabilityLevel.LOW,
            total_ram_gb=total_ram_gb,
            is_apple_silicon=is_apple_silicon,
            model_variant="HY-MT1.5-1.8B",
            quant="Q4_K_M",
            n_gpu_layers=0,
            concurrency=1,
            can_run_local=True,
            reason=f"系统内存 {total_ram_gb:.0f}GB，使用轻量版 1.8B 模型（CPU）",
        )

    if total_ram_gb < 24:
        # 16-24GB: 7B 模型，部分 GPU 层
        gpu_layers = -1 if is_apple_silicon else 16
        return SystemCapability(
            level=CapabilityLevel.MEDIUM,
            total_ram_gb=total_ram_gb,
            is_apple_silicon=is_apple_silicon,
            model_variant="HY-MT1.5-7B",
            quant="Q4_K_M",
            n_gpu_layers=gpu_layers,
            concurrency=1,
            can_run_local=True,
            reason=f"系统内存 {total_ram_gb:.0f}GB，使用 7B 模型（{'GPU' if is_apple_silicon else 'CPU+GPU'}）",
        )

    # 24GB+: 7B 模型，全部 GPU 层
    return SystemCapability(
        level=CapabilityLevel.HIGH,
        total_ram_gb=total_ram_gb,
        is_apple_silicon=is_apple_silicon,
        model_variant="HY-MT1.5-7B",
        quant="Q4_K_M",
        n_gpu_layers=-1,
        concurrency=1 if not is_apple_silicon else 2,
        can_run_local=True,
        reason=f"系统内存 {total_ram_gb:.0f}GB，使用 7B 模型（GPU 加速）",
    )


def _get_total_ram_gb() -> float:
    """获取总物理内存 (GB)"""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        # fallback: 尝试读取系统信息
        try:
            import subprocess
            # macOS
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return int(result.stdout.strip()) / (1024 ** 3)
        except Exception:
            pass
        # 保守估计: 假设配置低
        return 6.0


def _is_apple_silicon() -> bool:
    """检测是否为 Apple Silicon (M 系列)"""
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.optional.arm64"],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0 and result.stdout.strip() == "1"
    except Exception:
        return False
```

**Step 2: 添加依赖**

`sidecar/requirements.txt` 追加：

```txt
# 系统检测
psutil>=6.0.0
```

---

### Task 1: Config & 环境准备

**Files:**
- Modify: `sidecar/config.py:48-54`
- Modify: `sidecar/requirements.txt`
- Create: `sidecar/.env.example`

**Step 1: 添加翻译引擎配置项（含国内镜像）**

修改 `sidecar/config.py`，在 LLM 配置区块下方追加：

```python
# ===== Translation Provider =====
# 自动: 由 system_capability 检测后决定 local/cloud；也可手动指定
TRANSLATION_PROVIDER = os.getenv("TRANSLATION_PROVIDER", "auto")  # "auto" | "local" | "deepseek"
# 模型镜像源：hf (HuggingFace) | hf-mirror (国内镜像) | modelscope
MODEL_MIRROR = os.getenv("MODEL_MIRROR", "hf-mirror")
# 模型文件路径（自动检测后覆盖此值）
LOCAL_TRANSLATION_MODEL_DIR = os.getenv("LOCAL_TRANSLATION_MODEL_DIR", str(MODEL_CACHE))
# llama-cpp-python 推理参数（自动检测后覆盖）
LLAMA_N_CTX = int(os.getenv("LLAMA_N_CTX", "2048"))
LLAMA_N_THREADS = int(os.getenv("LLAMA_N_THREADS", "4"))
LLAMA_N_GPU_LAYERS = int(os.getenv("LLAMA_N_GPU_LAYERS", "-1"))  # -1 = all on GPU
TRANSLATION_CONCURRENCY = int(os.getenv("TRANSLATION_CONCURRENCY", "1"))
```

**Step 2: 添加依赖**

`sidecar/requirements.txt` 追加：

```txt
# Local Translation Engine (HY-MT1.5)
llama-cpp-python>=0.3.0
# Model download
huggingface-hub>=0.27.0
# 系统检测
psutil>=6.0.0
```

注意：`llama-cpp-python` 在 Mac 上需要先安装 CMake：

```bash
brew install cmake
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

**Step 3: 创建 `.env.example`**

```env
# ===== LLM (通用问答) =====
LLM_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-your-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-pro

# ===== Translation Provider =====
# auto: 自动检测硬件后决定 local/cloud；local: 强制本地；deepseek: 强制云端
TRANSLATION_PROVIDER=auto
# 模型镜像源：hf | hf-mirror | modelscope
MODEL_MIRROR=hf-mirror
```

**Step 4: 验证**

```bash
cd sidecar && pip install -r requirements.txt
python -c "from llama_cpp import Llama; print('llama-cpp ready')"
```

Expected: `llama-cpp ready`

---

### Task 2: 模型下载机制（国内镜像优先）

**Files:**
- Create: `sidecar/translation/model_download.py`

**下载策略（优先级）：**
1. 检查本地缓存 → 已存在且有效则直接返回
2. 根据 `MODEL_MIRROR` 配置选择下载源
3. 从选定的镜像源下载，支持断点续传
4. 失败时自动 fallback 到其他镜像

**模型与镜像的映射表：**

| 模型 | HuggingFace | HF-Mirror | ModelScope |
|------|-------------|-----------|------------|
| HY-MT1.5-7B-Q4_K_M | `Mungert/HY-MT1.5-7B-GGUF` | 同 HF (自动换 endpoint) | 需确认 repo 名 |
| HY-MT1.5-1.8B-Q4_K_M | `Mungert/HY-MT1.5-1.8B-GGUF` | 同 HF | 需确认 repo 名 |

**Step 1: 创建下载脚本**

`sidecar/translation/model_download.py`：

```python
"""翻译模型下载器 — 国内镜像优先，多源 fallback"""

import logging
import os
from pathlib import Path
from typing import Optional

from .. import config
from ..system_capability import SystemCapability

# 模型清单：不同硬件等级对应的模型文件
MODEL_REGISTRY = {
    "HY-MT1.5-7B": {
        "repo_id": "Mungert/HY-MT1.5-7B-GGUF",
        "filename": "hy-mt1.5-7b-q4_k_m.gguf",
        "expected_size_gb": 4.5,
    },
    "HY-MT1.5-1.8B": {
        "repo_id": "Mungert/HY-MT1.5-1.8B-GGUF",
        "filename": "hy-mt1.5-1.8b-q4_k_m.gguf",
        "expected_size_gb": 1.2,
    },
}


def _get_hf_endpoint() -> str:
    """根据配置选择 HuggingFace 镜像源"""
    mirror = os.getenv("MODEL_MIRROR", config.MODEL_MIRROR)
    if mirror == "hf-mirror":
        return "https://hf-mirror.com"
    elif mirror == "modelscope":
        # ModelScope 使用独立的 SDK，不走 HF endpoint
        return "https://hf-mirror.com"  # fallback
    return "https://huggingface.co"  # 官方源


def ensure_translation_model(capability: SystemCapability) -> Optional[Path]:
    """
    根据系统能力下载对应模型。
    返回模型文件路径；如果下载失败返回 None。
    """
    if not capability.can_run_local:
        return None

    variant = capability.model_variant
    info = MODEL_REGISTRY.get(variant)
    if not info:
        logging.error(f"Unknown model variant: {variant}")
        return None

    model_path = config.MODEL_CACHE / info["filename"]

    # 检查本地缓存
    if model_path.exists():
        file_size = model_path.stat().st_size
        if file_size > info["expected_size_gb"] * 0.5 * 1e9:  # 至少 50% 才算有效
            logging.info(f"Translation model cached: {model_path} ({file_size/1e9:.1f}GB)")
            return model_path
        else:
            logging.warning(f"Model file incomplete ({file_size/1e9:.1f}GB), re-downloading...")
            model_path.unlink()

    # 尝试从各镜像下载
    endpoints = _get_download_endpoints()
    last_error = None

    for endpoint in endpoints:
        try:
            logging.info(f"Downloading {info['repo_id']}/{info['filename']} from {endpoint} ...")
            from huggingface_hub import hf_hub_download

            # 通过环境变量临时切换 endpoint
            old_endpoint = os.environ.get("HF_ENDPOINT", "")
            os.environ["HF_ENDPOINT"] = endpoint

            try:
                downloaded = Path(hf_hub_download(
                    repo_id=info["repo_id"],
                    filename=info["filename"],
                    cache_dir=config.MODEL_CACHE,
                    resume_download=True,
                ))
                logging.info(f"Download complete: {downloaded} ({downloaded.stat().st_size/1e9:.1f}GB)")
                return downloaded
            finally:
                if old_endpoint:
                    os.environ["HF_ENDPOINT"] = old_endpoint
                else:
                    os.environ.pop("HF_ENDPOINT", None)

        except Exception as e:
            last_error = e
            logging.warning(f"Download from {endpoint} failed: {e}")
            continue

    logging.error(f"All mirrors failed to download model. Last error: {last_error}")
    return None


def _get_download_endpoints() -> list[str]:
    """获取按优先级排列的镜像列表"""
    primary = os.getenv("MODEL_MIRROR", config.MODEL_MIRROR)
    # 主镜像优先，其他作 fallback
    all_mirrors = ["hf-mirror.com", "huggingface.co"]
    preferred = [f"https://{primary}"] if not primary.startswith("http") else [primary]
    rest = [f"https://{m}" for m in all_mirrors if m != primary]
    return preferred + rest
```

**关于 ModelScope 的说明：** 目前 ModelScope 上的 HY-MT1.5 模型未经确认。实测时可验证 `https://modelscope.cn/models` 上是否有对应模型。如果确认存在，可将 `_get_hf_endpoint` 扩展为使用 `modelscope` 的 Python SDK。

---

### Task 3: 本地翻译推理引擎

**Files:**
- Create: `sidecar/llm/local_translation_engine.py`

**设计要点：**
- 接收 `SystemCapability` 作为初始化参数
- 根据检测结果决定模型路径、GPU 层数、并发数
- 提供流式翻译接口，与现有 `chat_stream()` 兼容

**Step 1: 核心实现**

`sidecar/llm/local_translation_engine.py`：

```python
"""本地翻译推理引擎 — llama-cpp-python + HY-MT1.5 GGUF"""

import asyncio
import logging
from typing import AsyncIterator, Optional
from pathlib import Path

from .. import config
from ..system_capability import SystemCapability

_Llama = None


def _get_llama_backend():
    global _Llama
    if _Llama is None:
        from llama_cpp import Llama
        _Llama = Llama
    return _Llama


class LocalTranslationEngine:
    """封装 llama-cpp 推理，提供流式翻译接口"""

    def __init__(self):
        self._model = None
        self._load_error: str | None = None
        self._capability: SystemCapability | None = None

    def load(self, model_path: str | Path, capability: SystemCapability):
        """加载 GGUF 模型，按系统能力配置推理参数"""
        if self._model is not None:
            return

        self._capability = capability
        path = Path(model_path)
        if not path.exists():
            self._load_error = f"Model not found: {path}"
            raise FileNotFoundError(self._load_error)

        Llama = _get_llama_backend()
        logging.info(f"Loading translation model: {path.name} ({capability.level.value})")

        self._model = Llama(
            model_path=str(path),
            n_ctx=config.LLAMA_N_CTX,
            n_threads=config.LLAMA_N_THREADS,
            n_gpu_layers=capability.n_gpu_layers,
            verbose=False,
        )
        logging.info(f"Model loaded (n_ctx={config.LLAMA_N_CTX}, "
                     f"n_gpu_layers={capability.n_gpu_layers})")

    @property
    def is_available(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def capability(self) -> SystemCapability | None:
        return self._capability

    async def translate_stream(
        self,
        text: str,
    ) -> AsyncIterator[str]:
        """流式翻译单句。逐 token yield 译文片段。"""
        if self._model is None:
            raise RuntimeError(f"Model not loaded: {self._load_error}")

        prompt = f"Translate the following segment into Chinese, without additional explanation.\n\n{text}"

        loop = asyncio.get_running_loop()

        def _generate():
            for chunk in self._model.create_completion(
                prompt,
                max_tokens=config.LLM_MAX_TOKENS,
                temperature=0.1,
                top_p=0.3,
                repeat_penalty=1.05,
                stop=["</s>", "\n\n"],
                stream=True,
            ):
                token = chunk["choices"][0].get("text", "")
                if token:
                    yield token

        iterator = _generate()
        while True:
            try:
                token = await loop.run_in_executor(None, next, iterator)
                yield token
            except StopIteration:
                break


# Singleton
local_translation_engine = LocalTranslationEngine()
```

**与旧版的差异（代码中不需说明）：**
- `load()` 新增 `capability` 参数，替代从 config 读取固定值
- 移除了 `context` 参数（上下文由 translator.py 拼入 user_msg 后统一传入）
- 移除了 `translate_sync`（YAGNI，测试可用流式替代）

---

### Task 4: 修改启动流程 — 系统检测 + 模型预下载

**Files:**
- Modify: `sidecar/server.py:32-73`

**Step 1: 在 lifespan 中集成系统检测和模型加载**

修改 `sidecar/server.py`（约第 32-73 行的 lifespan）：

```python
from .system_capability import detect_capability
from .translation.model_download import ensure_translation_model
from .llm.local_translation_engine import local_translation_engine

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _ready, ocr_engine, stt_engine
    # ... 保持已有逻辑（graph, embeddings, ocr, stt）不变 ...

    # 系统能力检测（决定是否启用本地翻译）
    try:
        capability = await asyncio.to_thread(detect_capability)
        if capability.can_run_local:
            print(f"   Local translation: {capability.reason}")
            # 预下载模型（不阻塞启动）
            loop = asyncio.get_running_loop()
            model_path = await loop.run_in_executor(
                None, ensure_translation_model, capability
            )
            if model_path:
                # 预加载模型（后台任务，不阻塞 readiness）
                asyncio.create_task(_preload_translation_model(model_path, capability))
                print(f"   Translation model: downloading/loading in background")
            else:
                print(f"   Translation model: download failed, will use cloud API")
        else:
            print(f"   Local translation: {capability.reason}")
    except Exception as e:
        print(f"   Local translation: detection failed ({e})")

    _ready = True
    yield
    _ready = False


async def _preload_translation_model(model_path: Path, capability):
    """后台预加载翻译模型（不阻塞应用启动）"""
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, local_translation_engine.load, model_path, capability
        )
        print("   Translation model: loaded and ready", flush=True)
    except Exception as e:
        print(f"   Translation model: load failed ({e})", flush=True)
```

**设计决策：**
- 模型预加载放在后台任务（`asyncio.create_task`），不阻塞应用启动
- 如果模型尚未下载完成或加载失败，翻译请求会自动 fallback（见 Task 6）
- `detect_capability` 和 `ensure_translation_model` 都是同步函数，放在 executor 中运行

---

### Task 5: 修改 Translator 支持自动 Fallback

**Files:**
- Modify: `sidecar/translation/translator.py`

**Step 1: 重构 `translate_blocks` 支持 provider 策略**

```python
"""LLM 流式翻译：逐句翻译 + 并发控制（auto/local/deepseek 三模式）"""

import asyncio
import time
import traceback
from typing import Callable, Awaitable

from .extractor import Block
from .. import config
from ..system_capability import detect_capability

TRANSLATION_SYSTEM_PROMPT = """你是一个学术论文翻译助手。请将以下英文学术句子翻译为中文。

翻译要求：
- 保留专业术语的英文原文，如 "X-ray diffraction (XRD)"
- 保留引用标记，如 [1]、[23]
- 保留 LaTeX 数学公式原样
- 翻译准确、符合学术中文表达习惯
- 直接输出译文，不要加任何前缀或解释"""

_CLOUD_CONCURRENCY = 3


async def translate_blocks(
    blocks: list[Block],
    *,
    provider: str = "",
    on_token: Callable[[str, str, bool], Awaitable[None]],
    api_key: str = "",
) -> dict:
    """
    Translate all sentences with controlled concurrency.

    provider: "auto" | "local" | "deepseek" | "" (auto from config)
    - auto: 检测系统能力，本地可用则用本地，否则 fallback 云端
    - local: 强制本地（失败则报错）
    - deepseek: 强制云端
    """
    resolved_provider = await _resolve_provider(provider)
    concurrency = 1 if resolved_provider == "local" else _CLOUD_CONCURRENCY
    sem = asyncio.Semaphore(concurrency)
    t0 = time.time()
    tokens_sent = 0

    async def counted_on_token(sid: str, token: str, is_first: bool):
        nonlocal tokens_sent
        tokens_sent += 1
        await on_token(sid, token, is_first)

    async def translate_one(sentence_id: str, text: str, block: Block, sent_idx: int):
        async with sem:
            context_parts = []
            if sent_idx > 0:
                prev = block.sentences[sent_idx - 1]
                context_parts.append(f"前一句原文：{prev.text}")
                if prev.translation:
                    context_parts.append(f"前一句译文：{prev.translation}")
            if sent_idx < len(block.sentences) - 1:
                nxt = block.sentences[sent_idx + 1]
                context_parts.append(f"下一句原文：{nxt.text}")

            context = "\n".join(context_parts) if context_parts else ""
            user_msg = f"{context}\n\n待翻译句子：\n{text}" if context else text

            if resolved_provider == "local":
                from ..llm.local_translation_engine import local_translation_engine
                is_first = True
                async for token in local_translation_engine.translate_stream(
                    text=user_msg,
                ):
                    await counted_on_token(sentence_id, token, is_first)
                    is_first = False
            else:
                from ..llm.deepseek_proxy import llm_proxy
                is_first = True
                async for chunk in llm_proxy.chat_stream(
                    query=user_msg,
                    context="",
                    system_prompt=TRANSLATION_SYSTEM_PROMPT,
                    thinking=False,
                    api_key=api_key,
                ):
                    if chunk["type"] == "token":
                        await counted_on_token(sentence_id, chunk["token"], is_first)
                        is_first = False

    tasks = []
    for block in blocks:
        for si, s in enumerate(block.sentences):
            tasks.append(translate_one(s.id, s.text, block, si))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    errors = [r for r in results if isinstance(r, Exception)]
    if errors:
        print(f"[translator] {len(errors)}/{len(tasks)} sentences failed:", flush=True)
        for exc in errors[:3]:
            print(f"  {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exception(type(exc), exc, exc.__traceback__)

    total = len(tasks)
    return {"total_sentences": total, "duration_sec": round(time.time() - t0, 1),
            "tokens_sent": tokens_sent, "provider": resolved_provider}


async def _resolve_provider(provider: str) -> str:
    """解析最终使用的 provider"""
    if provider and provider != "auto":
        return provider

    configured = config.TRANSLATION_PROVIDER
    if configured == "local":
        return "local"
    elif configured == "deepseek":
        return "deepseek"

    # auto 模式：检测系统能力
    try:
        cap = await asyncio.to_thread(detect_capability)
        if cap.can_run_local:
            return "local"
    except Exception:
        pass
    return "deepseek"
```

---

### Task 6: 修改 Handler 适配自动模式

**Files:**
- Modify: `sidecar/translation/handler.py:86-128`

**Step 1: 修改 API key 校验和模型加载逻辑**

将 handler.py 中 Phase 2 的 API key 校验段（约第 86-96 行）替换为：

```python
    # Phase 2: Translate with streaming
    t0 = time.time()
    api_key = msg.get("apiKey", "")
    if api_key:
        api_key = api_key.encode("ascii", errors="ignore").decode("ascii").strip()

    # 决策最终 provider
    from .. import config
    provider = msg.get("provider", config.TRANSLATION_PROVIDER)

    if provider == "auto" or provider == "local":
        from ..llm.local_translation_engine import local_translation_engine
        if not local_translation_engine.is_available:
            # 尝试等待预加载完成（最多等 30 秒）
            from ..system_capability import detect_capability
            from .model_download import ensure_translation_model

            try:
                loop = asyncio.get_running_loop()
                capability = await loop.run_in_executor(None, detect_capability)
                if capability.can_run_local:
                    model_path = await loop.run_in_executor(
                        None, ensure_translation_model, capability
                    )
                    if model_path:
                        await loop.run_in_executor(
                            None, local_translation_engine.load, model_path, capability
                        )
            except Exception as e:
                print(f"[handler] Local model load failed: {e}", flush=True)

        # 最终判断
        if not local_translation_engine.is_available:
            if provider == "local":
                # 强制本地但加载失败 → 报错
                await ws.send_json({
                    "type": "status", "level": "error",
                    "code": "translation_error",
                    "message": "本地翻译模型未就绪，请在设置中切换为云端模式或重启应用",
                })
                return
            else:
                # auto 模式 → fallback 到云端
                provider = "deepseek"
                print(f"[handler] Local model unavailable, falling back to cloud API", flush=True)

    if provider == "deepseek":
        # DeepSeek 模式：校验 API key
        from ..llm.deepseek_proxy import llm_proxy
        if not api_key and not llm_proxy.is_available:
            await ws.send_json({
                "type": "status", "level": "error",
                "code": "translation_error",
                "message": "未配置 LLM API Key，请在设置中填入 DeepSeek API Key 或在 sidecar/.env 中设置 DEEPSEEK_API_KEY",
            })
            return
```

**Step 2: 翻译完成后透传 provider 信息**

```python
    await ws.send_json({
        "type": "translation_done",
        "totalBlocks": len(blocks),
        "totalSentences": total_sentences,
        "duration": result["duration_sec"],
        "provider": provider,  # 告知前端实际使用的引擎
    })
```

---

### Task 7: 前端感知（可选增强）

**Files:**
- Modify: `src/components/translation/TranslationToolbar.tsx`
- Modify: `src/stores/appStore.ts`

**Step 1: 后端在 `translation_blocks` 消息中透传 provider**

```python
    await ws.send_json({
        "type": "translation_blocks",
        "blocks": blocks_payload,
        "totalSentences": total_sentences,
        "provider": provider,
    })
```

**Step 2: 前端 Store 记录 provider**

`appStore.ts` 的 `translation_blocks` handler 中记录 `translationProvider`。

**Step 3: Toolbar 显示模式标签和 fallback 提示**

`TranslationToolbar.tsx`：

```tsx
{translationProvider && (
  <span className="text-xs px-2 py-0.5 rounded bg-gray-100 text-gray-500">
    {translationProvider === 'local' ? '本地模型' : '云端 API'}
  </span>
)}
```

如果检测到是因为配置低而 fallback 到云端，可显示提示：

```tsx
{translationFallbackReason && (
  <span className="text-xs text-amber-500" title={translationFallbackReason}>
    ⚡ 当前配置不支持本地模型，已切换至云端
  </span>
)}
```

---

### Task 8: 测试

**Files:**
- Create: `sidecar/tests/test_system_capability.py`
- Create: `sidecar/tests/test_local_translation.py`

**Step 1: 测试系统能力检测**

```python
"""测试系统能力检测模块"""

from ..system_capability import detect_capability, CapabilityLevel


def test_detect_runs_without_error():
    """检测函数应正常执行"""
    cap = detect_capability()
    assert cap.total_ram_gb > 0
    assert cap.level in CapabilityLevel
    assert cap.can_run_local == (cap.level != CapabilityLevel.UNSUPPORTED)
    assert len(cap.reason) > 0


def test_detect_apple_silicon_returns_bool():
    """Apple Silicon 检测应返回布尔值"""
    cap = detect_capability()
    assert isinstance(cap.is_apple_silicon, bool)
```

Run: `cd sidecar && python -m pytest tests/test_system_capability.py -v`
Expected: PASS

**Step 2: 测试模型下载模块**

```python
"""测试模型下载模块"""

from ..translation.model_download import MODEL_REGISTRY, _get_download_endpoints


def test_registry_has_7b():
    assert "HY-MT1.5-7B" in MODEL_REGISTRY
    assert MODEL_REGISTRY["HY-MT1.5-7B"]["filename"].endswith(".gguf")


def test_registry_has_1_8b():
    assert "HY-MT1.5-1.8B" in MODEL_REGISTRY
    assert MODEL_REGISTRY["HY-MT1.5-1.8B"]["filename"].endswith(".gguf")


def test_download_endpoints_prioritize_configured():
    endpoints = _get_download_endpoints()
    assert len(endpoints) >= 2  # 至少主镜像 + fallback
    assert all(e.startswith("https://") for e in endpoints)
```

Run: `cd sidecar && python -m pytest tests/test_local_translation.py -v`
Expected: PASS

**Step 3: 测试本地引擎初始化**

```python
"""测试本地翻译引擎"""

from ..llm.local_translation_engine import LocalTranslationEngine
from ..system_capability import SystemCapability, CapabilityLevel


def test_engine_init():
    engine = LocalTranslationEngine()
    assert engine.is_available is False
    assert engine.load_error is None


def test_translate_without_load_raises():
    engine = LocalTranslationEngine()
    import pytest
    with pytest.raises(RuntimeError, match="not loaded"):
        # 需要实际异步调用，只测试异常触发
        pass
```

---

### Task 9: 集成冒烟测试

**Step 1: 端到端验证**

```python
"""冒烟测试：实际加载模型并翻译短句"""

import asyncio
from ..system_capability import detect_capability
from ..translation.model_download import ensure_translation_model
from ..llm.local_translation_engine import local_translation_engine


async def test_smoke_translate():
    """完整链路：检测 → 下载 → 加载 → 翻译"""
    capability = detect_capability()
    assert capability.can_run_local, "This test requires a capable machine"

    model_path = ensure_translation_model(capability)
    assert model_path is not None, "Model download failed"
    assert model_path.exists()

    local_translation_engine.load(model_path, capability)
    assert local_translation_engine.is_available

    result = ""
    async for token in local_translation_engine.translate_stream("Hello world"):
        result += token

    assert len(result) > 0
    print(f"Translation: {result}")
```

Run: `cd sidecar && python -c "import asyncio; asyncio.run(test_smoke_translate())"`
Expected: 输出翻译结果

---

## 执行顺序

| 步骤 | 依赖 | 预计耗时 |
|------|------|---------|
| Task 0: 系统检测 | 无 | 15 min |
| Task 1: Config | 无 | 10 min |
| Task 2: 模型下载 | Task 0 | 20 min |
| Task 3: 推理引擎 | Task 1 | 30 min |
| Task 4: 启动流程 | Task 0, 2, 3 | 15 min |
| Task 5: Translator | Task 0, 3 | 15 min |
| Task 6: Handler | Task 5 | 15 min |
| Task 7: 前端感知 | Task 6 | 10 min |
| Task 8: 测试 | Task 0, 2, 3 | 15 min |
| Task 9: 集成冒烟 | Task 2, 3 | 20 min |

**总预计：~2.5 小时**

---

## 回滚方案

1. `TRANSLATION_PROVIDER=deepseek` 即可秒切回云端，无需代码改动
2. 如果 `auto` 模式下检测到配置不足，自动 fallback 到云端，用户无感
3. 模型文件在 `~/.cache/zhiban/models/`，删除即可释放磁盘

---

## 注意事项

1. **国内镜像策略**：默认 `MODEL_MIRROR=hf-mirror`，下载失败时自动尝试 huggingface.co 官方源。支持通过环境变量随时切换
2. **模型选择自动决策**：
   - <8GB → 不尝试本地模型，直接 fallback 云端
   - 8-16GB → 下载 1.8B 版本，CPU only
   - 16-24GB → 下载 7B 版本，Apple Silicon 全 GPU 层，Intel Mac 部分 GPU 层
   - 24GB+ → 下载 7B 版本，全 GPU 层
3. **首次启动体验**：模型下载 + 加载 ~1-5 分钟（取决于网速），在此期间如有翻译请求会自动 fallback 到云端
4. **Prompt 格式**：HY-MT1.5 是指令微调模型，必须用 `Translate the following segment into Chinese, without additional explanation.\n\n{text}` 格式，不可用 ChatML
5. **ModelScope 扩展**：如果确认 ModelScope 上有对应模型，可在 `model_download.py` 中增加 `modelscope` 作为下载源，使用 `snapshot_download` 或 `modelscope.hub.file_download`
