"""本地 LLM 推理引擎 — 支持 GGUF (llama-cpp-python) 和 MLX (mlx-lm) 双后端。

自动检测模型格式：
  - .gguf 文件 → LlamaCppBackend（KV cache 可控：kv_cache_seq_rm / clear）
  - 包含 model.safetensors 的目录 → MLXBackend（Apple Silicon 原生加速）

公共接口（双后端统一）：
  - load() / unload() / reset()
  - chat() / chat_stream()
  - kv_cache_clear()          — 双后端均支持
  - kv_cache_seq_rm()         — 仅 GGUF 支持；MLX 抛 NotImplementedError
  - health_snapshot()         — 跨平台 VRAM + token 统计

KV cache 跨调用复用：持久化 cache 实例，单轮内 L0+L1 跨调用 100% 命中。
"""

from __future__ import annotations

import json
import logging
import socket
import subprocess
import time
import platform
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator

from .kv_cache_config import resolve_n_ctx, DEFAULT_N_GPU_LAYERS

__all__ = ["LocalChatEngine", "HealthSnapshot"]

logger = logging.getLogger("zhiban.local_engine")

# ======================================================================
# VRAM 监测（跨平台，仅日志用）
# ======================================================================

def _get_vram_mb() -> float:
    """当前进程显存/统一内存占用 (MB)。仅用于日志。"""
    try:
        if platform.system() == "Darwin":
            import torch
            if torch.backends.mps.is_available():
                mps_mem = torch.mps.current_allocated_memory()
                if mps_mem > 0:
                    return mps_mem / (1024 * 1024)
            cuda_mem = torch.cuda.memory_allocated()
            if cuda_mem > 0:
                return cuda_mem / (1024 * 1024)
            return _rss_mb()
        else:
            import pynvml
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            vram = (info.used - info.reserved) / (1024 * 1024)
            return vram if vram > 0 else _rss_mb()
    except Exception:
        return _rss_mb()


def _rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


# ======================================================================
# HealthSnapshot — 健康快照
# ======================================================================

class HealthSnapshot:
    """单次 LLM 调用的健康快照"""

    __slots__ = (
        "prefill_ms", "decode_ms", "decode_per_token_ms", "total_ms",
        "prefill_tokens", "output_tokens", "cache_hit_tokens", "cache_miss_tokens",
        "vram_before_mb", "vram_after_mb", "finish_reason",
    )

    def __init__(self):
        self.prefill_ms: float = 0.0
        self.decode_ms: float = 0.0
        self.decode_per_token_ms: float = 0.0
        self.total_ms: float = 0.0
        self.prefill_tokens: int = 0
        self.output_tokens: int = 0
        self.cache_hit_tokens: int = 0
        self.cache_miss_tokens: int = 0
        self.vram_before_mb: float = 0.0
        self.vram_after_mb: float = 0.0
        self.finish_reason: str = "unknown"

    @property
    def cache_hit_rate(self) -> float:
        if self.prefill_tokens <= 0:
            return 0.0
        return self.cache_hit_tokens / self.prefill_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "timing": {
                "prefill_ms": round(self.prefill_ms, 1),
                "decode_ms": round(self.decode_ms, 1),
                "decode_per_token_ms": round(self.decode_per_token_ms, 1),
                "total_ms": round(self.total_ms, 1),
            },
            "tokens": {
                "prefill_tokens": self.prefill_tokens,
                "output_tokens": self.output_tokens,
                "cache_hit_tokens": self.cache_hit_tokens,
                "cache_miss_tokens": self.cache_miss_tokens,
                "cache_hit_rate": round(self.cache_hit_rate, 4),
            },
            "memory": {
                "vram_before_mb": round(self.vram_before_mb, 1),
                "vram_after_mb": round(self.vram_after_mb, 1),
            },
        }


# ======================================================================
# EngineBackend — 抽象后端协议
# ======================================================================

class EngineBackend(ABC):
    """推理后端的抽象接口"""

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def is_loaded(self) -> bool: ...

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        top_p: float,
        repeat_penalty: float,
        stop: list[str] | None,
        thinking: bool | None = None,
    ) -> tuple[str, HealthSnapshot]: ...

    @abstractmethod
    def chat_stream(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
        top_p: float,
        repeat_penalty: float,
        stop: list[str] | None,
        thinking: bool | None = None,
    ) -> Iterator[tuple[bool, str] | HealthSnapshot]: ...

    @abstractmethod
    def kv_cache_clear(self) -> None: ...

    @abstractmethod
    def kv_cache_seq_rm(self, seq_id: int, p0: int, p1: int) -> None: ...

    def truncate_cache(self, keep_tokens: int) -> None:
        """截断 KV cache 到 keep_tokens 位置，移除之后的所有内容。
        默认等价于全清；支持 seq_rm 的后端（GGUF）做精确截断。
        """
        self.kv_cache_clear()

    @abstractmethod
    def count_tokens(self, text: str) -> int: ...

    @abstractmethod
    def tokenize(self, text: str) -> list[int]: ...

    @property
    @abstractmethod
    def n_ctx(self) -> int: ...

    @property
    @abstractmethod
    def n_tokens(self) -> int: ...

    @property
    @abstractmethod
    def backend_name(self) -> str: ...

    @property
    def supports_kv_cache_ops(self) -> bool:
        return True


# ======================================================================
# LlamaCppBackend — llama-cpp-python 直连后端（无子进程，原生 KV cache）
# ======================================================================

class _LlamaCppBackend(EngineBackend):
    """直接使用 llama-cpp-python 的 Llama 实例，不 spawn 子进程。

    同一实例内连续调用自动复用 prompt 前缀的 KV cache，
    无需依赖外部 llama-server。
    """

    def __init__(
        self,
        model_path: Path,
        n_ctx: int,
        n_gpu_layers: int,
        n_threads: int,
        use_mlock: bool,
        verbose: bool,
        flash_attn: bool = True,
        use_mmap: bool = True,
        n_batch: int = 2048,
        n_ubatch: int = 1024,
    ):
        self._model_path = model_path
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._n_threads = n_threads
        self._use_mlock = use_mlock
        self._verbose = verbose
        self._flash_attn = flash_attn
        self._use_mmap = use_mmap
        self._n_batch = n_batch
        self._n_ubatch = n_ubatch
        self._llm = None
        self._n_past = 0

    @property
    def backend_name(self) -> str:
        return "llama_cpp"

    @property
    def n_ctx(self) -> int:
        return self._n_ctx

    @property
    def n_tokens(self) -> int:
        return self._n_past

    @property
    def supports_kv_cache_ops(self) -> bool:
        return True

    def is_loaded(self) -> bool:
        return self._llm is not None

    def load(self) -> None:
        if self.is_loaded():
            return
        from llama_cpp import Llama
        self._llm = Llama(
            model_path=str(self._model_path),
            n_ctx=self._n_ctx,
            n_gpu_layers=self._n_gpu_layers,
            n_threads=self._n_threads,
            use_mlock=self._use_mlock,
            verbose=self._verbose,
            flash_attn=self._flash_attn,
            use_mmap=self._use_mmap,
            n_batch=self._n_batch,
            n_ubatch=self._n_ubatch,
        )
        self._n_past = 0

    def unload(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
            self._n_past = 0

    def kv_cache_clear(self) -> None:
        if self._llm is not None:
            try:
                self._llm._ctx.kv_cache_clear()
            except Exception:
                pass
            self._n_past = 0

    def kv_cache_seq_rm(self, seq_id: int, p0: int, p1: int) -> None:
        if self._llm is not None:
            self._llm._ctx.kv_cache_seq_rm(seq_id, p0, p1)
            self._n_past = min(self._n_past, p0)

    def truncate_cache(self, keep_tokens: int) -> None:
        if self._llm is not None:
            try:
                self._llm._ctx.kv_cache_seq_rm(0, keep_tokens, -1)
                self._n_past = keep_tokens
            except Exception:
                self.kv_cache_clear()

    def count_tokens(self, text: str) -> int:
        if self._llm is None:
            return len(text) // 2
        return len(self._llm.tokenize(text.encode("utf-8"), add_bos=False))

    def tokenize(self, text: str) -> list[int]:
        if self._llm is None:
            return []
        return self._llm.tokenize(text.encode("utf-8"), add_bos=False)

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repeat_penalty: float = 1.0,
        stop: list[str] | None = None,
        thinking: bool | None = None,
    ) -> tuple[str, HealthSnapshot]:
        t0 = time.time()
        vram_before = _get_vram_mb()
        if thinking:
            messages = self._inject_thinking_prompt(messages)
        result = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens if max_tokens else 0,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop=stop or [],
        )
        content = result["choices"][0]["message"]["content"] or ""
        usage = result.get("usage", {})
        timings = result.get("timings", {})

        health = HealthSnapshot()
        health.prefill_tokens = usage.get("prompt_tokens", 0)
        health.output_tokens = usage.get("completion_tokens", 0)
        health.total_ms = (time.time() - t0) * 1000
        health.prefill_ms = timings.get("prompt_eval_time", 0)
        health.decode_ms = timings.get("predicted_ms", 0)
        if health.output_tokens > 0:
            health.decode_per_token_ms = health.decode_ms / health.output_tokens
        health.vram_before_mb = vram_before
        health.vram_after_mb = _get_vram_mb()

        self._n_past = health.prefill_tokens + health.output_tokens
        return content, health

    @staticmethod
    def _inject_thinking_prompt(messages: list[dict]) -> list[dict]:
        """向 messages 注入 thinking 标签指令，引导模型用 <think> 包裹思考。

        策略：
        1. 修改 system 消息追加格式指令
        2. 修改最后一条 user 消息追加简短的格式提醒（近因效应）
        3. 指令不讨论格式本身，防止模型把"遵守格式"当作话题来展开
        """
        sys_instruction = (
            "【输出格式】\n"
            "用 <think> 和 </think> 标签包裹你的推理过程，标签外写正式回答。\n"
            "格式：<think>推理内容</think>\n\n正式回答内容。\n"
            "例如：<think>用户问的是XXX，需要从YYY角度回答</think>\n\n**关键发现**\n正式回答正文。"
        )
        user_reminder = "\n\n<think>"
        msgs = list(messages)
        # System prompt: 追加格式指令
        if msgs and msgs[0].get("role") == "system":
            msgs[0] = {**msgs[0], "content": msgs[0]["content"] + "\n\n" + sys_instruction}
        else:
            msgs.insert(0, {"role": "system", "content": sys_instruction})
        # Last user message: 追加 <think> 开头提示（最直接的格式引导）
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].get("role") == "user":
                msgs[i] = {**msgs[i], "content": msgs[i]["content"] + user_reminder}
                break
        return msgs

    @staticmethod
    def _find_thinking_split(text: str) -> int:
        """回退启发式：在无 <think> 标签时，检测思考→回答的转折点。

        返回分割位置（回答开始的字符索引），-1 表示无法判断。
        """
        import re
        # 优先级 1：思考结束标识词
        conclusion_pattern = re.compile(
            r"(?:直接给出答案[即可。]*|现在(?:开始)?(?:正式)?回答[：:]?"
            r"|以下是(?:最终)?(?:正式)?回答[：:]?"
            r"|开始(?:回答|撰写|输出)[^。\n]*[。]?"
            r"|(?:好的|明白了)[，,]?(?:我(?:来|将|会))?(?:直接)?回答[：:]?"
            r"|让我(?:们|直接)?(?:正式)?回答[：:]?"
            r"|回答如下[：:]?"
            r"|正式回复[：:]?"
            r"|基于以上(?:分析|信息|思考)[，,]?\s*(?:我|现在)?(?:来|将)?回答[：:]?"
            # 小模型自我指示句：说完"让我基于/我来总结/让我来写"之后的内容是正文
            r"|让我(?:基于|根据|来)(?:这些|以上|已有)(?:信息|内容|结果|数据)"
            r"(?:来|进行|撰写|总结|组织|整理)?(?:正式)?(?:回答|总结|摘要)[。]?"
            r"|我来(?:总结|撰写|回答|整理)[^。\n]*[。]?)"
            r"\s*\n",
            re.MULTILINE,
        )
        match = conclusion_pattern.search(text)
        if match:
            return match.end()

        # 优先级 2：**较长标题**(≥5 个非 * 字符) 独立成行
        heading = re.search(r'\n\*\*([^\*]{5,80})\*\*\s*\n', text)
        if heading:
            return heading.start()

        # 优先级 3：双换行后紧跟中文回答引导词
        answer_starters = (
            r"(?:现在|好的|根据|以下是|综上|总结|最终|让我|我来|下面|接下来"
            r"|基于以上|据此|因此|所以|那么|OK|好的|本文"
            r"|综合来看|总结一下|简要来说|简而言之|总而言之"
            r"|首先|第一|首先我要|我会从|我们将从)"
        )
        pattern = re.compile(rf"\n\n\s*(?={answer_starters})", re.MULTILINE)
        match = pattern.search(text)
        if match:
            return match.start()

        # 优先级 4：无引导词的双换行（前文至少 40 字，降低阈值适配短思考）
        dnl_pos = text.find("\n\n")
        if dnl_pos >= 40:
            return dnl_pos

        # 优先级 5：句号/问号 + 换行 + 中文（允许中间有引号等标点）
        boundary = re.search(r'[。？！]\n[""\']?\n?(?=[\u4e00-\u9fff])', text)
        if boundary:
            return boundary.end()

        return -1

    @staticmethod
    def _content_looks_like_thinking(content: str) -> bool:
        """检查内容开头是否像模型的自我对话/思考过程。

        用于 thinking=False 时判断是否需要运行启发式分界，
        防止小模型的推理文本泄露到正文。
        """
        if not content:
            return False
        # 检查前 200 字符内是否含思考模式
        head = content[:200]
        thinking_patterns = [
            '用户问的是', '用户询问', '用户的问题',
            '根据我的', '让我分析', '我需要', '我应该',
            '我是知伴', '我是ZhiBan', '首先我', '让我先',
            '好的我', '明白了我',
        ]
        for p in thinking_patterns:
            pos = head.find(p)
            if pos >= 0 and pos < 200:
                return True
        # 也检查开头（适配 GGUF 多 token chunk）
        starts = ('用户', '根据', '让我', '我需', '我是',
                  '首先', '这是', '但是', '不过', '好的', '明白了')
        if content.startswith(starts):
            return True
        return False

    @staticmethod
    def _split_by_think_tags(content: str, in_think: bool):
        """按 <think>/</think> 标签位置拆分 chunk，yield (is_thinking, segment)。

        标签不在 chunk 边界时（同一 chunk 内同时有 thinking 和 answer），
        精确按标签位置分割，避免整 chunk 被错误归类。
        """
        import re
        parts = re.split(r'(</?think>)', content)
        for part in parts:
            if part == "<think>":
                in_think = True
            elif part == "</think>":
                in_think = False
            elif part:
                yield (in_think, part)

    def chat_stream(
        self,
        messages: list[dict],
        max_tokens: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repeat_penalty: float = 1.0,
        stop: list[str] | None = None,
        thinking: bool | None = None,
    ):
        t0 = time.time()
        vram_before = _get_vram_mb()

        # thinking=True 时注入标签指令
        if thinking:
            messages = self._inject_thinking_prompt(messages)

        stream = self._llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens if max_tokens else 0,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop=stop or [],
            stream=True,
        )
        token_count = 0
        finish_reason = "unknown"
        for chunk in stream:
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content", "")
                reasoning = delta.get("reasoning_content", "")
                fr = choices[0].get("finish_reason", "")
                if fr:
                    finish_reason = fr
                if content:
                    token_count += 1
                    # 不做思考/正文分离，lamacpp 输出什么就推什么
                    yield (False, content)
                elif reasoning:
                    token_count += 1
                    yield (False, reasoning)

        logger.info("chat_stream finished: finish_reason=%s tokens=%d",
                    finish_reason, token_count)
        # yield health at end
        health = HealthSnapshot()
        health.output_tokens = token_count
        health.total_ms = (time.time() - t0) * 1000
        health.vram_before_mb = vram_before
        health.vram_after_mb = _get_vram_mb()
        health.finish_reason = finish_reason
        self._n_past += token_count
        yield health

    def raw_generate(
        self,
        prompt: str,
        max_tokens: int = 0,
        temperature: float = 0.7,
        top_p: float = 1.0,
        repeat_penalty: float = 1.0,
        stop: list[str] | None = None,
    ) -> tuple[str, HealthSnapshot]:
        t0 = time.time()
        result = self._llm(
            prompt,
            max_tokens=max_tokens if max_tokens else 0,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop=stop or [],
        )
        output = result["choices"][0]["text"] or ""
        health = HealthSnapshot()
        health.output_tokens = result.get("usage", {}).get("completion_tokens", 0)
        health.total_ms = (time.time() - t0) * 1000
        self._n_past += health.output_tokens
        return output, health

    def raw_generate_stream(
        self,
        prompt: str,
        max_tokens: int = 0,
        temperature: float = 0.7,
        top_p: float = 1.0,
        repeat_penalty: float = 1.0,
        stop: list[str] | None = None,
    ):
        stream = self._llm(
            prompt,
            max_tokens=max_tokens if max_tokens else 0,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop=stop or [],
            stream=True,
        )
        token_count = 0
        for chunk in stream:
            choices = chunk.get("choices", [])
            if choices:
                text = choices[0].get("text", "")
                if text:
                    token_count += 1
                    yield text
        health = HealthSnapshot()
        health.output_tokens = token_count
        self._n_past += token_count
        yield health


# ======================================================================
# (LlamaServerBackend removed — fully migrated to _LlamaCppBackend)
# ======================================================================

class _MLXBackend(EngineBackend):
    """基于 mlx-lm 的 MLX 推理后端。

    Apple Silicon 原生加速，8GB 统一内存设备友好。
    局限：MLX 不暴露 kv_cache_seq_rm（不支持部分序列移除）。
    """

    def __init__(
        self,
        model_path: Path,
        n_ctx: int,
        n_gpu_layers: int,  # ignored, 兼容参数
        n_threads: int,      # ignored
        use_mlock: bool,     # ignored
        verbose: bool,
        flash_attn: bool = True,     # ignored, MLX already optimized
        use_mmap: bool = True,       # ignored
        n_batch: int = 2048,         # ignored
        n_ubatch: int = 1024,        # ignored
    ):
        self._path = model_path
        self._n_ctx = n_ctx
        self._verbose = verbose
        self._model: Any = None
        self._tokenizer: Any = None
        self._n_past: int = 0

    @property
    def backend_name(self) -> str:
        return "mlx"

    @property
    def n_ctx(self) -> int:
        return self._n_ctx

    @property
    def n_tokens(self) -> int:
        return self._n_past

    @property
    def supports_kv_cache_ops(self) -> bool:
        return False  # MLX 不支持 seq_rm

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        import mlx_lm
        logger.info("Loading MLX: %s", self._path)
        t0 = time.time()
        self._model, self._tokenizer = mlx_lm.load(str(self._path))
        self._n_past = 0
        logger.info("MLX loaded in %.1fs, VRAM: %.0fMB", time.time() - t0, _get_vram_mb())

    def unload(self) -> None:
        del self._model
        del self._tokenizer
        self._model = None
        self._tokenizer = None
        self._n_past = 0
        logger.info("MLX unloaded")

    def kv_cache_clear(self) -> None:
        """MLX: generate_step 每次调用创建新的 Cache，跨调用无持久缓存。重置追踪计数器即可。"""
        self._n_past = 0
        logger.debug("MLX: kv_cache_clear — n_past reset to 0")

    def kv_cache_seq_rm(self, seq_id: int, p0: int, p1: int) -> None:
        """MLX 不支持部分 KV cache 序列移除。"""
        raise NotImplementedError(
            "kv_cache_seq_rm is not supported on MLX backend. "
            "MLX does not expose sequence-level KV cache manipulation. "
            "Use kv_cache_clear() to clear all cache, or switch to a GGUF model."
        )

    def truncate_cache(self, keep_tokens: int) -> None:
        """MLX: 不支持部分截断，全清即可。"""
        self.kv_cache_clear()

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            return len(text) // 3
        return len(self._tokenizer.encode(text))

    def tokenize(self, text: str) -> list[int]:
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded")
        return self._tokenizer.encode(text)

    def _apply_chat_template(self, messages: list[dict], thinking: bool | None = None) -> str:
        """将 OpenAI 格式的 messages 转为模型 prompt 文本。

        thinking 参数会作为 enable_thinking 传入 Jinja chat template。
        仅当模型模板支持 enable_thinking 时生效（如 Qwen OptiQ 系列），
        不支持的模型（如 DeepSeek-V4-Flash 蒸馏版）会静默忽略。
        """
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded")
        try:
            kwargs = {}
            if thinking is not None:
                kwargs["enable_thinking"] = thinking
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **kwargs,
            )
        except Exception:
            # fallback: 手动构建（不包含 thinking 控制）
            parts = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                parts.append(f"<|{role}|>\n{content}")
            parts.append("<|assistant|>\n")
            return "\n".join(parts)

    def chat(
        self, messages, max_tokens=4096, temperature=0.7, top_p=1.0,
        repeat_penalty=1.0, stop=None, thinking=None,
    ) -> tuple[str, HealthSnapshot]:
        if self._model is None:
            raise RuntimeError("Model not loaded")
        import mlx_lm
        from mlx_lm.sample_utils import make_sampler

        # MLX 需要正整数的 max_tokens，0 透传时用剩余上下文作为兜底
        if not max_tokens:
            max_tokens = max(4096, self._n_ctx - self._n_past)

        health = HealthSnapshot()
        health.vram_before_mb = _get_vram_mb()

        # thinking=True 时注入标签指令（chat template 可能不支持 enable_thinking）
        if thinking:
            messages = _LlamaCppBackend._inject_thinking_prompt(messages)

        prompt = self._apply_chat_template(messages, thinking=thinking)
        health.prefill_tokens = len(self._tokenizer.encode(prompt))
        health.cache_hit_tokens = self._n_past
        health.cache_miss_tokens = max(0, health.prefill_tokens - health.cache_hit_tokens)

        t0 = time.time()

        sampler = make_sampler(temp=temperature, top_p=top_p)  # MLX 不支持 repetition_penalty
        response = mlx_lm.generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=self._verbose,
        )

        health.total_ms = (time.time() - t0) * 1000

        output_ids = self._tokenizer.encode(response)
        health.output_tokens = len(output_ids)
        new_evaled = health.cache_miss_tokens + health.output_tokens
        self._n_past += new_evaled

        if health.output_tokens > 0:
            health.decode_ms = health.total_ms * 0.9
            health.prefill_ms = health.total_ms - health.decode_ms
            health.decode_per_token_ms = health.decode_ms / health.output_tokens

        health.vram_after_mb = _get_vram_mb()
        return response, health

    def chat_stream(
        self, messages, max_tokens=4096, temperature=0.7, top_p=1.0,
        repeat_penalty=1.0, stop=None, thinking=None,
    ) -> Iterator[tuple[bool, str] | HealthSnapshot]:
        if self._model is None:
            raise RuntimeError("Model not loaded")
        import mlx_lm
        from mlx_lm.sample_utils import make_sampler

        # MLX 需要正整数的 max_tokens，0 透传时用剩余上下文作为兜底
        if not max_tokens:
            max_tokens = max(4096, self._n_ctx - self._n_past)

        health = HealthSnapshot()
        health.vram_before_mb = _get_vram_mb()

        # thinking=True 时注入标签指令（chat template 可能不支持 enable_thinking）
        if thinking:
            messages = _LlamaCppBackend._inject_thinking_prompt(messages)

        prompt = self._apply_chat_template(messages, thinking=thinking)
        health.prefill_tokens = len(self._tokenizer.encode(prompt))
        health.cache_hit_tokens = self._n_past
        health.cache_miss_tokens = max(0, health.prefill_tokens - health.cache_hit_tokens)

        t0 = time.time()
        token_count = 0

        sampler = make_sampler(temp=temperature, top_p=top_p)
        for token_result in mlx_lm.stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            token_count += 1
            text = token_result.text
            # 不做思考/正文分离，lamacpp 输出什么就推什么
            yield (False, text)

        health.total_ms = (time.time() - t0) * 1000
        health.output_tokens = token_count
        new_evaled = health.cache_miss_tokens + token_count
        self._n_past += new_evaled

        if token_count > 0:
            health.decode_ms = health.total_ms * 0.9
            health.prefill_ms = health.total_ms - health.decode_ms
            health.decode_per_token_ms = health.decode_ms / token_count

        health.vram_after_mb = _get_vram_mb()
        yield health


# ======================================================================
# LocalChatEngine — 门面类
# ======================================================================

def _detect_backend(model_path: Path) -> str:
    """自动检测模型格式：GGUF → llama_cpp, MLX → mlx"""
    if model_path.is_file() and model_path.suffix == ".gguf":
        return "llama_cpp"
    if model_path.name.endswith(".gguf"):
        return "llama_cpp"
    if model_path.is_dir():
        ggufs = sorted([f for f in model_path.rglob("*.gguf") if not f.name.startswith("mmproj-")])
        if ggufs:
            return "llama_cpp"
        if (model_path / "model.safetensors").exists():
            return "mlx"
        if (model_path / "model.safetensors.index.json").exists():
            return "mlx"
    raise ValueError(
        f"Cannot detect model format for: {model_path}. "
        "Expected a .gguf file (llama.cpp) or a directory with model.safetensors (MLX)."
    )


class LocalChatEngine:
    """本地 LLM 推理引擎。

    自动检测模型格式并选择后端：
      - .gguf → llama-cpp-python (full KV cache control)
      - MLX dir → mlx-lm (Apple Silicon native, fast but limited KV cache ops)

    用法:
        engine = LocalChatEngine("model.gguf")
        engine.load()
        text, health = engine.chat([
            {"role": "system", "content": "你是知伴。"},
            {"role": "user", "content": "解释XRD原理"},
        ])
        print(text)
        print(health.to_dict())
        engine.unload()
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        n_ctx: int | None = None,
        n_gpu_layers: int = DEFAULT_N_GPU_LAYERS,
        n_threads: int = 0,
        use_mlock: bool = True,
        verbose: bool = False,
        flash_attn: bool = True,
        use_mmap: bool = True,
        n_batch: int = 2048,
        n_ubatch: int = 1024,
    ):
        self._model_path = Path(model_path)
        # 自动检测 n_ctx：传入 None 时读取 GGUF 元数据
        self._n_ctx = n_ctx if n_ctx is not None else resolve_n_ctx(str(model_path))
        self._backend_type = _detect_backend(self._model_path)

        # 如果是目录且检测为 GGUF，解析到实际的 .gguf 文件
        if self._backend_type == "llama_cpp" and self._model_path.is_dir():
            ggufs = sorted([f for f in self._model_path.rglob("*.gguf")
                          if not f.name.startswith("mmproj-")])
            if ggufs:
                self._model_path = ggufs[0]

        kwargs = dict(
            model_path=self._model_path,
            n_ctx=self._n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            use_mlock=use_mlock,
            verbose=verbose,
            flash_attn=flash_attn,
            use_mmap=use_mmap,
            n_batch=n_batch,
            n_ubatch=n_ubatch,
        )

        if self._backend_type == "llama_cpp":
            self._backend: EngineBackend = _LlamaCppBackend(**kwargs)
        else:
            self._backend = _MLXBackend(**kwargs)

    # === 属性 ===

    @property
    def is_loaded(self) -> bool:
        return self._backend.is_loaded()

    @property
    def model_path(self) -> Path:
        return self._model_path

    @property
    def backend_name(self) -> str:
        return self._backend.backend_name

    @property
    def n_ctx(self) -> int:
        return self._backend.n_ctx

    @property
    def n_tokens(self) -> int:
        return self._backend.n_tokens

    @property
    def supports_kv_cache_ops(self) -> bool:
        """是否支持 kv_cache_seq_rm。GGUF=True, MLX=False。"""
        return self._backend.supports_kv_cache_ops

    # === 生命周期 ===

    def load(self) -> None:
        self._backend.load()

    def unload(self) -> None:
        self._backend.unload()

    def reset(self) -> None:
        """重置：清空 KV cache，不重启进程。MLX 会重新加载模型。"""
        self._backend.kv_cache_clear()
        logger.info("Engine reset: KV cache cleared")

    # === KV Cache 管理 ===

    def kv_cache_clear(self) -> None:
        """清空全部 KV cache。双后端均支持。"""
        self._backend.kv_cache_clear()

    def kv_cache_seq_rm(self, seq_id: int, p0: int, p1: int) -> None:
        """移除序列 [p0, p1) 范围的 KV cache entry。

        仅 GGUF 后端支持。MLX 后端会抛 NotImplementedError。
        用于压缩时清除被压缩掉的旧轮次对应的 KV entry。
        """
        if not self._backend.supports_kv_cache_ops:
            raise NotImplementedError(
                f"kv_cache_seq_rm is not supported on {self.backend_name} backend. "
                "Use a GGUF model for sequence-level KV cache management."
            )
        self._backend.kv_cache_seq_rm(seq_id, p0, p1)

    def truncate_cache(self, keep_tokens: int) -> None:
        """截断 KV cache（全清后由 prefix caching 恢复）。

        ⚠️ 不再使用 seq_rm 精确截断（会导致 RoPE 位置错位）。
        统一使用 kv_cache_clear() 全清，依赖 llama.cpp 在下次调用时
        通过 prefix caching 自动复用共享前缀。
        """
        self._backend.truncate_cache(keep_tokens)
        logger.debug("truncate_cache(keep=%d): delegated to backend", keep_tokens)

    # === 双 Call 模式 KV cache 管理 ===

    def kv_cache_savepoint(self) -> int:
        """记录当前 KV cache 的"还原点"（返回当前 n_tokens）。

        在双 Call 架构中使用：
          Call 1 前 → sp = engine.kv_cache_savepoint()
          Call 1 完成后 → engine.kv_cache_restore(sp)
          Call 2 → 从共享前缀 L0+L1 开始，完全复用 KV cache

        ⚠️ GGUF 精确截断，MLX 全清 + prefix caching 自动恢复。
        """
        return self._backend.n_tokens if self._backend.supports_kv_cache_ops else -1

    def kv_cache_restore(self, savepoint: int) -> None:
        """回滚 KV cache 到 savepoint 位置。

        GGUF: seq_rm 移除 savepoint 之后的所有 entry。
        MLX: 全清 KV cache，依赖下次调用的 prefix caching 自动恢复共享前缀。
        """
        if savepoint <= 0:
            return
        if self._backend.supports_kv_cache_ops:
            current = self._backend.n_tokens
            if current > savepoint:
                try:
                    self._backend.kv_cache_seq_rm(0, savepoint, -1)
                    self._backend._n_past = savepoint
                    logger.info("kv_cache_restore: seq_rm(%d→%d), n_past=0→%d",
                               current, savepoint, savepoint)
                except Exception as e:
                    logger.warning("kv_cache_restore: seq_rm failed (%s), clearing", e)
                    self._backend.kv_cache_clear()
        else:
            # MLX: 全清后依赖 prefix caching 恢复
            self._backend.kv_cache_clear()
            logger.info("kv_cache_restore: MLX backend, cleared for prefix caching")

    # === 工具方法 ===

    def count_tokens(self, text: str) -> int:
        """估算文本 token 数"""
        return self._backend.count_tokens(text)

    def tokenize(self, text: str) -> list[int]:
        """将文本转为 token ID 列表"""
        return self._backend.tokenize(text)

    # === 对话 ===

    def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repeat_penalty: float = 1.0,
        stop: list[str] | None = None,
        thinking: bool | None = None,
    ) -> tuple[str, HealthSnapshot]:
        """同步单轮对话。返回 (回答文本, HealthSnapshot)。"""
        return self._backend.chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop=stop,
            thinking=thinking,
        )

    def chat_stream(
        self,
        messages: list[dict],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repeat_penalty: float = 1.0,
        stop: list[str] | None = None,
        thinking: bool | None = None,
    ):
        """流式单轮对话。yield (is_thinking, text) 元组和最终的 HealthSnapshot。

        for chunk in engine.chat_stream(messages, ...):
            if isinstance(chunk, HealthSnapshot):
                print("Done:", chunk.to_dict())
            else:
                is_thinking, text = chunk
                print(text, end="")
        """
        yield from self._backend.chat_stream(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop=stop,
            thinking=thinking,
        )

    def raw_generate(
        self,
        prompt_text: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repeat_penalty: float = 1.0,
        stop: list[str] | None = None,
    ) -> tuple[str, HealthSnapshot]:
        """使用原始文本 prompt 生成（绕过 chat template）。

        适用于翻译等场景：构造稳定前缀 prompt（如 system + 全文原文 + 指令），
        依赖 llama.cpp 原生 prefix caching 在连续调用间自动复用共享前缀，
        避免 chat template + truncate_cache 组合导致的 RoPE 位置错位。

        Args:
            prompt_text: 完整 prompt 文本（不含 chat template 标记）
            max_tokens: 最大输出 token 数
            temperature: 采样温度
            top_p: 核采样参数
            repeat_penalty: 重复惩罚
            stop: 停止字符串列表

        Returns:
            (generated_text, HealthSnapshot)
        """
        return self._backend.raw_generate(
            prompt_text,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop=stop,
        )

    def raw_generate_stream(
        self,
        prompt_text: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repeat_penalty: float = 1.0,
        stop: list[str] | None = None,
    ):
        """流式版本：使用原始文本 prompt 生成，逐 token 产出。

        用法:
            for chunk in engine.raw_generate_stream(prompt, ...):
                if isinstance(chunk, HealthSnapshot):
                    # 翻译完成
                    health = chunk
                else:
                    # 收到一个 token
                    ...
        """
        yield from self._backend.raw_generate_stream(
            prompt_text,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            stop=stop,
        )

    # === 健康指标 ===

    def health_snapshot(self) -> dict[str, Any]:
        return {
            "model_loaded": self.is_loaded,
            "backend": self.backend_name,
            "model_path": str(self._model_path),
            "n_ctx": self.n_ctx,
            "n_tokens_cached": self.n_tokens,
            "cache_usage_pct": (
                round(self.n_tokens / self.n_ctx * 100, 1)
                if self.n_ctx > 0 else 0
            ),
            "supports_kv_cache_ops": self.supports_kv_cache_ops,
            "vram_mb": round(_get_vram_mb(), 1),
        }

    def get_vram_usage(self) -> dict[str, float]:
        return {"vram_mb": round(_get_vram_mb(), 1)}

    # === Context 监控 ===

    def context_remaining(self) -> int:
        return max(0, self.n_ctx - self.n_tokens)

    def context_usage_ratio(self) -> float:
        return self.n_tokens / self.n_ctx if self.n_ctx > 0 else 0.0
