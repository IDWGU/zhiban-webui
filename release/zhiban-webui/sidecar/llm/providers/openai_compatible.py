"""OpenAI 兼容 Provider — 支持所有 /v1/chat/completions 端点的 LLM 服务

支持的服务：
- DeepSeek API (https://api.deepseek.com)
- Ollama (http://localhost:11434/v1)
- vLLM (http://localhost:8000/v1)
- llama.cpp server (http://localhost:8080/v1)
- LM Studio (http://localhost:1234/v1)
- 任何兼容 OpenAI API 的服务
"""

import asyncio
import concurrent.futures
import platform
import time
from typing import AsyncIterator

import httpx
from openai import AsyncOpenAI, OpenAI

__all__ = ["OpenAICompatibleProvider"]


def _get_vram_mb() -> float:
    """跨平台 VRAM/进程内存检测 (MB)。仅用于 health 日志。"""
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
        else:
            import pynvml
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            vram = (info.used - info.reserved) / (1024 * 1024)
            if vram > 0:
                return vram
    except Exception:
        pass
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


class OpenAICompatibleProvider:
    """OpenAI 兼容 API Provider

    支持 session 概念（通过 llm_utils._provider_cache 缓存实例实现跨调用复用）
    支持 cache_control 标记（DeepSeek Context Caching, Anthropic Prompt Caching）
    """

    def __init__(
        self,
        *,
        api_key: str = "",
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-v4-pro",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        extra_headers: dict | None = None,
        extra_body: dict | None = None,
        timeout: int = 60,
    ):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._extra_headers = extra_headers or {}
        self._extra_body = extra_body or {}
        self._timeout = timeout
        self._async_client: AsyncOpenAI | None = None
        self._sync_client: OpenAI | None = None

    # === Properties ===

    @property
    def provider_name(self) -> str:
        return "openai_compatible"

    @property
    def is_available(self) -> bool:
        return True  # 只要配置了 base_url 即可用

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def model(self) -> str:
        return self._model

    @property
    def supports_cache_control(self) -> bool:
        """DeepSeek API 支持 Context Caching，通过 cache_control 标记缓存前缀。"""
        return "deepseek" in self._base_url.lower()

    # === cache_control 标记 ===

    def _apply_cache_control(self, messages: list[dict]) -> list[dict]:
        """为消息添加 cache_control 标记。

        DeepSeek Context Caching: 标记 system prompt (L0) 为 ephemeral 缓存块。
        服务端自动缓存前缀匹配的内容，后续请求 prefill 量大幅减少。

        L0 (system prompt) 恒定不变 → 100% 命中缓存。
        L1 (conversation prefix) 累积变化 → 服务端按前缀匹配自然缓存。
        """
        if not self.supports_cache_control:
            return messages

        result = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Only mark system prompt (L0) for caching
            # L0 is the constant system prompt, always at position 0
            if role == "system" and i == 0 and isinstance(content, str) and content:
                result.append({
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                })
            else:
                result.append(msg)

        return result

    # === Client 管理 ===

    def _get_async_client(self) -> AsyncOpenAI:
        if self._async_client is not None:
            return self._async_client
        kwargs = {
            "base_url": self._base_url,
            "timeout": float(self._timeout),
            "max_retries": 1,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        else:
            kwargs["api_key"] = "ollama"  # Ollama 需要任意非空值
        if self._extra_headers:
            kwargs["default_headers"] = self._extra_headers
        self._async_client = AsyncOpenAI(**kwargs)
        return self._async_client

    def _get_sync_client(self) -> OpenAI:
        if self._sync_client is not None:
            return self._sync_client
        kwargs = {
            "base_url": self._base_url,
            "timeout": float(self._timeout),
            "max_retries": 1,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        else:
            kwargs["api_key"] = "ollama"
        if self._extra_headers:
            kwargs["default_headers"] = self._extra_headers
        self._sync_client = OpenAI(**kwargs)
        return self._sync_client

    def reset_clients(self):
        """更换配置后重置客户端缓存"""
        self._async_client = None
        self._sync_client = None

    # === 思考模式映射 ===

    def _build_thinking_body(self, thinking: bool | None) -> dict | None:
        """将通用 thinking 开关翻译为 Provider 特定的 extra_body 参数。

        thinking=True  → 显式启用深度思考（输出 reasoning_content）
        thinking=False → 显式禁用深度思考（仅输出 content）
        thinking=None  → 不发送 thinking 参数，由模型默认行为决定

        各 Provider 的具体参数：
          DeepSeek:  {"thinking": {"type": "enabled"}}  /  {"thinking": {"type": "disabled"}}
          Ollama:   {"chat_template_kwargs": {"enable_thinking": true/false}}
          其他:      无标准参数，返回 None（模型自行决定）
        """
        if thinking is None:
            return None
        if "deepseek" in self._base_url.lower():
            return {"thinking": {"type": "enabled" if thinking else "disabled"}}
        if "11434" in self._base_url or "ollama" in self._base_url.lower():
            return {"chat_template_kwargs": {"enable_thinking": thinking}}
        import logging
        logging.getLogger("zhiban.provider").warning(
            "Thinking mode requested (%s) but provider %s may not support it",
            "enabled" if thinking else "disabled", self._base_url,
        )
        return None

    # === 流式对话 ===

    async def chat_stream(
        self,
        messages: list[dict],
        *,
        model: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        thinking: bool | None = None,
        extra_headers: dict | None = None,
        extra_body: dict | None = None,
        timeout: int = 60,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
    ) -> AsyncIterator[dict]:
        client = self._get_async_client()

        extra = dict(self._extra_body)
        thinking_body = self._build_thinking_body(thinking)
        if thinking_body:
            extra.update(thinking_body)
        if extra_body:
            extra.update(extra_body)

        # Apply cache_control markers for DeepSeek Context Caching (L0 system prompt)
        messages = self._apply_cache_control(messages)

        vram_before = _get_vram_mb()
        t_start = time.time()
        stream = await client.chat.completions.create(
            model=model or self._model,
            messages=messages,
            max_tokens=max_tokens or self._max_tokens,
            temperature=temperature if temperature is not None else self._temperature,
            top_p=top_p if top_p is not None else self._top_p,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=extra if extra else None,
            timeout=float(timeout or self._timeout),
            frequency_penalty=frequency_penalty or 0.0,
            presence_penalty=presence_penalty or 0.0,
            stop=stop or None,
        )

        total_tokens = 0
        had_content = False
        first_token_time = None
        prompt_tokens = 0
        finish_reason = "unknown"
        async for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                fr = getattr(chunk.choices[0], "finish_reason", None)
                if fr:
                    finish_reason = fr
                rc = getattr(delta, "reasoning_content", None)
                if delta.content:
                    if first_token_time is None:
                        first_token_time = time.time()
                    had_content = True
                    total_tokens += 1
                    yield {"type": "token", "token": delta.content}
                if rc:
                    if first_token_time is None:
                        first_token_time = time.time()
                    total_tokens += len(rc)
                    yield {"type": "reasoning_token", "token": rc}
            if chunk.usage:
                total_tokens = chunk.usage.total_tokens
                if chunk.usage.prompt_tokens:
                    prompt_tokens = chunk.usage.prompt_tokens

        total_s = time.time() - t_start
        prefill_ms = round((first_token_time - t_start) * 1000, 1) if first_token_time else None
        decode_ms = round((time.time() - first_token_time) * 1000, 1) if first_token_time else None

        vram_after = _get_vram_mb()
        # VRAM monitoring (log only, per Section 5.2)
        import logging
        logging.getLogger("zhiban.provider").debug(
            "[VRAM] pre=%.0fMB post=%.0fMB", vram_before, vram_after,
        )
        yield {
            "type": "done",
            "total_tokens": total_tokens,
            "finish_reason": finish_reason,
            "health": {
                "timing": {
                    "prefill_ms": prefill_ms,
                    "decode_ms": decode_ms,
                    "decode_per_token_ms": round(decode_ms / total_tokens, 1) if decode_ms and total_tokens else None,
                    "total_ms": round(total_s * 1000, 1),
                },
                "tokens": {
                    "prefill_tokens": prompt_tokens,
                    "output_tokens": total_tokens,
                    "cache_hit_rate": None,  # API mode
                },
                "memory": {
                    "vram_before_mb": round(vram_before, 1),
                    "vram_after_mb": round(vram_after, 1),
                    "vram_mb": round(vram_after, 1),  # alias for convenience
                },
            },
        }

    # === 同步对话 ===

    def chat_sync(
        self,
        messages: list[dict],
        *,
        model: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        thinking: bool | None = None,
        extra_headers: dict | None = None,
        extra_body: dict | None = None,
        timeout: int = 30,
        cancel_event: asyncio.Event | None = None,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
    ) -> tuple[str, dict]:
        client = self._get_sync_client()
        t0 = time.time()

        extra = dict(self._extra_body)
        thinking_body = self._build_thinking_body(thinking)
        if thinking_body:
            extra.update(thinking_body)
        if extra_body:
            extra.update(extra_body)

        # Apply cache_control markers for DeepSeek Context Caching (L0 system prompt)
        messages = self._apply_cache_control(messages)

        vram_before = _get_vram_mb()

        def _call():
            return client.chat.completions.create(
                model=model or self._model,
                messages=messages,
                max_tokens=max_tokens or self._max_tokens,
                temperature=temperature if temperature is not None else self._temperature,
                top_p=top_p if top_p is not None else self._top_p,
                extra_body=extra if extra else None,
                timeout=float(timeout or self._timeout),
                frequency_penalty=frequency_penalty or 0.0,
                presence_penalty=presence_penalty or 0.0,
                stop=stop or None,
            )

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_call)
                while not future.done():
                    try:
                        resp = future.result(timeout=0.5)
                    except concurrent.futures.TimeoutError:
                        if cancel_event and cancel_event.is_set():
                            future.cancel()
                            raise asyncio.CancelledError("Query cancelled")
                        continue

            text = resp.choices[0].message.content or ""
            # Some models (e.g. Qwen optiq) put output only in reasoning_content
            if not text:
                rc = getattr(resp.choices[0].message, 'reasoning_content', None)
                if rc:
                    text = rc
            usage_info = resp.usage
            total_s = round(time.time() - t0, 2)
            prompt_tokens = usage_info.prompt_tokens if usage_info else 0
            completion_tokens = usage_info.completion_tokens if usage_info else 0
            vram_after = _get_vram_mb()
            # VRAM monitoring (log only, per Section 5.2)
            import logging
            logging.getLogger("zhiban.provider").debug(
                "[VRAM] pre=%.0fMB post=%.0fMB", vram_before, vram_after,
            )
            usage = {
                "input": prompt_tokens,
                "output": completion_tokens,
                "elapsed": total_s,
                # Phase 3: structured health fields for API mode
                "health": {
                    "timing": {
                        "prefill_ms": None,  # API mode: can't measure separately in sync
                        "decode_ms": None,
                        "decode_per_token_ms": None,
                        "total_ms": round(total_s * 1000, 1),
                    },
                    "tokens": {
                        "prefill_tokens": prompt_tokens,
                        "output_tokens": completion_tokens,
                        "cache_hit_rate": None,  # API mode: server-side cache unknown
                    },
                    "memory": {
                        "vram_before_mb": round(vram_before, 1),
                        "vram_after_mb": round(vram_after, 1),
                        "vram_mb": round(vram_after, 1),  # alias
                    },
                },
            }
            return text, usage
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"  [provider] sync error: {type(e).__name__}: {e}")
            return "", {"input": 0, "output": 0, "elapsed": round(time.time() - t0, 2),
                         "health": {"timing": {}, "tokens": {"cache_hit_rate": None}, "memory": {}}}

    # === 模型列表 ===

    async def list_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                headers = {"Authorization": f"Bearer {self._api_key or 'ollama'}"}
                if self._extra_headers:
                    headers.update(self._extra_headers)
                resp = await client.get(
                    f"{self._base_url}/v1/models",
                    headers=headers,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    models = [m["id"] for m in data.get("data", [])]
                    models.sort()
                    return models
                return []
        except Exception:
            return []

    # === 连接测试 ===

    async def test_connection(self, model: str = "") -> bool:
        try:
            client = self._get_async_client()
            stream = await client.chat.completions.create(
                model=model or self._model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
                temperature=0.0,
                stream=True,
                timeout=15.0,
            )
            async for _ in stream:
                pass
            return True
        except Exception:
            return False
