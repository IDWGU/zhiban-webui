"""LLM Provider 工厂 — 根据配置创建 Provider 实例"""

from .base import LLMProvider
from .openai_compatible import OpenAICompatibleProvider

__all__ = [
    "LLMProvider",
    "OpenAICompatibleProvider",
    "get_provider",
]


def get_provider(
    provider_type: str = "openai_compatible",
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
) -> LLMProvider:
    """根据 provider_type 创建 Provider 实例。

    当前仅支持 openai_compatible（覆盖 DeepSeek / Ollama / vLLM / LM Studio 等）。
    后续可扩展 anthropic / google 等 Provider。
    """
    if provider_type == "openai_compatible":
        return OpenAICompatibleProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            extra_headers=extra_headers,
            extra_body=extra_body,
            timeout=timeout,
        )
    raise ValueError(f"Unknown provider type: {provider_type}")
