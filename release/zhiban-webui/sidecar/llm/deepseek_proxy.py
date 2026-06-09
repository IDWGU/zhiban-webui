"""通用 LLM 代理 — 流式对话 + 恒定 System Prompt，委托给 Provider

System Prompt 为纯常量，不受任何外部因素影响。
所有动态信息（日期、话题）通过 L3 调用参数传入，不由 SP 携带。
"""

import json
import asyncio
from typing import AsyncIterator
from .providers import get_provider, LLMProvider
from .. import config

# 恒定 System Prompt — L0 永远命中 KV cache
# 针对小模型优化：极简，避免能力复述触发小模型输出冗余
SYSTEM_PROMPT = """你是知伴（ZhiBan），论文伴读助手。

规则：基于知识库回答。引用：【来源: Paper #编号, 章节】。搜不到说"暂无"。不编造。中文回答。"""


def _parse_json_env(val: str) -> dict | None:
    """安全解析 JSON 环境变量"""
    if not val or val == "{}":
        return None
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        return None


def build_provider(
    *,
    api_key: str = "",
    base_url: str = "",
    model: str = "",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    top_p: float = 0.95,
    extra_headers: dict | None = None,
    extra_body: dict | None = None,
    timeout: int = 60,
) -> LLMProvider:
    """构建 Provider 实例，优先使用传入参数，fallback 到全局 config"""
    return get_provider(
        provider_type=config.LLM_PROVIDER,
        api_key=api_key or config.LLM_API_KEY,
        base_url=base_url or config.LLM_BASE_URL,
        model=model or config.LLM_MODEL,
        max_tokens=max_tokens or config.LLM_MAX_TOKENS,
        temperature=temperature if temperature is not None else config.LLM_TEMPERATURE,
        top_p=top_p if top_p > 0 else config.LLM_TOP_P,
        extra_headers=extra_headers or _parse_json_env(config.LLM_EXTRA_HEADERS),
        extra_body=extra_body or _parse_json_env(config.LLM_EXTRA_BODY),
        timeout=timeout or config.LLM_TIMEOUT,
    )


class LLMProxy:
    """通用 LLM 流式对话代理 (首次调用时才初始化客户端)"""

    def __init__(self):
        self._provider: LLMProvider | None = None
        self._init_error: str | None = None

    def _get_provider(self) -> LLMProvider:
        if self._provider is not None:
            return self._provider
        if self._init_error:
            raise RuntimeError(self._init_error)
        try:
            self._provider = build_provider()
        except Exception as e:
            self._init_error = str(e)
            raise RuntimeError(self._init_error)
        return self._provider

    @property
    def is_available(self) -> bool:
        try:
            return self._get_provider().is_available
        except RuntimeError:
            return False

    async def chat_stream(
        self,
        query: str,
        context: str = "",
        history: list[dict] | None = None,
        api_key: str = "",
        system_prompt: str | None = None,
        model: str = "",
        thinking: bool | None = None,
        base_url: str = "",
    ) -> AsyncIterator[dict]:
        """流式对话，逐 token yield."""
        prompt = system_prompt or SYSTEM_PROMPT
        messages = [{"role": "system", "content": prompt}]

        if history:
            messages.extend(history)

        user_content = query
        if context:
            user_content = (
                f"【当前屏幕上下文 — 用户正在阅读以下段落】\n{context}\n\n"
                f"【用户提问】\n{query}"
            )
        messages.append({"role": "user", "content": user_content})

        # Per-request API key → use per-request provider; otherwise use global
        if api_key or base_url:
            provider = build_provider(api_key=api_key, base_url=base_url)
        else:
            provider = self._get_provider()

        async for chunk in provider.chat_stream(
            messages,
            model=model,
            thinking=thinking,
        ):
            yield chunk


# Singleton
llm_proxy = LLMProxy()
