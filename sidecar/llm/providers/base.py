"""LLM Provider 协议 — 所有 Provider 必须实现的接口"""

from typing import Protocol, AsyncIterator, runtime_checkable

__all__ = ["LLMProvider"]


@runtime_checkable
class LLMProvider(Protocol):
    """LLM Provider 协议。

    每个模型接入需要的参数：
    - api_key: API Key，本地模型可留空
    - base_url: API 端点地址
    - model: 模型名称/ID
    - max_tokens: 输出 token 上限
    - temperature: 采样温度
    - top_p: 核采样
    - extra_headers: 自定义 HTTP 头
    - extra_body: 请求体额外参数
    - thinking: 思考/推理模式开关（Provider 自行翻译为具体 API 参数）
    - timeout: 请求超时秒数
    """

    @property
    def provider_name(self) -> str: ...

    @property
    def is_available(self) -> bool: ...

    @property
    def base_url(self) -> str: ...

    async def chat_stream(
        self,
        messages: list[dict],
        *,
        model: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 1.0,
        thinking: bool = False,
        extra_headers: dict | None = None,
        extra_body: dict | None = None,
        timeout: int = 60,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
    ) -> AsyncIterator[dict]:
        """流式对话，逐 token yield {"type": "token", "token": str} / {"type": "done", "total_tokens": int}"""
        ...

    def chat_sync(
        self,
        messages: list[dict],
        *,
        model: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 1.0,
        thinking: bool = False,
        extra_headers: dict | None = None,
        extra_body: dict | None = None,
        timeout: int = 30,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
    ) -> tuple[str, dict]:
        """同步对话，返回 (text, usage_dict)"""
        ...

    async def list_models(self) -> list[str]:
        """列出可用模型"""
        ...

    async def test_connection(self, model: str = "") -> bool:
        """测试连接是否可用"""
        ...
