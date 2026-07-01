"""Agent 工具定义 — 函数式工具，支持并发执行与结果缓存。

每个工具是一个纯函数 + 元数据，agent loop 负责调度。
工具结果会缓存到消息列表中，形成工具调用历史。

设计要点（来自 cc-haha）:
  - isConcurrencySafe: 标记是否可并行执行
  - isReadOnly: 标记是否修改状态
  - 结果截断: 大结果自动摘要，保留 KV cache 友好的大小
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine


@dataclass
class ToolResult:
    """工具执行结果"""
    tool_name: str
    success: bool
    content: str           # 格式化后的文本结果
    raw_data: Any = None   # 原始数据（供后续处理）
    error: str = ""
    duration_ms: float = 0


@dataclass
class AgentTool:
    """Agent 可调用的工具定义

    name: 工具名（用于 function calling 匹配）
    description: 简短描述（拼入 system prompt）
    parameters: JSON Schema 格式的参数定义
    handler: 异步执行函数，接收参数 dict，返回 ToolResult
    is_read_only: True = 不修改外部状态，可安全重试
    is_concurrency_safe: True = 可与其他工具并行执行
    max_result_chars: 结果最大字符数，超此值截断
    """
    name: str
    description: str
    parameters: dict
    handler: Callable[..., Coroutine[Any, Any, ToolResult]]
    is_read_only: bool = True
    is_concurrency_safe: bool = True
    max_result_chars: int = 2000

    def to_openai_schema(self) -> dict:
        """转为 OpenAI function calling 兼容格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_prompt_text(self) -> str:
        """转为纯文本格式（用于不支持 function calling 的本地模型）"""
        params_desc = []
        props = self.parameters.get("properties", {})
        required = self.parameters.get("required", [])
        for name, spec in props.items():
            req_mark = " (必填)" if name in required else ""
            params_desc.append(f"    - {name}: {spec.get('description', '')}{req_mark}")
        params_text = "\n".join(params_desc) if params_desc else "    无参数"
        return f"工具: {self.name}\n描述: {self.description}\n参数:\n{params_text}"


def _truncate_result(text: str, max_chars: int = 2000) -> str:
    """截断过长结果，保留头部信息"""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return (
        text[:half]
        + f"\n\n... (结果过长，已截断，共 {len(text)} 字符) ...\n\n"
        + text[-half:]
    )


# ── 内置工具: 知识库搜索 ──

def create_search_tool(
    vector_search_fn: Callable,
    mmr_rerank_fn: Callable | None = None,
) -> AgentTool:
    """创建知识库搜索工具。

    vector_search_fn(query, top_k) -> list[dict] 其中每个 dict 含:
      content, metadata, distance
    """

    async def handler(
        query: str = "",
        top_k: int = 5,
        **kwargs,
    ) -> ToolResult:
        t0 = time.time()
        try:
            results = await vector_search_fn(query, top_k)
            if not results:
                return ToolResult(
                    tool_name="search",
                    success=True,
                    content="（未找到相关内容）",
                    duration_ms=(time.time() - t0) * 1000,
                )

            # 格式化搜索结果
            parts = [f"搜索「{query}」的结果 ({len(results)} 条):\n"]
            for i, r in enumerate(results[:top_k], 1):
                content = r.get("content", "")[:500]
                meta = r.get("metadata", {})
                paper_id = meta.get("paper_id", "?")
                section = meta.get("section", "")
                dist = r.get("distance", 0)
                relevance = f"相关度: {1 - dist:.2f}" if dist else ""
                loc = f"Paper #{paper_id}" + (f", {section}" if section else "")
                parts.append(f"[{i}] {loc} | {relevance}")
                parts.append(f"    {content}\n")

            result_text = "\n".join(parts)
            result_text = _truncate_result(result_text, max_chars=2000)

            return ToolResult(
                tool_name="search",
                success=True,
                content=result_text,
                raw_data=results,
                duration_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool_name="search",
                success=False,
                content=f"搜索失败: {e}",
                error=str(e),
                duration_ms=(time.time() - t0) * 1000,
            )

    return AgentTool(
        name="search_knowledge_base",
        description="搜索学术论文知识库。当需要查找论文内容、概念解释、实验数据时使用此工具。",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询词，使用中文关键词，应精确描述想查找的内容",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回结果数量，默认5。信息量大时可增加到8-10",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
        handler=handler,
        is_read_only=True,
        is_concurrency_safe=True,
        max_result_chars=2000,
    )


# ── 内置工具: 论文章节获取 ──

def create_paper_section_tool(
    get_section_fn: Callable | None = None,
) -> AgentTool:
    """创建论文章节获取工具。

    get_section_fn(paper_id, section_name) -> str
    """

    async def handler(
        paper_id: str = "",
        section: str = "",
        **kwargs,
    ) -> ToolResult:
        t0 = time.time()
        try:
            if get_section_fn is None:
                return ToolResult(
                    tool_name="get_paper_section",
                    success=False,
                    content="章节获取功能未配置",
                    error="no section function",
                )

            text = await get_section_fn(paper_id, section)
            if not text:
                return ToolResult(
                    tool_name="get_paper_section",
                    success=True,
                    content=f"（Paper #{paper_id} 中未找到「{section}」章节）",
                    duration_ms=(time.time() - t0) * 1000,
                )

            result_text = f"Paper #{paper_id} - {section}:\n\n{_truncate_result(text, max_chars=3000)}"
            return ToolResult(
                tool_name="get_paper_section",
                success=True,
                content=result_text,
                raw_data=text,
                duration_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool_name="get_paper_section",
                success=False,
                content=f"获取章节失败: {e}",
                error=str(e),
            )

    return AgentTool(
        name="get_paper_section",
        description="获取指定论文的特定章节全文。已知 paper_id 和章节名时使用。",
        parameters={
            "type": "object",
            "properties": {
                "paper_id": {
                    "type": "string",
                    "description": "论文编号，如 '1'",
                },
                "section": {
                    "type": "string",
                    "description": "章节名称，如 'Introduction'、'Results'、'Methods'",
                },
            },
            "required": ["paper_id", "section"],
        },
        handler=handler,
        is_read_only=True,
        is_concurrency_safe=True,
        max_result_chars=3000,
    )


# ── 内置工具: 阅读区上下文获取 ──

def create_reading_context_tool(
    get_context_fn: Callable | None = None,
) -> AgentTool:
    """创建阅读区上下文获取工具。

    LLM 在需要了解用户当前阅读位置时调用此工具。
    惰性推送：只有 LLM 主动请求时才返回阅读内容，不随每轮对话自动注入。
    """

    async def handler(**kwargs) -> ToolResult:
        t0 = time.time()
        try:
            if get_context_fn is None:
                return ToolResult(
                    tool_name="get_reading_context",
                    success=True,
                    content="（当前无阅读内容）",
                )
            text = get_context_fn()
            if not text:
                return ToolResult(
                    tool_name="get_reading_context",
                    success=True,
                    content="（当前无阅读内容）",
                )
            result_text = f"【用户当前阅读位置】\n{text[:3000]}"
            return ToolResult(
                tool_name="get_reading_context",
                success=True,
                content=_truncate_result(result_text, max_chars=3000),
                raw_data=text,
                duration_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool_name="get_reading_context",
                success=False,
                content=f"获取阅读上下文失败: {e}",
                error=str(e),
            )

    return AgentTool(
        name="get_reading_context",
        description="获取用户当前正在阅读的论文段落内容。当需要了解用户正在看什么、或者问题涉及「当前阅读内容」时调用。",
        parameters={
            "type": "object",
            "properties": {},
            "required": [],
        },
        handler=handler,
        is_read_only=True,
        is_concurrency_safe=True,
        max_result_chars=3000,
    )
