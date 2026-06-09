"""知伴自主 Agent 模块 — 基于 cc-haha 设计理念的有限自主决策系统。

核心特性:
  - Agent Loop: while 循环 + 工具调用，AI 自主决策搜索次数
  - KV Cache 复用: 分层提示词对齐，跨调用共享前缀缓存
  - 统一提示词: concise 风格（实测 2B/4B/9B 均最优），可选 strict/academic/verbose
  - 上下文压缩: snip/micro/auto 三级策略
  - 流式工具执行: 工具调用在流式响应中即时触发

架构:
  agent_loop.py   — 核心循环
  tools.py        — 工具定义与执行
  prompts.py      — 分层提示词 + 4 种风格变体
  compression.py  — KV Cache 感知的上下文管理
"""

from .agent_loop import AgentLoop, AgentConfig, AgentResult
from .prompts import build_agent_system_prompt, AGENT_TOOL_DEFS, PROMPT_VARIANTS
from .tools import (
    AgentTool,
    ToolResult,
    create_search_tool,
    create_paper_section_tool,
    create_reading_context_tool,
)

__all__ = [
    "AgentLoop",
    "AgentConfig",
    "AgentResult",
    "AgentTool",
    "ToolResult",
    "build_agent_system_prompt",
    "PROMPT_VARIANTS",
    "AGENT_TOOL_DEFS",
    "create_search_tool",
    "create_paper_section_tool",
    "create_reading_context_tool",
]
