"""知伴自主 Agent 模块 — 基于 cc-haha 设计理念的有限自主决策系统。

核心特性:
  - Agent Loop: while 循环 + 工具调用，AI 自主决策搜索次数
  - Agent Definitions: 声明式 agent 定义，按场景过滤工具和 prompt
  - Agent Memory: 跨会话持久记忆（MEMORY.md）
  - KV Cache 复用: 分层提示词对齐，跨调用共享前缀缓存
  - 统一提示词: concise 风格（实测 2B/4B/9B 均最优），可选 strict/academic/verbose
  - 上下文压缩: snip/micro/auto 三级策略

架构:
  definitions.py  — AgentDefinition + AgentRegistry
  builtin/         — 内置 agent（paper_explorer, paper_summarizer, paper_assistant）
  memory.py       — 跨会话记忆系统
  agent_loop.py   — 核心循环
  tools.py        — 工具定义与执行
  prompts.py      — 分层提示词 + 4 种风格变体
  compression.py  — KV Cache 感知的上下文管理
"""

from .agent_loop import AgentLoop, AgentConfig, AgentResult, AgentStep
from .definitions import AgentDefinition, AgentRegistry
from .memory import load_agent_memory, save_agent_memory, build_memory_prompt
from .prompts import (
    build_agent_system_prompt,
    build_agent_system_prompt_v2,
    AGENT_TOOL_DEFS,
    PROMPT_VARIANTS,
)
from .tools import (
    AgentTool,
    ToolResult,
    create_search_tool,
    create_paper_section_tool,
    create_reading_context_tool,
)

__all__ = [
    # Core
    "AgentLoop",
    "AgentConfig",
    "AgentResult",
    "AgentStep",
    # Definitions
    "AgentDefinition",
    "AgentRegistry",
    # Memory
    "load_agent_memory",
    "save_agent_memory",
    "build_memory_prompt",
    # Tools
    "AgentTool",
    "ToolResult",
    "create_search_tool",
    "create_paper_section_tool",
    "create_reading_context_tool",
    # Prompts
    "build_agent_system_prompt",
    "build_agent_system_prompt_v2",
    "PROMPT_VARIANTS",
    "AGENT_TOOL_DEFS",
]
