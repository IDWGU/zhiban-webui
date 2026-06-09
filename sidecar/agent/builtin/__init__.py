"""内置 Agent 注册 — 仿 cc-haha builtInAgents.ts 设计。

知伴内置三个 agent:
  - paper_explorer:  论文搜索专家（只读搜索，对应 cc-haha Explore agent）
  - paper_summarizer: 论文总结专家
  - paper_assistant:  通用论文助手（默认，对应 cc-haha general-purpose agent）
"""

from ..definitions import AgentRegistry
from .paper_explorer import PAPER_EXPLORER_AGENT
from .paper_summarizer import PAPER_SUMMARIZER_AGENT
from .paper_assistant import PAPER_ASSISTANT_AGENT


def register_all_builtin_agents(registry: AgentRegistry) -> None:
    """向 registry 注册所有内置 agent。"""
    registry.register(PAPER_EXPLORER_AGENT)
    registry.register(PAPER_SUMMARIZER_AGENT)
    registry.register(PAPER_ASSISTANT_AGENT)
