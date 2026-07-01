"""Agent 定义系统 — 仿 cc-haha loadAgentsDir.ts + builtInAgents.ts 设计。

Agent 定义是声明式的：描述 agent 的角色、可用工具、system prompt。
AgentRegistry 管理所有已注册 agent，支持按 agent_type 查找。

设计要点（来自 cc-haha）:
  - AgentDefinition: 不可变声明，包含角色、工具过滤、system prompt 生成器
  - AgentRegistry: 优先级合并（built-in < project < user），支持同名覆盖
  - 工具过滤: tools（白名单）和 disallowed_tools（黑名单）两级过滤
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class AgentDefinition:
    """Agent 定义 — 声明 agent 的角色、工具和提示词。

    cc-haha 对应: AgentDefinition in loadAgentsDir.ts
    """
    agent_type: str
    description: str
    get_system_prompt: Callable[[], str]
    tools: list[str] | None = None          # 白名单，None = 全部可用
    disallowed_tools: list[str] = field(default_factory=list)
    model: str | None = None                # 模型覆盖，None = 继承父级
    source: str = "built-in"               # built-in | project | user
    memory_scope: str | None = None         # user | project | local
    skills: list[str] = field(default_factory=list)
    max_turns: int = 5

    def allows_tool(self, tool_name: str) -> bool:
        """检查是否允许使用指定工具。"""
        if tool_name in self.disallowed_tools:
            return False
        if self.tools is not None:
            return tool_name in self.tools
        return True


class AgentRegistry:
    """Agent 注册中心 — 管理所有已注册的 agent 定义。

    cc-haha 对应: getAgentDefinitionsWithOverrides() in loadAgentsDir.ts

    优先级: user > project > built-in（后注册的同名 agent 覆盖先注册的）
    """

    def __init__(self):
        self._agents: dict[str, AgentDefinition] = {}

    def register(self, definition: AgentDefinition) -> None:
        """注册 agent 定义。同名 agent 后注册的覆盖先注册的。"""
        self._agents[definition.agent_type] = definition

    def get(self, agent_type: str) -> AgentDefinition | None:
        """按类型名获取 agent 定义。"""
        return self._agents.get(agent_type)

    def list_all(self) -> list[AgentDefinition]:
        """列出所有已注册 agent。"""
        return list(self._agents.values())

    def get_tool_prompt(self) -> str:
        """生成 Agent 工具的 system prompt 描述。

        用于让主 LLM 了解可用的 sub-agent 类型及何时使用。
        cc-haha 对应: getPrompt() in prompt.ts
        """
        if not self._agents:
            return ""
        lines = ["【可用 Sub-Agent】"]
        for ad in self._agents.values():
            lines.append(f"- {ad.agent_type}: {ad.description}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, agent_type: str) -> bool:
        return agent_type in self._agents
