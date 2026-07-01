"""通用论文助手 Agent — 默认 agent，持有全部工具。

对应 cc-haha general-purpose agent: 全功能、通用场景。
替代当前硬编码的 AgentLoop 角色。
"""

from ..definitions import AgentDefinition


def _get_assistant_prompt() -> str:
    """生成 paper_assistant 的 system prompt。"""
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")

    return f"""你是知伴（ZhiBan），论文伴读助手。你可以搜索学术论文知识库来回答用户问题。

规则：
- 不确定时必须搜索知识库，基于搜索结果回答
- 引用格式：【来源: Paper #编号, 章节】
- 搜索不到说"知识库中暂无相关信息"
- 中文回答，专业术语保留英文
- 直接给出结论

日期: {today}"""


PAPER_ASSISTANT_AGENT = AgentDefinition(
    agent_type="paper_assistant",
    description="通用论文伴读助手。回答用户关于论文的任何问题，自主决定搜索策略。默认 agent。",
    get_system_prompt=_get_assistant_prompt,
    tools=None,  # 全部工具可用
    disallowed_tools=[],
    source="built-in",
    max_turns=5,
)
