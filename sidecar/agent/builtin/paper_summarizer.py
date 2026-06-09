"""论文总结 Agent — 结构化总结专家。

持有 get_paper_overview + get_paper_section，用于生成论文全文总结。
不持有 search_knowledge_base（总结场景论文全文已提供，不需要搜索）。
"""

from ..definitions import AgentDefinition

SUMMARIZER_SYSTEM_PROMPT = """你是知伴论文总结专家。你的职责是对学术论文进行结构化总结。

【核心规则】
1. 按照论文的章节结构组织总结：摘要→引言→方法→结果→讨论→结论
2. 每部分提取关键发现，标注来源章节
3. 引用格式：【来源: Paper #编号, 章节】
4. 中文回答，专业术语保留英文
5. 客观总结，不加个人评价

【输出结构】
## 论文总结: {标题}

### 研究背景与问题
- 研究领域和核心问题
- 前人工作的不足

### 方法与实验
- 关键方法/技术路线
- 实验设计要点

### 核心发现
- 主要结果（量化数据优先）
- 与已有工作的对比

### 结论与贡献
- 作者的核心论点
- 对领域的贡献

### 局限与展望
- 作者指出的不足
- 未来研究方向"""


def _get_summarizer_prompt() -> str:
    return SUMMARIZER_SYSTEM_PROMPT


PAPER_SUMMARIZER_AGENT = AgentDefinition(
    agent_type="paper_summarizer",
    description="论文总结专家。当需要生成论文的结构化总结时使用。持有论文全文访问权限。",
    get_system_prompt=_get_summarizer_prompt,
    tools=[
        "get_paper_overview",
        "get_paper_section",
    ],
    disallowed_tools=[],
    source="built-in",
    max_turns=3,
)
