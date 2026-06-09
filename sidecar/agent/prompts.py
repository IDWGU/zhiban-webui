"""分层提示词构建 — 统一长版，多风格变体可选。

分层结构:
  L0: 静态 System Prompt（恒定，KV cache 永久复用）
  L1: 对话历史（append-only，前缀恒定 → 前缀缓存命中）
  L2: 动态上下文（每轮变化，放末尾 → 不影响前缀缓存）
  L3: 当前用户消息 + 工具调用结果

提示词变体:
  concise  — 默认版，精简指令，少说教多行动（实测最优）
  strict   — 强调"先搜再答"，减少凭空回答
  academic — 学术风格，强调引用和来源
  verbose  — 旧版默认，行为规则 + 工具说明（偏啰嗦）
"""

import re
from datetime import datetime
from typing import Sequence

from .tools import AgentTool


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _build_tool_prompt(tools: Sequence[AgentTool], *, native_tools: bool = False) -> str:
    """构建工具描述。

    native_tools=False (本地模式):
      输出 <tool_call> XML 格式文本指令，模型通过文本输出调用工具。

    native_tools=True (API 模式):
      不输出 XML 格式指令（API 原生 tools 参数已提供 function calling），
      仅列出可用工具及用途，帮助模型决策何时搜索。
    """
    if not tools:
        return ""

    if native_tools:
        # cc-haha style: tools are sent as structured JSON schemas via API.
        # Prompt only tells WHEN to use them, not HOW.
        lines = []
        for t in tools:
            props = t.parameters.get("properties", {})
            param_names = list(props.keys())
            param_str = ", ".join(param_names) if param_names else "无参数"
            lines.append(f"- {t.name}({param_str}): {t.description}")
        lines.append("")
        lines.append("需要信息时调用对应函数。不确定就搜索，信息足够后直接回答。中文回答。")
        return "\n".join(lines)

    # ── 本地模式: 文本 XML 格式 ──
    tool_texts = [t.to_prompt_text() for t in tools]
    tools_desc = "\n\n".join(tool_texts)

    # 用第一个真实工具构建具体示例
    first_tool = tools[0]
    first_param = ""
    props = first_tool.parameters.get("properties", {})
    if props:
        first_param = next(iter(props.keys()))
        first_param_val = {
            "query": "电化学催化剂 d带中心",
            "paper_id": "1",
            "section": "introduction",
        }.get(first_param, "具体值")
    example = (
        f'{{"name":"{first_tool.name}","arguments":{{'
        f'"{first_param}":"{first_param_val}"'
        f'}}}}'
    ) if first_param else f'{{"name":"{first_tool.name}","arguments":{{}}}}'

    return f"""【工具调用格式 — 必须严格遵守】
收到问题后先搜索。只输出一个 <tool_call> 块，不要输出任何其他文字：

<tool_call>
{example}
</tool_call>

{tools_desc}

规则：
- 先搜再答，禁止不搜就直接回答
- 只输出 <tool_call> 块，不要加任何解释、前缀、后缀
- JSON 必须合法：双引号，不换行"""


# ═══════════════════════════════════════════════════════
# 提示词变体
# ═══════════════════════════════════════════════════════

def _variant_default(tools_str: str, knowledge_brief: str) -> str:
    """默认版 — 平衡行为规则和工具指导"""
    base = f"""你是知伴（ZhiBan），论文伴读助手，可搜索学术论文知识库获取信息。

【规则】
1. 不确定时先搜索知识库，不要编造
2. 基于搜索结果回答，引用格式：【来源: Paper #编号, 章节】
3. 知识库没有的信息直接说"暂无相关信息"
4. 搜索不到换关键词重试，最多 3 次
5. 信息足够后直接回答，不要多余搜索
6. 中文回答，专业术语保留英文

【日期】{_today()}"""
    if knowledge_brief:
        base += f"\n\n【知识库】{knowledge_brief}"
    if tools_str:
        base += f"\n\n{tools_str}"
    return base


def _variant_strict(tools_str: str, knowledge_brief: str) -> str:
    """严格版 — 强制搜索优先，减少自由发挥"""
    base = f"""你是知伴（ZhiBan），论文伴读助手。你只能基于知识库搜索的结果回答，绝对不编造任何信息。

【强制规则】
1. 收到问题后，必须先搜索知识库。即使你觉得自己知道答案，也必须搜索验证
2. 如果搜索结果为空或不相关，换关键词重新搜索，最多 3 次
3. 只有确认搜索结果覆盖了问题，才能开始回答
4. 回答必须引用具体来源：【来源: Paper #编号, 章节】
5. 知识库完全没有相关信息时，回复"知识库中暂无相关信息"，然后可以补充你的常识性理解（需标注"以下为常识补充"）
6. 中文回答，术语保留英文
7. 简洁直接，不要复述你的能力或重复问题

【日期】{_today()}"""
    if knowledge_brief:
        base += f"\n\n【知识库】{knowledge_brief}"
    if tools_str:
        base += f"\n\n{tools_str}"
    return base


def _variant_concise(tools_str: str, knowledge_brief: str) -> str:
    """精简版 — 最少规则，正向引导，适合小模型"""
    base = f"""你是知伴（ZhiBan），论文伴读助手。

规则：不确定就搜索。基于结果回答。搜不到说"暂无"。不编造。中文回答。引用：【来源: Paper #编号, 章节】。

回答方式：直接给出结论，用简洁的段落组织。不要解释规则、不要复述问题、不要自我对话。"""

    if knowledge_brief:
        base += f"\n\n知识库: {knowledge_brief}"
    if tools_str:
        if '<tool_call>' not in tools_str:
            # Native tools mode: tools_str is already clean, use as-is
            base += f"\n\n{tools_str}"
        else:
            # Text-based mode: build concise XML format for local models
            first_tool_match = re.search(r'工具:\s*(\S+)', tools_str)
            first_tool = first_tool_match.group(1) if first_tool_match else 'search_knowledge_base'
            example = f'{{"name":"{first_tool}","arguments":{{"query":"搜索关键词"}}}}'
            concise_tools = (
                f"需要搜索时只输出一个<tool_call>块，不要加任何文字：\n"
                f"<tool_call>\n{example}\n</tool_call>\n\n"
                f"不需要搜索时直接回答。不确定就搜，搜不到说\"暂无\"。"
            )
            base += f"\n\n{concise_tools}"
    return base


def _variant_academic(tools_str: str, knowledge_brief: str) -> str:
    """学术版 — 强调严谨引用和结构化回答"""
    base = f"""你是知伴（ZhiBan），一位严谨的学术论文伴读助手。你的回答必须基于知识库检索结果，遵循学术规范。

【工作流程】
1. 分析用户问题的核心学术概念
2. 用精准关键词搜索知识库（可多次搜索，角度互补）
3. 评估检索结果的相关性和覆盖度
4. 基于结果撰写结构化回答

【回答规范】
- 引用格式：【来源: Paper #编号, 章节】
- 多篇论文支持同一结论时，合并引用: 【来源: Paper #1, #3】
- 区分"论文结论"和"学界共识"，后者需注明
- 知识库信息不足时，明确指出缺口，不强行回答
- 中文为主，专业术语首次出现时标注英文全称

【日期】{_today()}"""
    if knowledge_brief:
        base += f"\n\n【知识库概况】\n{knowledge_brief}"
    if tools_str:
        base += f"\n\n{tools_str}"
    return base


# ═══════════════════════════════════════════════════════
# 变体注册表
# ═══════════════════════════════════════════════════════

PROMPT_VARIANTS = {
    "concise":  _variant_concise,   # 默认 — 实测 2B/4B/9B 均最优
    "strict":   _variant_strict,
    "academic": _variant_academic,
    "verbose":  _variant_default,   # 旧默认，偏啰嗦，保留用于对比
}


# ═══════════════════════════════════════════════════════
# 公共 API
# ═══════════════════════════════════════════════════════

def build_agent_system_prompt(
    tools: Sequence[AgentTool] | None = None,
    knowledge_brief: str = "",
    variant: str = "concise",
    native_tools: bool = False,
) -> str:
    """构建 Agent 系统提示词。

    variant 可选: concise(默认), strict, academic, verbose
    native_tools=True: API 模式，不包含 XML 格式指令（由 API tools 参数处理）
    """
    tools = tools or []
    tools_str = _build_tool_prompt(tools, native_tools=native_tools) if tools else ""
    builder = PROMPT_VARIANTS.get(variant, _variant_default)
    return builder(tools_str, knowledge_brief)


def build_agent_system_prompt_v2(
    definition,  # AgentDefinition (forward ref, imported lazily)
    tools: Sequence[AgentTool] | None = None,
    knowledge_brief: str = "",
    memory_text: str = "",
    native_tools: bool = False,
    variant: str = "concise",
) -> str:
    """构建 Agent 系统提示词 (V2) — 基于 AgentDefinition。

    与 V1 的区别:
      - 接受 AgentDefinition，从中获取 get_system_prompt()
      - 自动注入 agent memory
      - 工具描述支持 native/XML 两种模式

    cc-haha 对应: getAgentSystemPrompt() in runAgent.ts
    """
    tools = tools or []
    tools_str = _build_tool_prompt(tools, native_tools=native_tools) if tools else ""

    # 获取 agent 定义的基础 system prompt
    base_prompt = definition.get_system_prompt() if definition else ""

    # 注入 memory
    memory_block = ""
    if memory_text and memory_text.strip():
        from .memory import build_memory_prompt
        memory_block = build_memory_prompt(memory_text)

    # 组装
    parts = [base_prompt]
    if knowledge_brief:
        parts.append(f"\n【知识库概况】{knowledge_brief}")
    if tools_str:
        parts.append(f"\n{tools_str}")
    if memory_block:
        parts.append(f"\n{memory_block}")

    return "\n".join(parts)


def build_user_context(
    question: str,
    screen_ctx: str = "",
    screen_changed: bool = False,
    l2_text: str = "",
    current_topic: str = "",
    history_str: str = "",
    wake_records_text: str = "",
    question_type: str = "",
) -> str:
    """构建用户消息前缀（L2 动态上下文）。

    screen_ctx 仅在阅读位置发生变化时才自动注入；否则 LLM 需通过
    get_reading_context 工具主动获取，避免每轮浪费上下文。

    question_type="summary" 时：传递完整论文全文 + 总结专用指令。
    """
    parts = []
    if screen_ctx and screen_changed:
        parts.append(f"【用户当前阅读位置（已更新）】\n{screen_ctx[:500]}")
    if current_topic:
        parts.append(f"【当前话题】{current_topic}")
    if l2_text:
        if question_type == "summary":
            # 总结场景：传递完整论文全文（上限50K字符），不截断
            _l2_limit = 50000
            _display = l2_text[:_l2_limit]
            if len(l2_text) > _l2_limit:
                _display += f"\n\n[全文共 {len(l2_text)} 字，以上为前 {_l2_limit} 字]"
            parts.append(f"【论文全文】\n{_display}")
        else:
            parts.append(f"【论文内容】\n{l2_text[:3000]}")
    if wake_records_text:
        parts.append(f"【历史相关记录】\n{wake_records_text}")
    if history_str:
        parts.append(f"【近期对话】\n{history_str[:800]}")
    if parts:
        # 总结场景：添加专用指令
        if question_type == "summary" and l2_text:
            parts.append(
                "【总结指令】论文全文已在上面提供。请直接基于全文生成总结，"
                "不需要调用搜索工具。按照论文的章节结构（摘要→引言→方法→结果→讨论→结论）"
                "梳理核心要点，每部分提取关键发现。"
            )
        parts.append(f"\n【用户问题】{question}")
    else:
        parts.append(question)
    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════
# 工具调用解析
# ═══════════════════════════════════════════════════════

def parse_tool_call(text: str) -> list[dict]:
    import json, re
    text = _strip_thinking_tags(text)
    tool_calls = []

    xml_pattern = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL | re.IGNORECASE)
    for match in xml_pattern.finditer(text):
        inner = match.group(1).strip()
        try:
            data = json.loads(inner)
            if "name" in data:
                args = data.get("arguments", {})
                if isinstance(args, str):
                    args = _parse_kv_args(args)
                tool_calls.append({"name": data["name"], "arguments": args})
                continue
        except json.JSONDecodeError:
            pass
        parsed = _parse_yaml_tool_call(inner)
        if parsed:
            tool_calls.append(parsed)
    if tool_calls:
        return tool_calls

    unclosed_pattern = re.compile(r'<tool_call>\s*\n(.+?)$', re.DOTALL | re.IGNORECASE | re.MULTILINE)
    for match in unclosed_pattern.finditer(text):
        parsed = _parse_yaml_tool_call(match.group(1).strip())
        if parsed:
            tool_calls.append(parsed)
    if tool_calls:
        return tool_calls

    yaml_pattern = re.compile(r'\[工具调用:\s*(\w+)\]\s*\n(.*?)(?=\[工具调用:|$)', re.DOTALL)
    for match in yaml_pattern.finditer(text):
        args = _parse_kv_args(match.group(2))
        tool_calls.append({"name": match.group(1), "arguments": args})
    return tool_calls


def _strip_thinking_tags(text: str) -> str:
    import re
    # 完整 <think>...</think> 对
    text = re.sub(r'<\s*(?:think|opti-q-think)[^>]*>.*?<\s*/\s*(?:think|opti-q-think)\s*>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # 未闭合的 <think> 块（从 <think> 到下一个 <tool_call> 或文本末尾）
    text = re.sub(r'<\s*(?:think|opti-q-think)[^>]*>.*?(?=<tool_call>|$)', '', text, flags=re.DOTALL | re.IGNORECASE)
    # 孤立的 </think> 闭合标签
    text = re.sub(r'<\s*/\s*(?:think|opti-q-think)\s*>', '', text, flags=re.IGNORECASE)
    # "Thinking Process:" 文本标记
    text = re.sub(r'Thinking Process:[\s\S]*?(?=\n\n|\Z)', '', text, flags=re.IGNORECASE)
    return text.strip()


def _parse_yaml_tool_call(text: str) -> dict | None:
    import re
    name_match = re.search(r'(?:^|\n)\s*name\s*[:：]\s*(\S+)', text, re.IGNORECASE)
    if not name_match:
        return None
    name = name_match.group(1).strip()
    args = {}
    params_match = re.search(r'(?:^|\n)\s*(?:参数|args?|arguments?)\s*[:：]\s*(.+)', text, re.IGNORECASE)
    if params_match:
        args.update(_parse_kv_args(params_match.group(1)))
    for k, v in re.findall(r'(?:^|\n)\s*(\w+)\s*[:：=]\s*(.+)', text, re.IGNORECASE):
        if k.lower() not in ('name', '参数', 'args', 'arguments'):
            v = v.strip().strip('"\'')
            try:
                args[k] = float(v) if '.' in v else int(v)
            except (ValueError, TypeError):
                args[k] = v
    return {"name": name, "arguments": args} if name else None


def _parse_kv_args(text: str) -> dict:
    import re
    args = {}
    for match in re.finditer(r'(?:- )?(\w+)\s*[:：=]\s*(.+)', text, re.IGNORECASE):
        key = match.group(1).strip()
        value = match.group(2).strip().strip('"\'')
        try:
            value = float(value) if '.' in value else int(value)
        except (ValueError, TypeError):
            pass
        args[key] = value
    return args


def is_final_answer(text: str, tools_available: bool = True) -> bool:
    if not text.strip():
        return False
    if not tools_available:
        return True
    import re
    return not bool(re.search(r'<tool_call>', text, re.IGNORECASE))


def detect_loop(text: str, recent_texts: list[str], threshold: float = 0.7) -> bool:
    if not recent_texts:
        return False
    recent = text[-200:]
    for prev in recent_texts[-3:]:
        prev_tail = prev[-200:] if len(prev) > 200 else prev
        if recent == prev_tail:
            return True
    return False


def detect_stream_loop(full_text: str, min_repeat: int = 4, window: int = 50) -> bool:
    """检测流式输出中的重复模式（单次 LLM 调用内循环）。

    两阶段检测：先查整段重复（最可靠），再查滑动窗口。
    """
    if len(full_text) < 600:
        return False
    # 阶段 1：整段重复检测 — 同样的长句出现 >= 3 次
    import re
    long_sents = re.findall(r'[^。！？\n]{40,}', full_text[-len(full_text)//2:])
    sent_counts: dict[str, int] = {}
    for s in long_sents:
        key = s[:40]
        sent_counts[key] = sent_counts.get(key, 0) + 1
        if sent_counts[key] >= 3:
            return True
    # 阶段 2：滑动窗口 — 窗口 50 字符出现 >= 4 次
    tail = full_text[-window * 8:]
    step = max(window // 3, 15)
    seen: dict[str, int] = {}
    for i in range(0, len(tail) - window + 1, step):
        chunk = tail[i:i + window].strip()
        if len(chunk) < 20:
            continue
        seen[chunk] = seen.get(chunk, 0) + 1
        if seen[chunk] >= min_repeat:
            return True
    return False


def detect_chinese_loop(full_text: str) -> bool:
    """检测中文模型特有的循环模式。

    双重判断：长句重复 >= 3 次（最可靠），或犹豫短语密度 > 1.5/百字。
    """
    import re
    if len(full_text) < 400:
        return False
    # 阶段 1：长句重复（与 detect_stream_loop 共享逻辑，但阈值更低）
    long_sents = re.findall(r'[^。！？\n]{40,}', full_text[-len(full_text)//2:])
    sent_counts: dict[str, int] = {}
    for s in long_sents:
        key = s[:40]
        sent_counts[key] = sent_counts.get(key, 0) + 1
        if sent_counts[key] >= 3:
            return True
    # 阶段 2：犹豫短语密度
    hesitation = r'(但等等|不过我应该|所以我的回答|让我再|再仔细看看|我应该诚实地|'
    hesitation += r'实际上.*用户.*可能|根据当前阅读内容.*我需要|基于这篇综述的内容|'
    hesitation += r'我需要基于这篇|用户当前阅读的是|关于引用格式.*我需要)'
    tail = full_text[-len(full_text)//2:]
    count = len(re.findall(hesitation, tail))
    if count >= 5:
        density = count / max(len(tail) / 100, 1)
        if density > 1.5:
            return True
    # 阶段 3：结构化循环——"让我构建回答结构"→分析→重复
    structural_loop = re.findall(r'让我构建回答结构', full_text)
    if len(structural_loop) >= 3:
        return True
    return False


# ═══════════════════════════════════════════════════════
# 答案质量自检
# ═══════════════════════════════════════════════════════

# 正则快速检测：模型表达了工具调用意图但不按 <tool_call> 格式
_NEEDS_TOOL_PATTERNS = [
    r'我需要搜索', r'让我搜索', r'需要检索', r'让我先查',
    r'先搜索', r'应该搜索', r'需要调用.*工具', r'先查一下',
    r'需要.*搜索.*知识', r'先.*检索',
    # 使用具体工具名
    r'应该使用.*工具', r'需要使用.*工具', r'让我调用',
    r'使用\s*(?:search|get_|paper_)\w*', r'调用.*工具来',
    r'我先用', r'先.*调用', r'通过.*工具',
]

# 正则快速检测：答案含明显的模型自我对话/思考泄漏
_THINKING_LEAK_PATTERNS = [
    r'用户问的是', r'用户询问的是', r'根据用户.*问题',
    r'让我分析', r'我应该', r'我需要基于',
    r'首先.*我需要', r'好的.*让我', r'明白了.*我',
    r'我应该使用', r'让我先获取', r'我需要先',
    r'让我来', r'我先看', r'我来回答',
]


def detect_tool_intent(text: str) -> bool:
    """检测模型是否表达了搜索意图但未正确使用 <tool_call> 格式。"""
    import re
    if re.search(r'<tool_call>', text, re.IGNORECASE):
        return False  # 已有 tool_call，不需要干预
    return any(re.search(p, text) for p in _NEEDS_TOOL_PATTERNS)


def detect_thinking_leak(text: str) -> bool:
    """检测答案中是否混入了明显的模型自我对话。"""
    import re
    return any(re.search(p, text) for p in _THINKING_LEAK_PATTERNS)


def build_answer_eval_prompt(question: str, answer: str) -> str:
    """构建答案质量自检 prompt。短 prompt，期望短 JSON 回复。"""
    return (
        f"检查这个回答的质量：\n\n"
        f"问题：{question}\n\n"
        f"回答：{answer[:2000]}\n\n"
        f"回答这3个问题（仅输出JSON，不要输出其他内容）：\n"
        f"1. 回答是否完整？(complete/incomplete)\n"
        f'2. 回答中是否包含AI的自我对话或思考过程？如"我需要搜索""让我分析""用户问的是"等 (yes/no)\n'
        f"3. 是否需要搜索知识库来补充信息？如需，关键词是什么？(null或关键词)\n\n"
        f'{{"completeness":"complete","has_thinking":false,"need_search":null,"clean_answer":null}}'
    )


def parse_eval_json(text: str) -> dict:
    """从 LLM 输出中提取评估 JSON。容错：允许 LLM 在 JSON 前后附加文字。"""
    import json
    import re
    # 尝试提取最外层 JSON 对象
    m = re.search(r'\{[^{}]*"completeness"[^{}]*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    # 回退：用简单的键值正则提取
    result = {}
    for key in ["completeness", "need_search", "clean_answer"]:
        m = re.search(rf'"{key}"\s*:\s*"(.*?)"', text)
        if m:
            result[key] = m.group(1)
    m = re.search(r'"has_thinking"\s*:\s*(true|false)', text, re.IGNORECASE)
    if m:
        result["has_thinking"] = m.group(1).lower() == "true"
    return result


AGENT_TOOL_DEFS: list[dict] = []
