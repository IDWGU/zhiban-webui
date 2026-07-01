"""V11 决策+回答 prompt 构建器 — LLM 有限决断工作流。

两阶段架构:
  Phase 1: build_decision_prompt_l3() → LLM 分析上下文，产出搜索关键词
  Phase 2: build_eval_answer_prompt_l3() → LLM 评估检索结果 + 流式回答

保留 V10 兼容函数: build_classify_prompt_l3(), build_answer_prompt_l3()
"""

from datetime import datetime


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _current_year() -> str:
    return str(datetime.now().year)


# 分类指令模板（纯文本片段，拼入 L3）
CLASSIFY_INSTRUCTION = """【分类指令】
分析用户问题，仅输出 JSON。不要输出任何解释文字。

mode: "chat"(闲聊) | "deep"(学术/论文/继续讨论)
topic: 中文话题概括(2-8字), topic_switch: bool
is_term_lookup: bool(仅"xxx是什么"时为true)
is_meta: bool, is_recommendation: bool
ref_number: int|null(参考文献编号,绝大部分情况为null)
search_terms: 2-3个中文关键词
corrected_question: 空字符串或修正后的问题

⚠️ 对话流控制词("继续/接着说/然后呢/还有吗")→mode="deep",
如果【用户阅读】非空，search_terms应从用户阅读的段落内容中提取关键词（而非仅从当前话题），
corrected_question应引导模型聚焦用户选中的段落内容

⚠️ 上下文中的"用户阅读"通常是参考资料，但当用户配合流控词("详细说说/展开讲讲"等)使用时，应视为用户的关注焦点，从中提取search_terms

仅输出JSON。"""


def build_classify_prompt_l3(
    question: str,
    screen_ctx: str = "",
    history_str: str = "",
    current_topic: str = "",
    open_papers_str: str = "",
    is_first: bool = False,
    wake_records: str = "",
    l2_text: str = "",
) -> str:
    """构建 L3 分类消息文本（作为 user message 拼入 LLM 调用）。

    l2_text 放在固定位置以与 answer prompt 对齐 KV cache 前缀。
    """
    parts = [f"【用户提问】{question}"]
    parts.append(f"【当前日期: {_today()}】")

    # L2 阅读内容 — 与 build_answer_prompt_l3 同位置，确保 KV cache 前缀对齐
    if l2_text:
        parts.append(f"【阅读内容】\n{l2_text}")

    if wake_records:
        parts.append(f"【已唤醒的对话记录】\n{wake_records}")

    # 上下文信息
    ctx_parts = []
    if screen_ctx:
        ctx_parts.append(f"【用户阅读】{screen_ctx[:500]}")
    if history_str:
        ctx_parts.append(f"【近期对话】{history_str[:500]}")
    if current_topic:
        ctx_parts.append(f"【当前话题】{current_topic}")
    if open_papers_str:
        ctx_parts.append(f"【已打开论文】{open_papers_str[:300]}")
    if is_first:
        ctx_parts.append("【首条消息】")

    if ctx_parts:
        parts.append("【上下文】\n" + "\n".join(ctx_parts))

    parts.append(CLASSIFY_INSTRUCTION)

    return "\n\n".join(parts)


def build_answer_prompt_l3(
    question: str,
    screen_ctx: str = "",
    search_context: str = "",
    wake_records: str = "",
    current_topic: str = "",
    l2_text: str = "",
    question_type: str = "analytical",
) -> str:
    """构建 L3 回答消息文本（不含分类指令）。

    question_type 控制回答风格:
      concise — 简短直接，不需结构化
      pedagogical — 通俗解释，面向初学者
      conversational — 自然对话，给建议和看法
      analytical — 深度分析（默认），RAG + 结构化
    """
    # 序言：日期 + 背景
    parts = [f"【当前日期: {_today()}】"]

    if wake_records:
        parts.append(f"【已唤醒的对话记录】\n{wake_records}")

    if screen_ctx:
        parts.append(f"【用户阅读（背景参考）】\n{screen_ctx}")

    if search_context:
        parts.append(f"【知识库检索结果】\n{search_context}")

    # L2 论文全文
    if l2_text:
        parts.append(f"【论文全文】\n{l2_text}")

    if current_topic:
        parts.append(f"当前话题: {current_topic}")

    # 用户提问 — 放在所有参考资料之后、指令之前
    # 确保模型在处理了长篇论文后仍能聚焦用户真正的问题
    parts.append(f"【用户提问】{question}")

    # 根据问题类型选择不同的回答指令
    _INSTRUCTIONS = {
        "concise": (
            "用一两句话简要回答以上问题。不需要结构化、不需要展开、不需要总结。"
            "直接给出答案即可。"
            "引用格式【来源: Paper#编号, 文件名, 章节】。"
        ),
        "pedagogical": (
            "请用通俗易懂的方式回答，面向初学者。"
            "从最基本的概念讲起，用简单的语言解释复杂的原理。"
            "避免堆砌术语，每引入一个新概念先解释它是什么。"
            "不需要引用论文。"
        ),
        "conversational": (
            "以自然对话的方式回答。给出你的看法和建议，"
            "就像和一个同事聊天一样。不需要结构化格式，不需要引用论文。"
        ),
        "analytical": (
            "请回答用户问题。注意：上面的资料是参考，不是必读——"
            "只挑选直接相关的信息，果断忽略无关内容。不要试图覆盖所有资料。"
            "用自己的语言组织 1-3 个最核心的论点，讲清楚逻辑关系。"
            "优先使用实验数据和结论部分的信息，引言仅用于交代背景。"
            "引用格式【来源: Paper#编号, 文件名, 章节】。信息不足时请明确说明。"
        ),
        "translation": (
            "请将以上内容翻译为中文。保持原文的学术风格和专业术语的准确性。"
            "不需要添加解释或评论，只输出翻译结果。"
        ),
    }

    # 有 L2 全文 + analytical → 全文优先指令
    # 但如果用户已经引用了特定段落（question 包含聚焦指令），
    # 则用标准 analytical 指令，不要覆盖用户意图
    _question_has_focus = any(kw in question for kw in [
        "请针对该段落", "不要扩展到全文", "聚焦于段落", "不要重复其他编号",
        "用户选中了以下", "用户引用了以下",
    ])
    if l2_text and question_type == "analytical" and not _question_has_focus:
        parts.append(
            "用户正在阅读以上论文全文。请基于全文回答用户问题——"
            "只挑选最相关的 1-3 个论点深入展开，"
            "果断跳过无关章节，不要试图面面俱到。"
            "前面的检索结果仅作补充。"
            "讲清楚逻辑关系，不要逐条罗列。"
            "引用格式【来源: Paper#编号, 文件名, 章节】。信息不足时请明确说明。"
        )
    else:
        parts.append(_INSTRUCTIONS.get(question_type, _INSTRUCTIONS["analytical"]))

    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════
# V11: LLM 有限决断工作流 — Decision + Eval&Answer
# ═══════════════════════════════════════════════════════════════

DECISION_INSTRUCTION = """【搜索决策 — 只输出一行，不要解释】
判断是否需要从知识库检索：

不需要搜索
（论文全文已足够 / 纯寒暄 / 意见询问时用此格式）

搜索关键词: 关键词1, 关键词2, 关键词3
（需要检索具体概念时用此格式，2-3个核心概念即可）

严禁输出任何其他内容（不要解释、不要列举规则、不要分析）。只输出上面两行之一。"""


def build_decision_prompt_l3(
    question: str,
    screen_ctx: str = "",
    l2_text: str = "",
    current_topic: str = "",
    history_str: str = "",
    question_type: str = "analytical",
) -> str:
    """构建 V11 搜索决策 prompt。

    LLM 看到完整上下文后决定: 搜什么（或不需要搜）。
    输出格式: "搜索关键词: k1, k2, k3" 或 "不需要搜索"
    """
    parts = [f"【当前日期: {_today()}】"]

    if screen_ctx:
        parts.append(f"【用户阅读/引用】\n{screen_ctx[:2000]}")

    if l2_text:
        outline = _build_outline(l2_text)
        if outline:
            parts.append(f"【论文概要】\n{outline}")
        else:
            parts.append(f"【论文全文】\n{l2_text[:3000]}")

    if history_str:
        parts.append(f"【近期对话】\n{history_str[:800]}")

    if current_topic:
        parts.append(f"当前话题: {current_topic}")

    parts.append(f"【用户提问】{question}")
    parts.append(DECISION_INSTRUCTION)

    return "\n\n".join(parts)


# ── Eval + Answer 合并指令 ──

EVAL_ANSWER_INSTRUCTION = """请回答用户问题。

重要规则：
1. 如果【知识库检索结果】提供了足够的信息，直接回答。检索结果是参考不是必读——只挑直接相关的，果断忽略无关内容。
2. 如果检索结果不足以回答问题（缺少关键数据、未覆盖用户追问的深度），且你认为需要补充检索，请在回复最开头加上一行：
   [需要补充检索: 关键词1, 关键词2]
   系统会补充检索后让你重新回答。
3. 如果检索结果确实不足但你仍能基于论文全文或常识给出有价值的回答，就直接回答并标注信息局限性。
4. 不要重复对话历史中你已经说过的内容。如果用户追问"详细说说"且引用了你之前的某个论点，请深入展开该论点，不要复读。
5. 如果用户引用了对话内容（标记为"[用户引用了以下"），聚焦于被引用的部分深入论述。
6. 用自己的语言组织 1-3 个最核心的论点，讲清楚逻辑关系。不要逐条罗列。
7. 引用格式【来源: Paper#编号, 文件名, 章节】。信息不足时明确说明。"""


def build_eval_answer_prompt_l3(
    question: str,
    screen_ctx: str = "",
    search_context: str = "",
    l2_text: str = "",
    current_topic: str = "",
    question_type: str = "analytical",
    last_assistant_reply: str = "",
) -> str:
    """构建 V11 评估+回答合并 prompt。

    与 answer prompt 的区别:
    - 不含独立的分类指令
    - 包含检索结果评估指引
    - 包含反重复指令（引用上一轮助理回复）
    - 支持 [需要补充检索:] 信号
    """
    parts = [f"【当前日期: {_today()}】"]

    if screen_ctx:
        parts.append(f"【用户阅读/引用】\n{screen_ctx[:2000]}")

    if search_context:
        parts.append(f"【知识库检索结果】\n{search_context}")

    if l2_text:
        parts.append(f"【论文全文】\n{l2_text}")

    if current_topic:
        parts.append(f"当前话题: {current_topic}")

    # 上一轮助理回复 — 用于反重复和追问上下文
    if last_assistant_reply:
        excerpt = last_assistant_reply[-2000:] if len(last_assistant_reply) > 2000 else last_assistant_reply
        parts.append(
            "【你上次回答的内容（不要重复它，但要理解用户可能引用了其中某一部分进行追问）】\n"
            + excerpt
        )

    parts.append(f"【用户提问】{question}")

    # 选择回答风格指令
    _INSTRUCTIONS = {
        "concise": (
            "用一两句话简要回答以上问题。不需要结构化、不需要展开。"
            "引用格式【来源: Paper#编号, 文件名, 章节】。"
        ),
        "pedagogical": (
            "请用通俗易懂的方式回答，面向初学者。"
            "从最基本的概念讲起，用简单的语言解释复杂的原理。"
        ),
        "conversational": (
            "以自然对话的方式回答。给出你的看法和建议，"
            "就像和一个同事聊天一样。不需要结构化格式。"
        ),
        "analytical": EVAL_ANSWER_INSTRUCTION,
        "translation": "请将以上内容翻译为中文。保持学术风格和术语准确性。",
    }
    parts.append(_INSTRUCTIONS.get(question_type, _INSTRUCTIONS["analytical"]))

    return "\n\n".join(parts)


def _build_outline(full_text: str) -> str:
    """从全文提取论文结构概要（零 LLM 调用，纯字符串匹配）。

    返回空字符串表示未检测到多章节结构。
    """
    import re
    _SEC_RE = re.compile(
        r'^(\d+\.?\s*)?'
        r'(Abstract|摘要|'
        r'Introduction|引言|INTRODUCTION|'
        r'Experimental|实验|Methods?|方法|Materials?|材料|'
        r'Results?\s*(and|\&)\s*Discussion|结果与讨论|'
        r'Results?|结果|Discussion|讨论|'
        r'Conclusion|结论|CONCLUSIONS?|'
        r'Background|背景)'
        r'\s*$',
        re.IGNORECASE,
    )

    def _normalize(name: str) -> str:
        n = name.strip().lower()
        if any(w in n for w in ['abstract', '摘要']):
            return '摘要'
        if any(w in n for w in ['introduction', '引言', 'background', '背景']):
            return '引言'
        if any(w in n for w in ['experimental', '实验', 'method', '方法', 'materials', '材料']):
            return '实验方法'
        if 'results' in n and 'discussion' in n or '结果与讨论' in n:
            return '结果与讨论'
        if 'discussion' in n or '讨论' in n:
            return '讨论'
        if 'results' in n or '结果' in n:
            return '结果'
        if any(w in n for w in ['conclusion', '结论']):
            return '结论'
        return name.strip()

    lines = full_text.split('\n')
    sections: list[tuple[str, str]] = []
    current_name = '正文'
    current_text: list[str] = []

    for line in lines:
        stripped = line.strip()
        m = _SEC_RE.match(stripped)
        if m and len(stripped) < 80:
            if current_text:
                text = ' '.join(current_text).strip()
                if len(text) > 20:
                    sections.append((_normalize(current_name), text))
            current_name = m.group(0) or stripped
            current_text = []
        else:
            current_text.append(line)

    if current_text:
        text = ' '.join(current_text).strip()
        if len(text) > 20:
            sections.append((_normalize(current_name), text))

    if len(sections) <= 1:
        return ""

    parts = ["【论文结构概要】"]
    for sec_name, sec_text in sections:
        preview = sec_text[:150].replace('\n', ' ').strip()
        parts.append(f"- {sec_name}: {preview}...")

    return '\n'.join(parts)


# ═══════════════════════════════════════════════════════════════
# V11.1: 小模型简化 Prompt — 2B 模型无法遵循多层指令，需极简格式
# ═══════════════════════════════════════════════════════════════

TINY_DECISION_INSTRUCTION = """只输出一行：
不需要搜索
或
搜索关键词: 词1, 词2"""


def build_tiny_decision_prompt(
    question: str,
    screen_ctx: str = "",
    l2_text: str = "",
    current_topic: str = "",
) -> str:
    """≤2B 模型专用：极简决策 prompt。去掉所有上下文细节，只给问题+指令。"""
    parts = []
    if l2_text:
        outline = _build_outline(l2_text)
        if outline:
            parts.append(outline)
        else:
            parts.append(f"论文: {l2_text[:1000]}")
    if screen_ctx:
        parts.append(f"用户阅读: {screen_ctx[:800]}")
    if current_topic:
        parts.append(f"话题: {current_topic}")
    parts.append(f"问题: {question}")
    parts.append(TINY_DECISION_INSTRUCTION)
    return "\n\n".join(parts)


TINY_ANSWER_INSTRUCTION = """直接回答以上问题。不要复述指令、不要解释你在做什么、不要说你看到了什么资料。只输出回答内容。"""


def build_tiny_answer_prompt(
    question: str,
    screen_ctx: str = "",
    search_context: str = "",
    l2_text: str = "",
    current_topic: str = "",
    recent_context: str = "",
) -> str:
    """≤2B 模型专用：极简回答 prompt。单层指令，防止背诵 prompt。"""
    parts = []
    if screen_ctx:
        parts.append(f"用户引用:\n{screen_ctx[:2000]}")
    if search_context:
        parts.append(f"参考资料:\n{search_context[:1500]}")
    if l2_text:
        parts.append(f"论文:\n{l2_text[:2000]}")
    if current_topic:
        parts.append(f"话题: {current_topic}")
    if recent_context:
        parts.append(f"最近对话:\n{recent_context}")
    parts.append(f"问题: {question}")
    parts.append(TINY_ANSWER_INSTRUCTION)
    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════
# 指令背诵检测 — 安全网：所有模型通用
# ═══════════════════════════════════════════════════════════════

# 模型开始背诵 prompt 指令的特征模式
_INSTRUCTION_ECHO_PATTERNS = [
    "用户问的是",
    "用户提问",
    "用户希望",
    "你是知伴",
    "论文伴读助手",
    "知识库没有的信息",
    "知识库检索",
    "知识库没有的信息直接说",
    "请回答用户问题",
    "直接回答以上问题",
    "用自己的语言组织",
    "【搜索决策",
    "【分类指令",
    "当前日期",
    "【用户提问",
    "【用户阅读",
    "【论文全文",
    "【知识库检索结果",
    "引用格式【来源",
    "不要逐条罗列",
    "不要扩展到全文",
    "不要复述指令",
    "不要解释你在做什么",
]

_ECHO_PATTERN_RE = None


def _get_echo_re():
    global _ECHO_PATTERN_RE
    if _ECHO_PATTERN_RE is None:
        import re
        _ECHO_PATTERN_RE = re.compile(
            '|'.join(re.escape(p) for p in _INSTRUCTION_ECHO_PATTERNS)
        )
    return _ECHO_PATTERN_RE


def strip_reasoning_preamble(text: str) -> str:
    """剥离 2B 模型回答前的推理独白。

    小模型输出模式: "用户询问X...我应该...回答如下：\n\n实际内容"
    去掉开头的元认知部分，只保留实际回答。
    """
    if not text or len(text) < 30:
        return text

    import re

    # 模式1: "用户询问/用户问的是/用户要求..." 开头的推理前缀
    # 匹配到第一个实质性段落结束（空行或换行后的中文/英文开头）
    m = re.match(
        r'(?:用户(?:询问|问的是|要求|希望|提到|提供|现在的问题是|的问题是)[^。\n]*[。\n]\s*)+'
        r'(?:根据[^。\n]*[。\n]\s*)*'
        r'(?:我(?:需要|应该|可以|必须)[^。\n]*[。\n]\s*)*'
        r'(?:这[^。\n]*[。\n]\s*)*'
        r'(?:回答[^：:\n]*[：:]\s*)?'
        r'(?:[\d]+\.\s*[^\n]+\n)*',  # 编号列表项
        text
    )
    if m and m.end() > 20 and m.end() < len(text) - 10:
        stripped = text[m.end():].strip()
        if len(stripped) > 10:
            return stripped

    # 模式2: 以"嗯"/"好的"/"让我"等开头的推理
    m2 = re.match(
        r'(?:嗯[，,]\s*)?(?:好的[，,]\s*)?(?:让我[^。\n]*[。\n]\s*)+',
        text
    )
    if m2 and m2.end() > 10 and m2.end() < len(text) - 10:
        stripped = text[m2.end():].strip()
        if len(stripped) > 10:
            return stripped

    return text


def detect_instruction_echo(text: str) -> bool:
    """检测文本是否是指令背诵而非正常回答。

    小模型常见退化模式：输出 prompt 指令文本而非实际回答。
    在流式前 200 字符内检测，命中即中止+重试。
    """
    if not text or len(text) < 10:
        return False
    # 在更小的窗口内检查避免误判
    window = text[:200]
    return bool(_get_echo_re().search(window))
