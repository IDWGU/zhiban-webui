"""知伴工作流引擎 — Agent 自主决策 + KV cache 复用

组件: Conversation → AgentLoop(自主工具调用) → RAG(按需多次) → Answer(流式)

核心改进:
  - Agent 自主决策: AI 自主决定搜索关键词和搜索次数（替代固定 Call1/Call2）
  - KV Cache 复用: 分层提示词对齐，跨调用共享前缀缓存
  - message_store 双视图消息管理（完整视图 + 活跃切片）
  - session_state 状态机 + pending queue
  - 回复完成后触发压缩检查
  - 兼容旧版 V11 有限决断循环（USE_AGENT_LOOP=False 时回退）
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
import uuid
from collections import defaultdict
from typing import Callable

logger = logging.getLogger("zhiban.engine")

from .. import config
from ..llm.deepseek_proxy import SYSTEM_PROMPT
from ..llm.message_store import MessageStore
from ..llm.session_state import SessionState, ChatState
from ..llm.kv_cache_config import COMPRESS_BUFFER, COMPRESS_TRIGGER_RATIO

from .classifier import (
    build_classify_prompt_l3,
    build_answer_prompt_l3,
    build_decision_prompt_l3,
    build_eval_answer_prompt_l3,
    build_tiny_decision_prompt,
    build_tiny_answer_prompt,
    detect_instruction_echo,
    strip_reasoning_preamble,
    _today,
)
from .conversation import Conversation
from .llm_utils import (
    sync_call_llm,
    stream_call_llm,
    clear_provider_cache,
    is_local_mode,
    get_local_engine,
    _parse_stop_tokens,
    strip_thinking_tags,
)
from .search_utils import (
    vector_search as _vector_search,
    mmr_rerank as _mmr_rerank,
    extract_json as _extract_json,
    build_history_from_messages as _build_history_from_messages,
    select_with_section_diversity as _select_with_section_diversity,
)

# ── Agent 模块 ──
from ..agent import (
    AgentLoop,
    AgentConfig,
    AgentResult,
    AgentTool,
    create_search_tool,
    create_reading_context_tool,
    create_paper_section_tool,
    ToolResult,
    build_agent_system_prompt,
)
from ..agent.prompts import parse_tool_call as _agent_parse_tool_call
from ..agent.compression import estimate_tokens as _estimate_tokens

SEARCH_DIST_THRESHOLD = 0.50

# ── 论文结构概要 ──

def _build_paper_outline(full_text: str) -> str:
    """从全文提取论文结构概要（零 LLM 调用，纯字符串匹配）。

    扫描章节标题（Introduction、Results 等），每章取前 150 字作为预览，
    拼接成一条 < 2000 字的"论文地图"。

    返回空字符串表示未检测到结构。
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
        return ""  # 没检测到多章节，不生成概要

    parts = ["【论文结构概要】"]
    for sec_name, sec_text in sections:
        preview = sec_text[:150].replace('\n', ' ').strip()
        parts.append(f"- {sec_name}: {preview}...")

    return '\n'.join(parts)


class WorkflowError(Exception):
    """带环节标签的异常，方便前端展示具体失败步骤。"""
    def __init__(self, step: str, detail: str = ""):
        self.step = step
        self.detail = str(detail)
        super().__init__(f"[{step}] {detail}" if detail else step)


# ── 启发式分类 ──

def _detect_question_type(q: str) -> str:
    """检测问题的类型，决定回答风格和 RAG 策略。

    concise:    短定义/事实性问题，一句话答完，不跑 RAG
    pedagogical: 教学引导类，通俗解释，不跑 RAG
    conversational: 意见/建议类，自然对话，不跑 RAG
    analytical: 深入分析类，跑完整 RAG + 结构化回答（默认）
    translation: 翻译请求，路由到翻译模块
    """
    q_len = len(q.strip())

    if any(kw in q for kw in ["翻译", "译成", "translate"]):
        return "translation"

    # 全文总结 → summary（需要完整L2上下文，不走RAG搜索）
    _summary_kw = ["总结全文", "全文总结", "全文核心内容", "概括全文",
                   "总结这篇文章", "全文概括", "总结本论文", "总结一下全文",
                   "全文内容总结", "概括这篇文章", "总结一下这篇文章", "总结这篇论文"]
    if any(kw in q for kw in _summary_kw):
        return "summary"

    # yes/no 确认类 → 简答
    if any(kw in q for kw in ["是吗", "对吗", "是不是", "能不能", "是否", "有无"]) or \
       re.search(r'(?:讲|说|指|谈|称|叫)的?(?:是|不是).*[吗呢吧]', q) or \
       (q.endswith("吗") and any(kw in q for kw in [
           "是", "有", "可以", "属于", "能", "需要", "必须", "适用", "存在", "会",
       ])):
        return "concise"

    # 短问题 + 定义/术语解释类关键词 → 简答
    # 注意：不包含 "怎么产生的/怎么形成的" — 那是机理解释，需要 RAG
    if q_len < 30 and any(kw in q for kw in [
        "是什么", "什么意思", "什么是", "指的是", "定义", "简称", "缩写",
        "指什么", "全称", "是谁", "啥是", "啥意思",
        "是怎么", "是怎样", "如何定义",
    ]):
        return "concise"

    # 长问题但核心是定义 → 仍然是简答
    if any(kw in q for kw in ["是什么意思", "指的是什么", "的定义是什么"]):
        return "concise"

    # 教学/入门类
    if any(kw in q for kw in [
        "初学者", "入门", "怎么学", "从哪里入手", "如何学习",
        "先理解什么", "先学什么", "基础概念", "新手",
    ]):
        return "pedagogical"

    # 意见/建议类 — 用户明确寻求主观意见时，即使含学术词也是对话
    _opinion_kw = ["你觉得", "你认为", "我应该", "我该", "推荐", "建议", "怎么看"]
    if any(kw in q for kw in _opinion_kw):
        # 如果用户同时在问具体学术问题（如"为什么"+"催化剂"），保留 analytical
        _is_specific_academic = any(kw in q for kw in [
            "为什么", "怎么产生", "机理", "机制详解", "电子结构",
            "反应路径", "DFT", "计算",
        ])
        if not _is_specific_academic:
            return "conversational"

    # 纯学术问题兜底（"有什么优缺点" → 不是意见，是事实查询）
    _has_academic = any(kw in q for kw in [
        "论文", "实验", "性能", "机理", "机制", "反应", "材料",
        "催化剂", "电极", "电池", "合成", "表征", "XRD", "SEM",
    ])

    # 默认：深度分析
    return "analytical"


def _classify_heuristic(question: str, current_topic: str = "", screen_ctx: str = "") -> dict | None:
    """纯规则意图分类，不依赖 LLM。

    按优先级从高到低匹配。命中返回完整分类 dict，未命中返回 None 让 LLM 兜底。
    专为本地小模型设计：规则覆盖 ~80% 场景，确定性强，永不出错。
    """
    q = question.strip()
    q_lower = q.lower()
    q_len = len(q)

    # 辅助：提取关键词
    terms = _simple_extract_keywords(q, current_topic=current_topic)

    # ── P1: 术语查询 ("xxx是什么") ──
    if re.match(r'^.{1,25}(是什么|是什么意思|的定义|指什么|全称|简称|缩写)', q) or \
       re.match(r'^(什么是|谁是|啥是)', q):
        return {
            "mode": "deep", "is_meta": False,
            "is_term_lookup": True, "is_recommendation": False,
            "topic": current_topic or (terms[0] if terms else q[:15]),
            "search_terms": terms or [q[:20]],
            "corrected_question": q,
            "topic_switch": False, "ref_number": None, "output_style": "conversation",
        }

    # ── P2: 论文推荐 ──
    rec_keywords = ["推荐", "有什么论文", "找一篇", "相关文献", "给我一些论文", "介绍几篇",
                    "推荐几篇", "还有哪些论文", "类似的研究", "有没有类似",
                    "有没有人做过", "类似的工作", "类似的文章", "相关的研究"]
    if any(kw in q for kw in rec_keywords):
        return {
            "mode": "deep", "is_meta": False,
            "is_term_lookup": False, "is_recommendation": True,
            "topic": terms[0] if terms else q[:15],
            "search_terms": terms or [q[:20]],
            "corrected_question": q,
            "topic_switch": False, "ref_number": None, "output_style": "conversation",
        }

    # ── P3: 纯寒暄 (短文本 + 无实义内容) ──
    # 短词用 \b 避免子串误匹配（如 "hi" 匹配 "highlight"）
    _short_greetings_re = re.compile(
        r'\b(hi|hey|oh|bye)\b|'
        r'(?<![a-zA-Z])(嗯|早|哦)(?![a-zA-Z])'
    )
    _long_greetings = ["你好", "hello", "嗨", "哈喽", "halo", "在吗", "谢谢", "你是谁",
                       "晚安", "再见", "早上好", "晚上好", "下午好", "hello!", "hi!"]
    greeting_hit = (
        bool(_short_greetings_re.search(q_lower)) or
        any(g in q_lower for g in _long_greetings)
    )
    if greeting_hit and q_len < 20:
        # 排除混有实质问题的 ("Hi，请帮我总结论文")
        substantive = any(
            kw in q for kw in [
                "论文", "paper", "文献", "总结", "概括", "解释", "分析",
                "怎么做", "怎么办", "为什么", "如何", "推荐",
                "翻译", "实验", "数据", "搜索", "查找",
            ]
        )
        if not substantive:
            return {
                "mode": "chat", "is_meta": True,
                "is_term_lookup": False, "is_recommendation": False,
                "topic": "", "search_terms": [],
                "corrected_question": q,
                "topic_switch": False, "ref_number": None, "output_style": "conversation",
            }

    # ── P4: 修正 ──
    correction_patterns = [
        "不对", "你错了", "搞错了", "错了", "不是这样", "不是这个", "不是这篇",
        "你说错了", "不准确", "有误", "错误", "不正确",
        "重新查", "再查", "重新搜", "你再看看", "确认一下",
        "不对吧", "你确定", "更正", "纠正", "再搜", "重新生成",
        "你再想想", "你确定吗", "真的吗", "有出处吗", "有依据吗",
    ]
    if any(p in q for p in correction_patterns):
        return {
            "mode": "deep", "is_meta": False,
            "is_term_lookup": False, "is_recommendation": False,
            "topic": current_topic or (terms[0] if terms else q[:15]),
            "search_terms": terms or [current_topic] if current_topic else [q[:20]],
            "corrected_question": q,
            "topic_switch": False, "ref_number": None, "output_style": "conversation",
        }

    # ── P5: 章节引用跟进 (第X部分/第X点 + 展开/详细) ──
    # 优先级高于 P6 (通用流控)，因为用户明确指向对话历史中的编号内容
    _section_match = re.search(
        r'第\s*([一二三四五六七八九十\d]+)\s*(?:个|部分|点|块|小块|模块|节|段|项|小点|小节)',
        q
    )
    if _section_match and (_section_match.group() in q[:15] or any(
        kw in q for kw in ["展开", "详细", "具体", "深入", "介绍", "说说", "讲"]
    )):
        _sec_num = _section_match.group(1)
        _sec_label = _section_match.group()
        corrected = (
            f"注意：你上次的回答（已在对话历史中）包含了编号内容，"
            f"其中「{_sec_label}」已被列出。"
            f"请针对这「{_sec_label}」展开详细论述，"
            f"不要重复其他编号的内容，也不要说你没有上下文。"
        )
        return {
            "mode": "chat", "is_meta": False,
            "is_term_lookup": False, "is_recommendation": False,
            "topic": current_topic,
            "search_terms": [],
            "corrected_question": corrected,
            "topic_switch": False, "ref_number": None, "output_style": "conversation",
        }

    # ── P6: 流控/追问 (继续/展开/详细/举例) ──
    flow_patterns = ["继续", "接着说", "然后呢", "还有吗", "接下来", "往下说",
                     "再讲", "继续讲", "接着讲", "讲下去", "然后呢?",
                     "展开说", "展开讲", "展开聊聊", "详细说", "详细讲",
                     "具体点", "说具体", "举个例子", "多举几个", "举例说明",
                     "换个角度", "用更简单的话", "说人话", "通俗一点",
                     "深入一点", "再深入", "深挖一下", "详细一点",
                     "再说说", "再说一下", "再讲讲", "讲讲这个", "说说这个",
                     "针对这个", "就这个", "关于这个", "顺着这个"]
    # 用户用了指代词（这个/那个）+ 有引用内容 → 从引用内容提取搜索词
    _has_pointer = any(p in q for p in ["这个", "那个", "它", "这方", "前面",
                                          "上述", "上边", "刚刚", "刚才", "那段"])
    if any(p in q for p in flow_patterns) or (_has_pointer and screen_ctx):
        if screen_ctx:
            # 用户选中了文字 + 流控词 → 从选中内容提取关键词用于 RAG 检索
            para_keywords = _simple_extract_keywords(screen_ctx)
            search_kw = para_keywords if para_keywords else (
                terms if terms else ([current_topic] if current_topic else [q[:20]])
            )
            corrected = (
                f"用户引用了以下内容（见上方「用户引用」或「用户阅读/引用」部分），"
                f"请针对该内容进行深入展开论述，聚焦于其中的具体论点、实验数据、"
                f"关键概念和论证逻辑。用户的问题是：「{q}」"
            )
            return {
                "mode": "deep", "is_meta": False,
                "is_term_lookup": False, "is_recommendation": False,
                "topic": current_topic,
                "search_terms": search_kw,
                "corrected_question": corrected,
                "topic_switch": False, "ref_number": None, "output_style": "conversation",
            }
        # 无 screen_ctx → 保持原有逻辑
        corrected = f"请继续之前关于{current_topic}的讨论" if current_topic else q
        return {
            "mode": "deep", "is_meta": False,
            "is_term_lookup": False, "is_recommendation": False,
            "topic": current_topic,
            "search_terms": [current_topic] if current_topic else (terms or [q[:20]]),
            "corrected_question": corrected,
            "topic_switch": False, "ref_number": None, "output_style": "conversation",
        }

    # ── P6: 上下文指代 (依赖已有话题) ──
    pronoun_keywords = ["它", "这个", "那个", "上面", "前面", "刚才", "之前"]
    if any(kw in q for kw in pronoun_keywords) and current_topic:
        return {
            "mode": "deep", "is_meta": False,
            "is_term_lookup": False, "is_recommendation": False,
            "topic": current_topic,
            "search_terms": terms or [current_topic],
            "corrected_question": q,
            "topic_switch": False, "ref_number": None, "output_style": "conversation",
        }

    # ── P6.5: 教学/入门引导类 → chat，不跑 RAG ──
    _teaching_patterns = [
        "作为初学者", "作为新手", "怎么入门", "如何入门", "从哪里入手",
        "先理解什么", "先学什么", "基础概念", "入门指南", "学习路线",
        "初学者应该", "新手应该", "刚开始学", "零基础",
    ]
    if any(p in q for p in _teaching_patterns):
        corrected = (
            f"你是一位耐心的导师。用户是电化学初学者，"
            f"请用最简单易懂的语言解释，从最基本的概念讲起。"
            f"每引入一个新术语，先用一句话解释它是什么。"
            f"用户的问题是：「{q}」"
        )
        return {
            "mode": "chat", "is_meta": False,
            "is_term_lookup": False, "is_recommendation": False,
            "topic": current_topic,
            "search_terms": [],
            "corrected_question": corrected,
            "topic_switch": False, "ref_number": None, "output_style": "conversation",
        }

    # ── P7: 论文/学术关键词 → deep ──
    paper_keywords = [
        # 通用学术
        "论文", "paper", "文献", "研究", "学术", "科研", "领域",
        # 论文结构
        "摘要", "引言", "Abstract", "Introduction",
        "相关工作", "方法", "结果与讨论", "Discussion", "结论",
        "补充材料", "SI", "Supporting Information",
        "文章", "段落", "小节", "章节",
        # 实验/方法
        "实验", "合成", "表征", "制备", "测试", "分析",
        "XRD", "TEM", "SEM", "XPS", "BET", "FTIR", "Raman",
        "DFT", "TGA", "DSC", "NMR", "GC", "HPLC", "MS",
        "原位", "in-situ", "operando", "ex-situ",
        "对照", "对照组", "空白", "参比",
        # 性能/机理
        "性能", "机理", "机制", "反应", "催化", "活性",
        "选择性", "转化率", "产率", "过电位", "电流密度",
        "法拉第效率", "TOF", "TON",
        # 材料
        "纳米", "材料", "催化剂", "结构", "形貌", "晶面", "晶相",
        # 写作/发表
        "审稿", "投稿", "回复意见", "cover letter", "highlight",
        "期刊", "IF", "影响因子", "DOI", "引用", "发表",
        # 评价/分析
        "数据", "作者", "通讯作者", "第一作者",
        "创新点", "贡献", "不足", "局限", "展望",
        "重点", "对比", "区别", "优缺点",
        "总结", "概括", "归纳", "梳理", "综述",
        # 图表
        "图", "表格", "示意图", "图表", "公式",
        # 计算/数值
        "计算", "怎么算", "如何计算", "怎么测",
        "电化学", "光催化", "光电", "热催化",
        # 模糊阅读指令
        "观点", "核心观点", "发现", "关键", "主要内容",
        "本文", "在读", "讨论", "解决", "说明", "提出",
        "亮点", "讲一下", "概述", "简述", "简介",
        "概览", "梳理一下", "讲一讲", "说说", "说了什么",
    ]
    if any(kw in q for kw in paper_keywords):
        return {
            "mode": "deep", "is_meta": False,
            "is_term_lookup": False, "is_recommendation": False,
            "topic": current_topic or (terms[0] if terms else q[:15]),
            "search_terms": terms or [q[:20]],
            "corrected_question": q,
            "topic_switch": False, "ref_number": None, "output_style": "conversation",
        }

    # ── P8: 疑问/指令句式 → deep ──
    if re.search(
        r'(你觉得|你认为|你怎么看|你的看法|怎么看|从哪|从何|你能|你能不能|'
        r'可不可以|能不能|能否帮我|可以帮我|怎么样|合适吗|可靠吗|普适吗|'
        r'一致吗|有道理吗|合理吗|正确吗|可行吗|有没有可能|还有其他|'
        r'有出处吗|有依据吗|有数据支持吗|你的回答|'
        r'什么意思|是什么东西)',
        q,
    ) or re.match(
        r'^(什么|为什么|怎么|如何|怎样|哪里|哪个|是否|可以|能否|能|'
        r'请|帮我|帮我介绍一下|简单介绍|介绍|说说|说明|阐述|讲述|'
        r'描述|列举|比较|区分|定义|解释|翻译|润色|改写|概述|简述)',
        q,
    ):
        return {
            "mode": "deep", "is_meta": False,
            "is_term_lookup": False, "is_recommendation": False,
            "topic": current_topic or (terms[0] if terms else q[:15]),
            "search_terms": terms or [q[:20]],
            "corrected_question": q,
            "topic_switch": False, "ref_number": None, "output_style": "conversation",
        }

    # ── 未命中 → 需要 LLM 分类 ──
    return None


# ── 循环检测 ──

def _detect_loop(text: str, tail_chars: int = 500) -> bool:
    """检测文本末尾是否出现连续重复（LLM 循环的典型特征）。

    取末尾 500 字符，检查是否存在 >=20 字符的后缀与其紧邻前缀一致。
    例: \"...ABCABC\" → \"ABC\" 连续出现 → 循环。
    最小重复长度 20（而非 12）以减少科学术语的误判。
    """
    if not text or len(text) < 100:
        return False
    tail = text[-tail_chars:] if len(text) > tail_chars else text
    # 从长到短检查，优先匹配更明显的重复模式
    for n in range(min(200, len(tail) // 2), 19, -1):
        if tail[-n:] == tail[-n*2:-n]:
            return True
    return False


def _is_tiny_model(model_path: str) -> bool:
    """已废弃 — 所有模型统一使用长版提示词。保留函数避免旧代码报错。"""
    return False


# ── V11: 决策循环辅助函数 ──

def _parse_decision_keywords(text: str) -> list[str]:
    """从 LLM 决策输出中解析搜索关键词。

    支持格式:
      "搜索关键词: k1, k2, k3"
      "不需要搜索" → 返回空列表
    """
    if not text:
        return []
    text = text.strip()
    if "不需要搜索" in text or "无需搜索" in text or "不用搜索" in text:
        return []
    m = re.search(r"搜索关键词[:：]\s*(.+)", text)
    if m:
        raw = m.group(1).strip()
        terms = [t.strip() for t in re.split(r"[,，、;；]+", raw) if t.strip()]
        return terms[:5]


# ── Agent 辅助函数 ──

def _has_tools_in_response(text: str) -> bool:
    """快速检测文本是否含工具调用标记（用于决策是否需要后续 LLM 调用）"""
    if not text:
        return False
    import re
    return bool(re.search(r'<tool_call>|\[工具调用:', text, re.IGNORECASE))


def _extract_after_marker(text: str, marker: str) -> str:
    """提取 marker 之后的内容。"""
    idx = text.rfind(marker)
    if idx >= 0:
        return text[idx + len(marker):]
    return ''

def _clean_answer_text(text: str) -> str:
    """移除 XML 标签和重复段落。"""
    import re
    if not text:
        return text

    # Step 1: 移除 XML 标签（<think>, <tool_call> 等）
    cleaned = strip_thinking_tags(text)
    # 也移除 <tool_call> 块（模型在答案步可能重复输出工具调用）
    cleaned = re.sub(r'<tool_call>.*?</tool_call>', '', cleaned, flags=re.DOTALL)
    # 移除孤立的 JSON 块（工具调用的残留）
    cleaned = re.sub(r'\{\s*"name"\s*:\s*"工具名".*?\}', '', cleaned, flags=re.DOTALL)

    # Step 2: 移除连续重复的段落（循环检测的兜底）
    paragraphs = [p.strip() for p in cleaned.split('\n\n') if p.strip()]
    if len(paragraphs) >= 2:
        seen = set()
        unique = []
        for p in paragraphs:
            fp = p[:60]
            if fp not in seen:
                seen.add(fp)
                unique.append(p)
        cleaned = '\n\n'.join(unique)

    return cleaned.strip()


def _extract_reasoning_text(text: str) -> str:
    """从模型输出中提取推理文本（剥离工具调用块），用于 thinking 展示。"""
    if not text:
        return ""
    import re
    # 移除 <tool_call>...</tool_call> 块
    cleaned = re.sub(
        r'<tool_call>.*?</tool_call>',
        '', text, flags=re.DOTALL | re.IGNORECASE,
    )
    # 移除未闭合的 <tool_call> 块
    cleaned = re.sub(
        r'<tool_call>.*$', '', cleaned, flags=re.DOTALL | re.IGNORECASE,
    )
    # 移除 [工具调用: ...] 块
    cleaned = re.sub(
        r'\[工具调用:.*?(?=\[工具调用:|$)', '', cleaned, flags=re.DOTALL,
    )
    return cleaned.strip()


def _parse_research_signal(text: str) -> str:
    """检查文本开头是否包含重搜信号。

    格式: [需要补充检索: 关键词1, 关键词2]
    返回关键词字符串，无信号时返回空字符串。
    """
    m = re.search(r"\[需要补充检索[:：]\s*([^\]]+)\]", text)
    if m:
        return m.group(1).strip()
    return ""


def _get_last_assistant_content(active_slice: list[dict]) -> str:
    """从活跃切片中提取上一次 assistant 回复的内容。"""
    for m in reversed(active_slice):
        if m.get("role") == "assistant" and m.get("content"):
            return m["content"]
    return ""


def _build_recent_context(active_slice: list[dict], max_exchanges: int = 2) -> str:
    """从活跃切片中提取最近 N 轮对话（用户+助手），不再截断单条长度。"""
    exchanges: list[str] = []
    assistant_count = 0
    for m in reversed(active_slice):
        role = m.get("role", "")
        content = m.get("content", "")
        if role in ("user", "assistant") and content.strip():
            label = "用户" if role == "user" else "助手"
            exchanges.append(f"[{label}]: {content}")
            if role == "assistant":
                assistant_count += 1
                if assistant_count >= max_exchanges:
                    break
    exchanges.reverse()
    return "\n\n".join(exchanges)


def _build_llm_params(base_url: str) -> dict:
    """从 config 构建传递给 LLM 调用的反重复参数字典。"""
    stop = _parse_stop_tokens(getattr(config, "LLM_STOP_TOKENS", ""))
    return {
        "temperature": config.LLM_TEMPERATURE,
        "repeat_penalty": config.LLM_REPEAT_PENALTY,
        "top_k": config.LLM_TOP_K,
        "top_p": config.LLM_TOP_P,
        "stop": stop,
        "frequency_penalty": config.LLM_FREQUENCY_PENALTY,
        "presence_penalty": config.LLM_PRESENCE_PENALTY,
    }


def _build_no_kb_notice() -> str:
    """动态构建知识库空结果提示词，从 graph_store 实时获取论文数。"""
    try:
        from ..rag.graph_store import graph_store
        count = graph_store.paper_count
        kb_desc = f"（知识库共 {count} 篇论文）" if count else ""
    except Exception as e:
        logger.warning("Failed to get paper count for KB notice: %s", e)
        kb_desc = ""
    return f"""【系统提示 — 知识库检索状态】
⚠️ 本次在论文知识库{kb_desc}中未检索到与用户问题直接匹配的内容。

请按以下要求回答：
1. 在回答开头明确告知用户：「⚠️ 知识库中未检索到相关内容，以下回答基于我的通用知识，可能不完全适用于本知识库的语境。」
2. 基于你自己的通用知识尽量回答用户问题。
3. 回答中不要编造论文编号或具体实验数据（因为你没有检索到相关内容）。
4. 如果问题涉及具体论文的实验细节而你无法确定，请诚实说明。"""


def _simple_topic(text: str) -> str:
    """分类失败时的简单话题提取：取前 10 个中文字符作为话题。"""
    import re
    chinese = re.findall(r'[一-鿿]+', text)
    joined = ''.join(chinese)
    return joined[:15] if joined else text[:20]


def _simple_extract_keywords(text: str, current_topic: str = "") -> list[str]:
    """分类失败时的简单关键词提取：中文字符 + 话题词。"""
    import re
    # 提取中文词组（2字以上）
    chinese_words = re.findall(r'[一-鿿]{2,}', text)
    # 去重 + 限制数量
    seen = set()
    terms = []
    for w in chinese_words:
        if w not in seen and len(w) >= 2:
            seen.add(w)
            terms.append(w)
            if len(terms) >= 5:
                break
    if current_topic and current_topic.strip() and current_topic.strip() not in terms:
        terms.insert(0, current_topic.strip())
    return terms[:3]


class ZhibanV10:
    """知伴工作流引擎 — Agent 自主决策 + KV cache 复用"""

    # Agent 模块开关：设为 False 回退到旧版 V11 有限决断循环
    USE_AGENT_LOOP: bool = True

    def __init__(self):
        self.conversations: dict[str, Conversation] = {}
        self.active_conv_id: str | None = None
        self._persistence = None
        self._api_key = ""
        self._model = ""
        self._base_url = ""
        self._thinking = True
        self._cancel_events: dict[str, asyncio.Event] = {}  # conv_id → Event，隔离并发查询

        # Phase 2: 消息管理 + 状态机
        self._message_stores: dict[str, MessageStore] = {}
        self._session_state = SessionState()
        self._l2_cache: dict[str, str] = {}  # conv_id → 论文全文
        self._l2_outline_cache: dict[str, str] = {}  # conv_id → 论文结构概要
        self._l2_outline_used: dict[str, bool] = {}  # conv_id → 是否已用过全文

        # 压缩引擎（延迟初始化）
        self._compress_engine = None
        self._archived_index = None

        # Agent 工具（延迟初始化）
        self._agent_tools: list[AgentTool] | None = None
        self._agent_config: AgentConfig | None = None

        # 阅读区惰性推送：记录每轮 screen_ctx 用于变化检测
        self._last_screen_ctx: dict[str, str] = {}  # conv_id → 上一轮的 screen_ctx
        self._current_screen_ctx: str = ""          # 当前轮次的 screen_ctx（供工具读取）

    @property
    def _cancel_event(self) -> asyncio.Event | None:
        """当前活跃会话的取消事件（兼容旧代码）。"""
        return self._cancel_events.get(self.active_conv_id) if self.active_conv_id else None

    @_cancel_event.setter
    def _cancel_event(self, event: asyncio.Event | None):
        if event is None:
            self._cancel_events.pop(self.active_conv_id, None)
        elif self.active_conv_id:
            self._cancel_events[self.active_conv_id] = event

    # ── 初始化 ──

    def set_persistence(self, store):
        self._persistence = store

    def _get_message_store(self, conv_id: str) -> MessageStore:
        if conv_id not in self._message_stores:
            self._message_stores[conv_id] = MessageStore(SYSTEM_PROMPT)
        return self._message_stores[conv_id]

    def _get_embed_fn(self):
        """获取 embedding 函数（延迟导入避免循环依赖）"""
        try:
            from ..rag.embeddings import embedding_engine as ee
            if ee.is_available:
                return ee.embed_query
        except Exception:
            pass
        return None

    def _get_compress_engine(self):
        if self._compress_engine is None:
            from ..llm.compress_engine import CompressEngine
            embed_fn = self._get_embed_fn()
            n_ctx = self._resolve_n_ctx()
            self._compress_engine = CompressEngine(
                sync_llm=sync_call_llm,
                embed_fn=embed_fn,
                n_ctx=n_ctx,
            )
        return self._compress_engine

    def _resolve_n_ctx(self) -> int:
        """解析实际可用的上下文窗口大小，优先从加载的本地引擎获取。"""
        if is_local_mode(self._base_url):
            eng = get_local_engine()
            if eng is not None and eng.is_loaded:
                return eng.n_ctx
        from ..llm.kv_cache_config import resolve_n_ctx, DEFAULT_N_CTX
        env_val = getattr(config, "LLM_MAX_CONTEXT", "0")
        if env_val and str(env_val).strip() and int(str(env_val)) > 0:
            return resolve_n_ctx()
        return DEFAULT_N_CTX

    def _get_archived_index(self):
        if self._archived_index is None:
            from ..llm.archived_index import ArchivedIndex
            embed_dim = None
            try:
                from ..rag.embeddings import embedding_engine as ee
                if ee.model is not None and ee.is_available:
                    try:
                        embed_dim = ee.dim
                    except Exception:
                        pass
            except ImportError:
                pass
            self._archived_index = ArchivedIndex(embedding_dim=embed_dim)
        return self._archived_index

    @property
    def session_state(self) -> SessionState:
        return self._session_state

    # ── Agent 工具 ──

    def _get_agent_tools(self) -> list[AgentTool]:
        """延迟初始化 Agent 工具列表"""
        if self._agent_tools is None:
            async def _search_fn(query: str, top_k: int = 5):
                return await asyncio.to_thread(_vector_search, query, top_k)

            async def _paper_overview_fn(**kwargs):
                conv_id = getattr(self.conv, 'id', '') if self.conv else ''
                full_text = self.get_l2_context(conv_id) if conv_id else ''
                if not full_text:
                    return ToolResult(
                        tool_name="get_paper_overview",
                        success=True,
                        content="（当前无已打开的论文）",
                    )
                return ToolResult(
                    tool_name="get_paper_overview",
                    success=True,
                    content=full_text[:4000],
                    raw_data=full_text,
                )

            self._agent_tools = [
                create_search_tool(
                    vector_search_fn=_search_fn,
                    mmr_rerank_fn=_mmr_rerank,
                ),
                create_reading_context_tool(
                    get_context_fn=lambda: self._current_screen_ctx,
                ),
                create_paper_section_tool(
                    get_section_fn=None,  # TODO: 接入实际章节获取
                ),
            ]

            # 添加论文概要工具
            from ..agent.tools import AgentTool
            self._agent_tools.append(AgentTool(
                name="get_paper_overview",
                description="获取当前已打开论文的全文内容。当需要总结论文、了解全文结构、或回答涉及论文整体内容的问题时使用。",
                parameters={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
                handler=_paper_overview_fn,
                is_read_only=True,
                max_result_chars=4000,
            ))
        return self._agent_tools

    def _get_agent_config(self) -> AgentConfig:
        """延迟初始化 Agent 配置"""
        if self._agent_config is None:
            n_ctx = self._resolve_n_ctx()
            self._agent_config = AgentConfig(
                max_steps=5,
                max_search_rounds=3,
                thinking_budget=1024,
                answer_max_tokens=0,  # 0 = 不限制，llama.cpp 用满剩余上下文
                verbose=config.DEBUG,
                kv_cache_reuse=True,
            )
        return self._agent_config

    # ── Agent LLM 调用桥接 ──

    async def _agent_llm_call(
        self,
        messages: list[dict],
        max_tokens: int = 0,
        **llm_params,
    ) -> tuple[str, dict]:
        """Agent → 现有 LLM 基础设施的桥接函数。

        messages[0] 为 system prompt（如需）。
        返回 (text, usage_dict)。
        """
        # 提取 system prompt (如果有)
        system_prompt = ""
        user_msgs = list(messages)
        if messages and messages[0].get("role") == "system":
            system_prompt = messages[0]["content"]
            user_msgs = messages[1:]

        if not system_prompt:
            system_prompt = SYSTEM_PROMPT

        # 调用现有同步 LLM 接口
        try:
            text, usage = await asyncio.to_thread(
                sync_call_llm,
                system_prompt,
                user_msgs,
                max_tokens=max_tokens,
                api_key=self._api_key,
                model=self._model,
                base_url=self._base_url,
                thinking=False,  # Agent 决策时不开 thinking 以加速
                cancel_event=self._cancel_event,
                reuse_provider=True,  # KV cache 复用
                **llm_params,
            )
            text = text or ""
            loop_detected = False
            from ..agent.prompts import detect_chinese_loop
            if len(text) > 300 and detect_chinese_loop(text):
                logger.warning("Agent sync loop detected")
                loop_detected = True
                # 不截断：截断可能产生句末残缺，交给调用方决定是否重试
            return text, usage or {}, loop_detected
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("Agent LLM call failed: %s", e)
            return f"（调用出错: {e}）", {"error": str(e)}

    async def _agent_answer_llm_call(
        self,
        messages: list[dict],
        max_tokens: int = 0,
        **llm_params,
    ) -> tuple[str, dict]:
        """Agent LLM 流式调用。

        先 buffer 全部内容，流结束后根据是否有 tool_call 决定推思考区还是正文区。
        避免决策步的直接回答被错误推入思考区导致用户看不到。
        """
        system_prompt = ""
        user_msgs = list(messages)
        if messages and messages[0].get("role") == "system":
            system_prompt = messages[0]["content"]
            user_msgs = messages[1:]

        if not system_prompt:
            system_prompt = SYSTEM_PROMPT

        # 如果未设置 stream handler，回退到同步调用
        if not hasattr(self, '_on_token') or not self._on_token:
            text, usage, loop_detected = await self._agent_llm_call(messages, max_tokens, **llm_params)
            return text, usage, [], [], loop_detected

        # is_thinking 标志由上游 provider（local_chat_engine / OpenAI API）提供，
        # 引擎层直接信任不再做二次 <think> 标签解析，避免重复工作和窗口截断风险。
        # is_thinking=True  → 前端思考面板
        # is_thinking=False → 前端正文区
        content_parts: list[str] = []  # 所有 token，供解析和结果拼接
        usage_info: dict = {}
        content_token_count = [0]

        def _on_stream_token(token: str, is_thinking: bool = False):
            content_parts.append(token)
            content_token_count[0] += 1
            self._agent_body_streamed = True
            self._on_token(token, is_thinking)

            # 诊断：首尾 40 token 的 is_thinking 分布
            if content_token_count[0] <= 3 or content_token_count[0] % 200 == 0:
                logger.debug(
                    "stream_token #%d is_thinking=%s token=%.60s",
                    content_token_count[0], is_thinking, token,
                )

            # 环路检测
            if content_token_count[0] >= 40 and content_token_count[0] % 40 == 0:
                window = ''.join(content_parts[-200:])
                from ..agent.prompts import detect_stream_loop, detect_chinese_loop
                if detect_stream_loop(window) or detect_chinese_loop(window):
                    logger.warning("Stream loop suspected at token %d "
                                   "(NOT aborting, post-loop detection will handle)",
                                   content_token_count[0])

        try:
            result = await stream_call_llm(
                system_prompt,
                user_msgs,
                max_tokens=max_tokens,
                api_key=self._api_key,
                model=self._model,
                base_url=self._base_url,
                thinking=self._thinking,
                on_token=_on_stream_token,
                reuse_provider=True,
                **llm_params,
            )
            # 流结束：如果在 </think> 之前就结束了，剩余 token 已通过 _on_stream_token 发送为 thinking

            resp = result.get("response", "")
            usage_info = result.get("usage", {})

            if not content_parts and resp:
                content_parts = [resp]

            raw_text = "".join(content_parts)
            full_text = _clean_answer_text(raw_text)
            # 流式环路检测已改为非阻断（只记录日志），loop_detected 始终为 False。
            # 真正的循环由 agent_loop.py 的 post-loop detect_loop 处理（跨迭代对比）。
            loop_detected = False

            # 诊断日志：流式输出完成
            _has_think = '</think>' in raw_text
            logger.info(
                "Agent stream done: tokens=%d raw_chars=%d clean_chars=%d "
                "has_think=%s finish_reason=%s",
                content_token_count[0], len(raw_text), len(full_text),
                _has_think, result.get("finish_reason", "N/A"),
            )

            return full_text, usage_info, [], content_parts, loop_detected
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("Agent streaming LLM call failed: %s", e)
            resp_text = "（回答生成失败，请重试）"
            if self._on_token:
                self._on_token(resp_text, False)
            return resp_text, {"error": str(e)}

    # ── Agent 运行 ──

    async def _run_agent(
        self,
        question: str,
        screen_ctx: str,
        conv,
        msg_store: MessageStore,
        l2_text: str,
        history_str: str,
        wake_records_text: str,
        max_tokens: int,
        llm_params: dict,
        on_status: Callable | None = None,
        on_token: Callable | None = None,
        on_health: Callable | None = None,
        on_agent_step: Callable | None = None,
    ) -> dict:
        """使用 Agent 循环处理用户问题。

        替代旧版 _llm_decision_loop，AI 自主决定搜索策略。
        """
        tools = self._get_agent_tools()
        agent_config = self._get_agent_config()
        agent_config.answer_max_tokens = max_tokens

        # 保存 token 回调引用供 _agent_answer_llm_call 使用
        self._on_token = on_token

        self._agent_body_streamed = False  # 正文是否已实际推送过

        # 阅读区惰性推送：仅在内容变化时自动注入，否则由 LLM 通过工具获取
        conv_id = conv.id
        self._current_screen_ctx = screen_ctx or ""
        prev = self._last_screen_ctx.get(conv_id, "")
        screen_changed = bool(screen_ctx) and screen_ctx != prev
        if screen_changed:
            self._last_screen_ctx[conv_id] = screen_ctx

        loop = AgentLoop(
            config=agent_config,
            tools=tools,
            on_status=on_status,
            on_token=on_token,
            on_health=on_health,
        )

        active_slice = msg_store.get_active_slice()

        # 知识库概况
        knowledge_brief = ""
        if conv.open_papers:
            papers_str = conv.paper_titles_str()
            if papers_str:
                knowledge_brief = f"已打开论文: {papers_str}"

        # Agent LLM 调用：流式 + 循环检测 + 升温重试
        _final_answer_streamed = [False]
        _loop_retry_count: dict[int, int] = {}

        ANTI_LOOP_PROMPT = (
            "你刚才陷入了反复检查和重复输出的循环。"
            "现在请停止分析过程，忽略之前的推理，"
            "直接给出简洁的最终回答。"
            "只输出结论本身，不要加任何推理、解释或自我检查。"
        )

        async def _agent_llm_fn(messages, max_tokens, **params):
            step_idx = len(loop._steps)
            last_msg = messages[-1].get("content", "") if messages else ""
            has_tool_results = "【工具执行结果】" in last_msg
            is_maxed_out = step_idx + 1 >= agent_config.max_steps

            stream_max_tokens = max_tokens  # 不限制，让模型自然停止

            text, usage, think_tokens, content_tokens, loop_detected = \
                await self._agent_answer_llm_call(
                    messages, stream_max_tokens, **params)

            # 循环检测 → 升温重试（温度递增引入随机性打破循环）
            retries = _loop_retry_count.get(step_idx, 0)
            if loop_detected and retries < 3:
                _loop_retry_count[step_idx] = retries + 1
                orig_temp = float(params.get("temperature", 0.0))
                # 每次重试温度 +0.3，上限 1.2
                new_temp = min(orig_temp + 0.3 * (retries + 1), 1.2)
                logger.info("Loop at step %d, retry %d/3 temp=%.1f→%.1f",
                            step_idx, retries + 1, orig_temp, new_temp)
                if on_status:
                    on_status("thinking", f"检测到循环，正在重新生成... ({retries + 1}/3)")
                retry_msgs = list(messages) + [
                    {"role": "user", "content": ANTI_LOOP_PROMPT}
                ]
                retry_params = {**params, "temperature": new_temp}
                retry_text, retry_usage, _ = await self._agent_llm_call(
                    retry_msgs, max_tokens, **retry_params
                )
                if retry_text.strip():
                    text = retry_text
                    usage = retry_usage
                    if on_token:
                        on_token(retry_text, False)
                    logger.info("Retry %d OK temp=%.1f %d chars",
                                retries + 1, new_temp, len(retry_text))

            if not _has_tools_in_response(text) and not _final_answer_streamed[0]:
                _final_answer_streamed[0] = True
            return text, usage

        try:
            result: AgentResult = await loop.run(
                question=question,
                screen_ctx=screen_ctx,
                screen_changed=screen_changed,
                l2_text=l2_text,  # 总结场景传递全文，其余为空
                current_topic=conv.current_topic,
                history_str=history_str,
                wake_records_text=wake_records_text,
                question_type=_detect_question_type(question),
                active_slice=active_slice,
                knowledge_brief=knowledge_brief,
                llm_call_fn=_agent_llm_fn,
                llm_params=llm_params,
                cancel_event=self._cancel_event,
                on_agent_step=on_agent_step,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("Agent loop failed: %s", e)
            raise WorkflowError("agent_loop", str(e))

        # 清理 agent 抑制标志，恢复正文推送
        self._on_token = None

        # 发送 health 数据
        if on_health:
            for step in result.steps:
                usage = step.llm_usage
                health = usage.get("health", {}) if isinstance(usage, dict) else {}
                on_health({
                    "call": f"agent_step_{step.step_index}",
                    "timing": {"elapsed_ms": step.duration_ms},
                    "tokens": {
                        "input": usage.get("input", 0) if isinstance(usage, dict) else 0,
                        "output": usage.get("output", 0) if isinstance(usage, dict) else 0,
                    },
                    "debug_text": f"tools={len(step.tool_calls)} "
                                  f"results={len(step.tool_results)}",
                })

        return self._result(
            response=result.final_answer,
            expanded=result.total_tool_calls > 0,
            type="chat",
            mode=getattr(self, "_current_mode", "deep"),
            loop_detected=result.loop_detected,
            usage={
                "agent_steps": len(result.steps),
                "tool_calls": result.total_tool_calls,
            },
        )

    # ── L2 论文上下文管理 ──

    def set_l2_context(self, conv_id: str, paper_text: str) -> None:
        """设置论文上下文。首次使用全文，后续自动切换为结构概要。"""
        self._l2_cache[conv_id] = paper_text
        # 生成论文结构概要（纯字符串操作，零 LLM 调用）
        outline = _build_paper_outline(paper_text)
        self._l2_outline_cache[conv_id] = outline
        # 标记未使用：下次 get_l2_context 返回全文
        self._l2_outline_used[conv_id] = False

    def get_l2_context(self, conv_id: str, force_full: bool = False) -> str:
        """获取论文上下文。首次返回全文，后续返回结构概要。

        force_full=True 时始终返回全文（用于总结场景）。
        """
        full_text = self._l2_cache.get(conv_id, "")
        outline = self._l2_outline_cache.get(conv_id, "")
        used = self._l2_outline_used.get(conv_id, False)

        if not full_text:
            return ""
        if force_full:
            # 总结场景：始终返回全文
            return full_text
        if not used and outline:
            # 首次使用：返回全文，标记已用
            self._l2_outline_used[conv_id] = True
            return full_text
        if used and outline:
            # 后续使用：返回结构概要（省上下文），RAG 负责定向填充原文
            return outline
        # 降级：没有概要时返回全文
        return full_text[:6000] if used else full_text

    def clear_l2_context(self, conv_id: str) -> None:
        self._l2_cache.pop(conv_id, None)
        self._l2_outline_cache.pop(conv_id, None)
        self._l2_outline_used.pop(conv_id, None)

    # ── 取消检查 ──

    def _check_cancel(self):
        if self._cancel_event and self._cancel_event.is_set():
            raise asyncio.CancelledError("Query cancelled")

    def _result(self, **kwargs) -> dict:
        kwargs.setdefault("mode", getattr(self, "_current_mode", "chat"))
        kwargs.setdefault("model", getattr(self, "_model", ""))
        return kwargs

    _persist_lock = asyncio.Lock()

    async def _maybe_persist(self):
        if self._persistence and self.conv:
            from datetime import datetime
            self.conv.updated_at = datetime.now().isoformat()
            async with self._persist_lock:
                await self._persistence.save_full_conversation(self.conv)

    # ── 会话管理 ──

    def delete_conversation(self, conv_id: str) -> bool:
        if conv_id not in self.conversations:
            return False
        del self.conversations[conv_id]
        self._message_stores.pop(conv_id, None)
        self._l2_cache.pop(conv_id, None)
        # Also clean up archived index entries for this conversation
        try:
            archived = self._get_archived_index()
            archived.remove_by_conv_id(conv_id)
        except Exception:
            pass
        if self.active_conv_id == conv_id:
            remaining = list(self.conversations.keys())
            self.active_conv_id = remaining[0] if remaining else None
        if self._persistence:
            asyncio.ensure_future(self._persistence.delete_conversation(conv_id))
        return True

    def rename_conversation(self, conv_id: str, new_name: str) -> bool:
        if conv_id not in self.conversations:
            return False
        self.conversations[conv_id].name = new_name
        asyncio.ensure_future(self._maybe_persist())
        return True

    def add_paper_to_conv(self, pid: int | str, title: str, filename: str = "", filepath: str = ""):
        if self.conv and self.conv.add_paper(pid, title, filename, filepath):
            asyncio.ensure_future(self._maybe_persist())

    def branch_conversation(self, source_conv_id: str, at_message_index: int,
                            name: str = "") -> str | None:
        """从指定消息位置分叉一个新对话。返回新对话 ID。"""
        source = self.conversations.get(source_conv_id)
        if not source or at_message_index < 0:
            return None
        if at_message_index >= len(source.messages):
            return None

        new_id = str(uuid.uuid4())[:8]
        branched_messages = [dict(m) for m in source.messages[:at_message_index + 1]]
        branch_name = name or f"{source.name} (分支)"

        new_conv = Conversation(
            id=new_id,
            name=branch_name,
            open_papers=[dict(p) for p in source.open_papers],
            messages=branched_messages,
            current_topic=source.current_topic,
        )
        self.conversations[new_id] = new_conv
        self._message_stores[new_id] = MessageStore(SYSTEM_PROMPT)
        # 同步分支的消息到 msg_store
        for m in branched_messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "user":
                self._get_message_store(new_id).append_user(content)
            elif role == "assistant":
                self._get_message_store(new_id).append_assistant(content)
        asyncio.ensure_future(self._maybe_persist())
        return new_id

    def new_conversation(self, name: str = "新对话", conv_id: str | None = None) -> str:
        conv = Conversation(id=conv_id or str(uuid.uuid4())[:8], name=name)
        self.conversations[conv.id] = conv
        self._message_stores[conv.id] = MessageStore(SYSTEM_PROMPT)
        asyncio.ensure_future(self._maybe_persist())
        return conv.id

    def get_or_create_conversation(self, conv_id: str | None = None) -> Conversation:
        if conv_id and conv_id in self.conversations:
            self.active_conv_id = conv_id
            return self.conversations[conv_id]
        cid = conv_id or str(uuid.uuid4())[:8]
        self.active_conv_id = cid
        self.new_conversation(name="新对话", conv_id=cid)
        return self.conversations[cid]

    @property
    def conv(self) -> Conversation | None:
        return self.conversations.get(self.active_conv_id) if self.active_conv_id else None

    # ── 主流程 ──

    async def run(
        self,
        question: str,
        screen_ctx: str = "",
        history_hint: str = "",
        api_key: str = "",
        model: str = "",
        base_url: str = "",
        thinking: bool = True,   # 默认开：Qwopus 模型需 thinking 模式区分思考/回答
        on_status: Callable | None = None,
        on_token: Callable | None = None,
        on_health: Callable | None = None,
        on_agent_step: Callable | None = None,
    ) -> dict:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._thinking = thinking

        conv = self.conv
        if not conv:
            return self._result(response="请先创建对话窗口", refused=True, type="error")

        # 重复提交检查
        now = time.time()
        if question == conv.last_question and now - conv.last_question_time < 5.0:
            return self._result(duplicate=True, response="", refused=False, type="chat")

        conv.last_question = question
        conv.last_question_time = now

        self._check_cancel()

        # ── 状态: THINKING ──
        await self._session_state.transition(ChatState.THINKING)

        try:
            return await self._run_impl(
                question, screen_ctx, history_hint,
                conv, on_status, on_token, on_health, on_agent_step,
            )
        finally:
            clear_provider_cache()
            # 异常/取消时确保回到 IDLE（正常路径在 return 前已切换）
            if self._session_state.state == ChatState.THINKING:
                await self._session_state.transition(ChatState.IDLE)
            self._agent_body_streamed = False
            self._on_token = None

    async def _run_impl(
        self,
        question: str,
        screen_ctx: str,
        history_hint: str,
        conv,
        on_status: Callable | None,
        on_token: Callable | None,
        on_health: Callable | None,
        on_agent_step: Callable | None = None,
    ) -> dict:
        """run() 的实现体，由 try/finally 包裹以保证状态恢复。"""
        msg_store = self._get_message_store(conv.id)

        # 同步 conv.messages → msg_store（首次或恢复时）
        self._sync_conv_to_store(conv, msg_store)

        # 历史字符串（5 轮上下文，包含 AI 的论文介绍等回复）
        active_slice = msg_store.get_active_slice()
        history_str = _build_history_from_messages(active_slice, max_rounds=5)
        if history_hint and not history_str:
            history_str = history_hint

        # ── 检查唤醒 ──
        wake_records_text = ""
        archived_idx = self._get_archived_index()
        if archived_idx.entry_count > 0:
            compress_engine = self._get_compress_engine()
            embed_fn = self._get_embed_fn()
            wake_results = compress_engine.check_wake_needed(
                question, active_slice, archived_idx,
                embed_fn=embed_fn,
            )
            if wake_results:
                wake_parts = []
                for r in wake_results[:3]:
                    wake_parts.append(f"用户: {r['text'][:500]}")
                wake_records_text = "\n---\n".join(wake_parts)

        # ── 上下文压缩守卫 ──
        await self._guard_context(msg_store)

        # ── LLM 参数 ──
        llm_params = _build_llm_params(self._base_url)

        # L2 上下文
        # 总结类问题传递全文，其余场景 agent 通过 RAG 搜索工具自行获取
        _qtype = _detect_question_type(question)
        l2_text = self.get_l2_context(conv.id, force_full=True) if _qtype == "summary" else ""
        conv.is_first_message = False

        # ── Agent 自主循环 ──
        result = await self._run_agent(
            question=question,
            screen_ctx=screen_ctx,
            conv=conv,
            msg_store=msg_store,
            l2_text=l2_text,
            history_str=history_str,
            wake_records_text=wake_records_text,
            max_tokens=0,  # 0 = 不限制
            llm_params=llm_params,
            on_status=on_status,
            on_token=on_token,
            on_health=on_health,
            on_agent_step=on_agent_step,
        )
        resp = result.get("response", "")
        resp_clean = strip_thinking_tags(resp) if resp else resp

        if not resp_clean.strip():
            resp_clean = "抱歉，AI 暂时无法生成回答，请稍后重试。"
            result["response"] = resp_clean
            if on_token:
                on_token(resp_clean, False)

        now_ts = int(time.time() * 1000)
        msg_store.append_user(question)
        msg_store.append_assistant(resp_clean)
        # 只同步 user/assistant 消息, 过滤 system( prompt / compact_boundary / summary )
        conv.messages = [m for m in msg_store.messages if m.get("role") in ("user", "assistant")]

        await self._check_compress(msg_store)
        await self._session_state.transition(ChatState.IDLE)
        asyncio.ensure_future(self._maybe_persist())
        return result


    # ═══════════════════════════════════════════════════════════════
    # V11: LLM 有限决断循环
    # ═══════════════════════════════════════════════════════════════

    async def _llm_decision_loop(
        self, question: str, original_question: str,
        screen_ctx: str, conv, msg_store: MessageStore,
        l2_text: str, question_type: str, history_str: str,
        wake_records_text: str, max_tokens: int, llm_params: dict,
        mode_str: str,
        on_status, on_token, on_health,
    ) -> dict:
        """V11 LLM 有限决断循环: 决策→搜索→评估→回答, 最多 1 次重搜。

        替代 V10 的 Call1(分类)→RAG→GapEval→Call2(回答) 流水线。
        LLM 自己决定搜什么、搜到没、要不要补搜。
        """
        active_slice = msg_store.get_active_slice()
        search_context = ""
        loop_detected = [False]
        _verbose = config.DEBUG or os.environ.get("DEBUG_WORKFLOW") == "1"

        # ═══════════════════════════════════════
        # Phase 1: LLM 搜索决策
        # ═══════════════════════════════════════
        # 小模型检测：self._model 可能是 "deepseek-v4-pro"（默认值），
        # 实际运行的本地模型路径需要从 local_engine 获取
        _tiny = _is_tiny_model(self._model)
        if not _tiny:
            local_eng = get_local_engine()
            if local_eng and local_eng.is_loaded:
                _tiny = _is_tiny_model(str(local_eng.model_path))

        if on_status:
            on_status("deciding", "正在分析问题...")

        if _tiny:
            decision_prompt = build_tiny_decision_prompt(
                question=question,
                screen_ctx=screen_ctx,
                l2_text=l2_text,
                current_topic=conv.current_topic,
            )
        else:
            decision_prompt = build_decision_prompt_l3(
                question=question,
                screen_ctx=screen_ctx,
                l2_text=l2_text,
                current_topic=conv.current_topic,
                history_str=history_str,
                question_type=question_type,
            )
        decision_messages = list(active_slice)
        decision_messages.append({"role": "user", "content": decision_prompt})

        try:
            decision_text, decision_usage = await asyncio.to_thread(
                sync_call_llm,
                SYSTEM_PROMPT,
                decision_messages[1:],
                max_tokens=80,  # 紧凑：只够输出关键词行
                api_key=self._api_key, model=self._model,
                base_url=self._base_url,
                thinking=False,
                cancel_event=self._cancel_event,
                reuse_provider=True,
                **llm_params,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise WorkflowError("搜索决策", e)

        search_terms = _parse_decision_keywords(decision_text)

        # 回退：4B 小模型可能不遵循格式输出推理文本而非关键词
        # 推理文本本身就是"信息足够，无需搜索"的信号 → 跳过 RAG
        # 仅当 LLM 完全无输出时才从问题本身提取关键词
        if not search_terms and not decision_text.strip():
            # LLM 完全无输出（罕见）→ 从原始问题提取关键词兜底
            fallback_terms = _simple_extract_keywords(
                question, current_topic=conv.current_topic
            )
            if fallback_terms:
                search_terms = fallback_terms
                logger.info("Decision fallback: LLM produced no output, using question keywords: %s", search_terms)
        elif not search_terms and len(decision_text.strip()) > 20:
            # LLM 输出了推理文本 → 它认为信息足够，不需要搜索
            logger.info("Decision: LLM produced reasoning text (len=%d), treating as no-search-needed",
                        len(decision_text.strip()))

        if _verbose:
            import sys as _sys
            print(f"🧠 决策: terms={search_terms} tiny={_tiny} raw={decision_text[:100]}", file=_sys.stderr, flush=True)

        # Emit decision health
        if on_health:
            d_health = decision_usage.get("health", {}) if isinstance(decision_usage, dict) else {}
            on_health({
                "call": "decision",
                "timing": d_health.get("timing", {}),
                "tokens": d_health.get("tokens", {
                    "prefill_tokens": decision_usage.get("input", 0) if isinstance(decision_usage, dict) else 0,
                    "output_tokens": decision_usage.get("output", 0) if isinstance(decision_usage, dict) else 0,
                    "cache_hit_rate": None,
                }),
                "memory": d_health.get("memory", {}),
                "debug_text": f"terms={search_terms} mode={mode_str}",
            })

        # ═══════════════════════════════════════
        # Phase 2: 向量搜索
        # ═══════════════════════════════════════
        _SKIP_RAG_TYPES = {"concise", "pedagogical", "conversational", "summary"}
        if search_terms and question_type not in _SKIP_RAG_TYPES:
            if on_status:
                on_status("searching", f"检索: {', '.join(search_terms[:3])}")
            try:
                search_context = await self._do_rag_search(
                    question, question, search_terms, conv, "deep",
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                raise WorkflowError("知识库检索", e)
        elif not l2_text and question_type not in _SKIP_RAG_TYPES and not search_terms:
            # LLM 决定不搜索但有 L2 全文 → 跳过 RAG 正常
            pass

        # ═══════════════════════════════════════
        # Phase 3: 评估 + 回答（合并，最多 1 次重搜 + 1 次 thinking 重试）
        # ═══════════════════════════════════════

        # thinking 独立预算：llama.cpp 的 n_predict 统一计所有 token，
        # 无法像 Claude API 一样拆开。解决方案：
        #   - 给 llama.cpp 一个慷慨的总预算（4x 或 16384）
        #   - 应用层只数 answer token，达到预算时中止流
        _answer_budget = max_tokens
        _llm_max_tokens = max_tokens
        if self._thinking and is_local_mode(self._base_url):
            _llm_max_tokens = max(max_tokens * 4, 16384)
            logger.info("Local thinking: llama_budget=%d answer_budget=%d",
                        _llm_max_tokens, _answer_budget)

        last_assistant = _get_last_assistant_content(active_slice)
        _saved_thinking = self._thinking

        resp, usage = await self._do_answer_with_research_fallback(
            question=question,
            screen_ctx=screen_ctx,
            search_context=search_context,
            l2_text=l2_text,
            conv=conv,
            msg_store=msg_store,
            active_slice=active_slice,
            question_type=question_type,
            last_assistant=last_assistant,
            max_tokens=_llm_max_tokens,       # llama.cpp 总预算
            answer_budget=_answer_budget,      # 应用层回答预算
            tiny_model=_tiny,
            llm_params=llm_params,
            mode_str=mode_str,
            on_status=on_status,
            on_token=on_token,
            on_health=on_health,
            loop_detected=loop_detected,
        )

        # ═══════════════════════════════════════
        # Phase 4: 后处理
        # ═══════════════════════════════════════

        # 恢复 thinking 设置
        self._thinking = _saved_thinking

        if loop_detected[0] and not resp.strip():
            resp = "（模型回复出现重复，请点击下方按钮重新生成）"

        resp_clean = strip_thinking_tags(resp) if resp else resp

        # ── 小模型推理前缀剥离 ──
        # tiny+on_token → _tiny_sync_answer 已在内部完成剥离，此处跳过
        if _tiny and resp_clean and not (_tiny and on_token):
            before = len(resp_clean)
            resp_clean = strip_reasoning_preamble(resp_clean)
            if len(resp_clean) != before:
                resp = resp_clean  # 同步更新 raw resp
                logger.info("Stripped reasoning preamble: %d → %d chars", before, len(resp_clean))
                if _verbose:
                    import sys as _sys
                    print(f"✂️ 前缀剥离: {before} → {len(resp_clean)} chars", file=_sys.stderr, flush=True)

        # ── thinking 空回答重试：模型可能将所有 token 花在思考上 ──
        # Claude Code 的做法是 thinking 有独立预算；DeepSeek API 共享 max_tokens，
        # 小模型可能思考完就没额度输出了。此时自动关 thinking 重试一次。
        if not resp_clean.strip() and _saved_thinking:
            logger.warning(
                "Answer empty after stripping thinking tags (raw_len=%d). "
                "Retrying with thinking disabled.", len(resp or "")
            )
            if on_status:
                on_status("generating", "正在重新生成（关闭思考模式）...")
            self._thinking = False
            loop_detected[0] = False

            # 清除上次可能残留的 KV cache
            clear_provider_cache()

            # 重试不启用 thinking，无需分离预算
            resp, usage = await self._do_answer_with_research_fallback(
                question=question,
                screen_ctx=screen_ctx,
                search_context=search_context,
                l2_text=l2_text,
                conv=conv,
                msg_store=msg_store,
                active_slice=active_slice,
                question_type=question_type,
                last_assistant=last_assistant,
                max_tokens=_answer_budget,
                answer_budget=_answer_budget,
                tiny_model=_tiny,
                llm_params=llm_params,
                mode_str=mode_str,
                on_status=on_status,
                on_token=on_token,
                on_health=on_health,
                loop_detected=loop_detected,
            )
            self._thinking = _saved_thinking
            resp_clean = strip_thinking_tags(resp) if resp else resp

        if not resp_clean.strip():
            resp_clean = "抱歉，AI 暂时无法生成回答，请稍后重试。"
            resp = resp_clean  # 确保返回给前端的也是兜底文案，而非空串
            # 主动推送给前端：handler 只在 total_tokens==0 时才补发，
            # 如果 thinking 消耗了 token 但回答为空，前端永远收不到
            if on_token:
                on_token(resp_clean, False)

        now_ts = int(time.time() * 1000)
        msg_store.append_user(original_question)
        msg_store.append_assistant(resp_clean)
        conv.messages.append({"role": "user", "content": original_question, "timestamp": now_ts})
        conv.messages.append({"role": "assistant", "content": resp_clean, "timestamp": now_ts, "mode": mode_str, "model": self._model})
        try:
            await self._maybe_persist()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise WorkflowError("对话记录保存", e)

        await self._check_compress(msg_store)
        clear_provider_cache()
        await self._session_state.transition(ChatState.IDLE)

        return self._result(
            response=resp, usage=usage,
            refused=False,
            type="chat" if mode_str == "chat" else "search",
            topic=conv.current_topic,
            loop_detected=loop_detected[0],
        )

    async def _do_answer_with_research_fallback(
        self, question: str, screen_ctx: str, search_context: str,
        l2_text: str, conv, msg_store: MessageStore,
        active_slice: list[dict], question_type: str,
        last_assistant: str, max_tokens: int, answer_budget: int,
        tiny_model: bool, llm_params: dict, mode_str: str,
        on_status, on_token, on_health, loop_detected: list,
    ) -> tuple[str, dict]:
        """生成回答，如 LLM 请求则触发一次补搜后重新生成。

        小模型（≤2B）使用 sync 路径：流式 buffer 只有 80 字符，
        preamble 可能跨边界导致 strip 失败。sync 拿到完整回答后再预处理。"""
        _loop_cancel = [False]
        recent_context = _build_recent_context(active_slice, max_exchanges=2)

        # ── 小模型：sync 路径，完整回答后再 strip preamble ──
        if tiny_model and on_token:
            return await self._tiny_sync_answer(
                question, screen_ctx, search_context, l2_text, conv,
                msg_store, active_slice, last_assistant, max_tokens,
                llm_params, mode_str, on_status, on_token, on_health,
                loop_detected,
            )

        # ── 构建 prompt（小模型用极简版） ──
        if tiny_model:
            answer_prompt = build_tiny_answer_prompt(
                question=question,
                screen_ctx=screen_ctx,
                search_context=search_context,
                l2_text=l2_text,
                current_topic=conv.current_topic,
                recent_context=recent_context,
            )
        else:
            answer_prompt = build_eval_answer_prompt_l3(
                question=question,
                screen_ctx=screen_ctx,
                search_context=search_context,
                l2_text=l2_text,
                current_topic=conv.current_topic,
                question_type=question_type,
                last_assistant_reply=last_assistant,
            )

        answer_messages = list(active_slice)
        if search_context:
            answer_messages.append({"role": "system", "content": f"【知识库检索结果】\n{search_context}"})
        elif mode_str == "deep" and not l2_text:
            answer_messages.append({"role": "system", "content": _build_no_kb_notice()})
        answer_messages.append({"role": "user", "content": answer_prompt})

        # ── 流式生成 + 重搜信号检测 ──
        _research_buf: list[str] = []
        _research_done = False
        _research_triggered = [False]
        _research_terms = [""]
        _loop_token_count = 0

        def _stream_collect(t: str, is_thinking: bool = False):
            nonlocal _research_done, _loop_token_count
            resp_parts.append(t)

            if is_thinking:
                if on_token:
                    on_token(t, True)
                return

            if not _research_done:
                _research_buf.append(t)
                buffered = "".join(_research_buf)
                if len(buffered) > 80:
                    _research_done = True
                    terms = _parse_research_signal(buffered)
                    if terms:
                        _research_triggered[0] = True
                        _research_terms[0] = terms
                        _loop_cancel[0] = True
                        return
                    # 指令背诵检测：小模型可能背诵 prompt 而非回答
                    if detect_instruction_echo(buffered):
                        _research_triggered[0] = True
                        _research_terms[0] = "__echo__"
                        _loop_cancel[0] = True
                        logger.warning("Instruction echo detected, retrying with simplified prompt")
                        return
                    # 非重搜 → 冲刷缓冲（小模型先剥离推理前缀）
                    if on_token:
                        _flush = "".join(_research_buf)
                        if tiny_model:
                            _stripped = strip_reasoning_preamble(_flush)
                            if len(_stripped) != len(_flush):
                                logger.info("Stream preamble stripped: %d → %d chars",
                                            len(_flush), len(_stripped))
                                _flush = _stripped
                        on_token(_flush, False)
                    _research_buf.clear()
                    _loop_token_count = 0
            else:
                if on_token:
                    on_token(t, False)
                _loop_token_count += 1

            # 回答 token 预算控制：thinking token 不计入，
            # llama.cpp 给了慷慨的总预算 (max_tokens)，应用层只数 answer token
            if _loop_token_count > 0 and _loop_token_count >= answer_budget:
                _loop_cancel[0] = True
                logger.info("Answer budget reached: %d/%d tokens", _loop_token_count, answer_budget)
                return

            # 循环检测
            if on_token and not loop_detected[0] and _loop_token_count > 0 and _loop_token_count % 30 == 0:
                partial = "".join(resp_parts[-90:])
                if len(partial) > 200 and _detect_loop(partial):
                    loop_detected[0] = True
                    _loop_cancel[0] = True
                    logger.warning("Loop detected mid-stream, cancelling")
                    if on_status:
                        on_status("loop_warning", "检测到回复出现重复，已暂停输出")

        try:
            if on_token:
                resp_parts: list[str] = []
                usage = await stream_call_llm(
                    SYSTEM_PROMPT,
                    answer_messages[1:],
                    max_tokens=max_tokens or config.LLM_MAX_TOKENS,
                    api_key=self._api_key, model=self._model,
                    base_url=self._base_url,
                    thinking=self._thinking, on_token=_stream_collect,
                    reuse_provider=True,
                    _loop_cancel=_loop_cancel,
                    **llm_params,
                )
                resp = "".join(resp_parts)
            else:
                resp, usage = sync_call_llm(
                    SYSTEM_PROMPT,
                    answer_messages[1:],
                    max_tokens=max_tokens or config.LLM_MAX_TOKENS,
                    api_key=self._api_key, model=self._model,
                    base_url=self._base_url,
                    cancel_event=self._cancel_event,
                    reuse_provider=True,
                    **llm_params,
                )
                # 检查重搜信号
                terms = _parse_research_signal(resp[:200] if resp else "")
                if terms:
                    _research_triggered[0] = True
                    _research_terms[0] = terms
                loop_detected[0] = bool(resp and _detect_loop(resp))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise WorkflowError("回答生成", e)

        # ── 冲刷残留缓冲：短回答 (<80 chars) 不会触发 _research_done，需手动 flush ──
        if on_token and not _research_done and _research_buf:
            for bt in _research_buf:
                on_token(bt, False)
            _research_buf.clear()

        # ── 是否需要补搜？ ──
        if _research_triggered[0] and _research_terms[0] == "__echo__":
            # 指令背诵检测触发 → 用极简 prompt + 提高 temperature 重试
            logger.info("Instruction echo retry: using simplified prompt")
            if on_status:
                on_status("generating", "正在重新生成（简化模式）...")

            echo_answer_prompt = build_tiny_answer_prompt(
                question=question,
                screen_ctx=screen_ctx,
                search_context=search_context,
                l2_text=l2_text,
                current_topic=conv.current_topic,
                recent_context=recent_context,
            ) + "\n只输出回答。不要复述任何指令文字。"

            echo_messages = list(active_slice)
            if search_context:
                echo_messages.append({"role": "system", "content": f"【知识库检索结果】\n{search_context}"})
            echo_messages.append({"role": "user", "content": echo_answer_prompt})

            # 提高 temperature 打破背诵循环
            echo_params = dict(llm_params)
            echo_params["temperature"] = max(llm_params.get("temperature", 0.05), 0.3)

            _loop_cancel2 = [False]
            try:
                if on_token:
                    resp_parts2: list[str] = []
                    def _echo_collect(t: str, is_thinking: bool = False):
                        resp_parts2.append(t)
                        if on_token and not loop_detected[0]:
                            on_token(t, is_thinking)
                    usage = await stream_call_llm(
                        SYSTEM_PROMPT,
                        echo_messages[1:],
                        max_tokens=max_tokens or config.LLM_MAX_TOKENS,
                        api_key=self._api_key, model=self._model,
                        base_url=self._base_url,
                        thinking=False, on_token=_echo_collect,
                        reuse_provider=False,
                        _loop_cancel=_loop_cancel2,
                        **echo_params,
                    )
                    resp = "".join(resp_parts2)
                else:
                    resp, usage = sync_call_llm(
                        SYSTEM_PROMPT,
                        echo_messages[1:],
                        max_tokens=max_tokens or config.LLM_MAX_TOKENS,
                        api_key=self._api_key, model=self._model,
                        base_url=self._base_url,
                        cancel_event=self._cancel_event,
                        reuse_provider=False,
                        **echo_params,
                    )
            except Exception as e:
                raise WorkflowError("回答生成 (echo retry)", e)

        elif _research_triggered[0] and _research_terms[0]:
            logger.info("Re-search triggered: %s", _research_terms[0])
            if on_status:
                on_status("searching", f"补充检索: {_research_terms[0][:50]}...")

            extra_terms = [t.strip() for t in re.split(r"[,，、;；]+", _research_terms[0]) if t.strip()]
            try:
                search_context2 = await self._do_rag_search(
                    question, question, extra_terms[:3], conv, "deep",
                )
            except Exception:
                search_context2 = ""

            if search_context2:
                search_context = search_context + "\n---\n【补充检索】\n" + search_context2

            # 重建 prompt（标志：不再请求重搜）
            if tiny_model:
                answer_prompt2 = build_tiny_answer_prompt(
                    question=question,
                    screen_ctx=screen_ctx,
                    search_context=search_context,
                    l2_text=l2_text,
                    current_topic=conv.current_topic,
                    recent_context=recent_context,
                ) + "\n\n不要再请求补充检索，直接回答。"
            else:
                answer_prompt2 = build_eval_answer_prompt_l3(
                    question=question,
                    screen_ctx=screen_ctx,
                    search_context=search_context,
                    l2_text=l2_text,
                    current_topic=conv.current_topic,
                    question_type=question_type,
                    last_assistant_reply=last_assistant,
                ) + "\n\n注意：这是补充检索后的重新回答。不要再请求补充检索，直接基于现有信息回答。"

            answer_messages2 = list(active_slice)
            if search_context:
                answer_messages2.append({"role": "system", "content": f"【知识库检索结果】\n{search_context}"})
            answer_messages2.append({"role": "user", "content": answer_prompt2})

            _loop_cancel2 = [False]
            try:
                if on_token:
                    resp_parts2: list[str] = []
                    def _collect2(t: str, is_thinking: bool = False):
                        resp_parts2.append(t)
                        if on_token and not loop_detected[0]:
                            on_token(t, is_thinking)
                            if not is_thinking and len(resp_parts2) % 30 == 0 and len(resp_parts2) >= 90:
                                partial = "".join(resp_parts2)
                                if len(partial) > 200 and _detect_loop(partial):
                                    loop_detected[0] = True
                                    _loop_cancel2[0] = True
                                    if on_status:
                                        on_status("loop_warning", "检测到回复出现重复")
                    usage = await stream_call_llm(
                        SYSTEM_PROMPT,
                        answer_messages2[1:],
                        max_tokens=max_tokens or config.LLM_MAX_TOKENS,
                        api_key=self._api_key, model=self._model,
                        base_url=self._base_url,
                        thinking=self._thinking, on_token=_collect2,
                        reuse_provider=True,
                        _loop_cancel=_loop_cancel2,
                        **llm_params,
                    )
                    resp = "".join(resp_parts2)
                else:
                    resp, usage = sync_call_llm(
                        SYSTEM_PROMPT,
                        answer_messages2[1:],
                        max_tokens=max_tokens or config.LLM_MAX_TOKENS,
                        api_key=self._api_key, model=self._model,
                        base_url=self._base_url,
                        cancel_event=self._cancel_event,
                        reuse_provider=True,
                        **llm_params,
                    )
                    loop_detected[0] = bool(resp and _detect_loop(resp))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                raise WorkflowError("回答生成 (补搜)", e)

        if on_health:
            a_health = usage.get("health", {}) if isinstance(usage, dict) else {}
            on_health({
                "call": "answer",
                "timing": a_health.get("timing", {}),
                "tokens": a_health.get("tokens", {
                    "prefill_tokens": usage.get("input", 0) if isinstance(usage, dict) else 0,
                    "output_tokens": usage.get("output", 0) if isinstance(usage, dict) else 0,
                    "cache_hit_rate": None,
                }),
                "memory": a_health.get("memory", {}),
                "debug_text": (resp or "")[:800],
            })

        return resp, usage

    async def _tiny_sync_answer(
        self, question, screen_ctx, search_context, l2_text, conv,
        msg_store, active_slice, last_assistant, max_tokens,
        llm_params, mode_str, on_status, on_token, on_health,
        loop_detected,
    ) -> tuple[str, dict]:
        """小模型（≤2B）sync 回答：完整拿到回复 → strip preamble → 发送。"""
        recent_context = _build_recent_context(active_slice, max_exchanges=2)
        answer_prompt = build_tiny_answer_prompt(
            question=question,
            screen_ctx=screen_ctx,
            search_context=search_context,
            l2_text=l2_text,
            current_topic=conv.current_topic,
            recent_context=recent_context,
        )

        answer_messages = list(active_slice)
        if search_context:
            answer_messages.append({"role": "system", "content": f"【知识库检索结果】\n{search_context}"})
        elif mode_str == "deep" and not l2_text:
            answer_messages.append({"role": "system", "content": _build_no_kb_notice()})
        answer_messages.append({"role": "user", "content": answer_prompt})

        try:
            resp, usage = await asyncio.to_thread(
                sync_call_llm,
                SYSTEM_PROMPT,
                answer_messages[1:],
                max_tokens=max_tokens or config.LLM_MAX_TOKENS,
                api_key=self._api_key, model=self._model,
                base_url=self._base_url,
                thinking=self._thinking,
                cancel_event=self._cancel_event,
                reuse_provider=True,
                **llm_params,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise WorkflowError("回答生成 (tiny sync)", e)

        # Strip preamble before sending to frontend
        resp_clean = strip_thinking_tags(resp) if resp else resp
        if resp_clean:
            before = len(resp_clean)
            resp_clean = strip_reasoning_preamble(resp_clean)
            if len(resp_clean) != before:
                resp = resp_clean
                logger.info("Tiny sync: stripped preamble %d → %d chars", before, len(resp_clean))

        # 模拟流式输出：将完整回复按 ~3 字符切块，逐块推送
        if on_token and resp:
            chunk_size = 3
            for i in range(0, len(resp), chunk_size):
                on_token(resp[i:i+chunk_size], False)
                # 每 10 块暂停 15ms，模拟自然流式节奏
                if i % (chunk_size * 10) == 0 and i > 0:
                    await asyncio.sleep(0.015)

        if on_health:
            a_health = usage.get("health", {}) if isinstance(usage, dict) else {}
            on_health({
                "call": "answer",
                "timing": a_health.get("timing", {}),
                "tokens": a_health.get("tokens", {
                    "prefill_tokens": usage.get("input", 0) if isinstance(usage, dict) else 0,
                    "output_tokens": usage.get("output", 0) if isinstance(usage, dict) else 0,
                    "cache_hit_rate": None,
                }),
                "memory": a_health.get("memory", {}),
                "debug_text": (resp or "")[:800],
            })

        return resp, usage

    # ── 消息同步 ──

    # 恢复旧会话时保留最近 N 轮，其余插入 compact_boundary 防止上下文爆炸
    _RESTORE_MAX_RECENT_ROUNDS = 20

    def _sync_conv_to_store(self, conv: Conversation, msg_store: MessageStore) -> None:
        """将 Conversation.messages 同步到 MessageStore（仅当 store 为空时）。

        若会话消息过多（>20 轮），自动在中间插入 compact_boundary 截断旧历史。
        """
        if msg_store.raw_count > 1:  # 已有内容（至少 L0）
            return

        messages = conv.messages
        # 统计 user+assistant 轮次
        conversation_messages = [
            m for m in messages
            if m.get("role") in ("user", "assistant")
            and m.get("subtype") != "compact_boundary"
        ]
        total_rounds = len(conversation_messages) // 2

        # 判断是否需要截断：轮次过多 或 已有 compact_boundary
        has_boundary = any(m.get("subtype") == "compact_boundary" for m in messages)
        if has_boundary:
            # 已有 boundary，直接按原样还原
            for m in messages:
                role = m.get("role", "")
                subtype = m.get("subtype", "")
                content = m.get("content", "")
                if subtype == "compact_boundary":
                    msg_store.insert_boundary()
                elif subtype == "summary":
                    msg_store.insert_summary(content)
                elif role == "user":
                    msg_store.append_user(content)
                elif role == "assistant":
                    msg_store.append_assistant(content)
        elif total_rounds > self._RESTORE_MAX_RECENT_ROUNDS:
            # 无 boundary + 轮次过多 → 截断旧消息，保持最近 20 轮活跃
            recent_count = self._RESTORE_MAX_RECENT_ROUNDS * 2
            old_messages = conversation_messages[:-recent_count]
            recent = conversation_messages[-recent_count:]

            # 旧消息批量归档（不生成摘要，首轮 query 会通过入口守卫触发压缩）
            # 直接插入 compact_boundary 标记截断
            msg_store.insert_boundary()
            # 同步到 conv.messages 确保持久化
            ts = int(time.time() * 1000)
            conv.messages.append({
                "role": "system", "subtype": "compact_boundary",
                "content": "", "timestamp": ts,
            })
            logger.info("Auto-truncate: %d rounds → keeping recent %d rounds (%.0f%% dropped)",
                        total_rounds, self._RESTORE_MAX_RECENT_ROUNDS,
                        (1 - recent_count / len(conversation_messages)) * 100)

            # 还原最近 N 轮
            for m in recent:
                role = m.get("role", "")
                if role == "user":
                    msg_store.append_user(m.get("content", ""))
                elif role == "assistant":
                    msg_store.append_assistant(m.get("content", ""))
        else:
            # 轮次少，直接还原
            for m in messages:
                role = m.get("role", "")
                subtype = m.get("subtype", "")
                content = m.get("content", "")
                if subtype == "compact_boundary":
                    msg_store.insert_boundary()
                elif subtype == "summary":
                    msg_store.insert_summary(content)
                elif role == "user":
                    msg_store.append_user(content)
                elif role == "assistant":
                    msg_store.append_assistant(content)

    # ── RAG 检索 ──

    async def _do_rag_search(
        self, question: str, corrected: str,
        search_terms: list[str], conv: Conversation,
        mode: str, is_correction: bool = False,
        paper_ids: list[int] | None = None,
    ) -> str:
        """执行 RAG 检索，返回上下文文本。

        paper_ids 不为空时只在这些论文内搜（深度搜索/术语澄清）。
        """
        search_ids = paper_ids if paper_ids else conv.paper_ids()
        query = f"{corrected} {' '.join(search_terms[:2])}" if corrected or search_terms else question

        if is_correction:
            topk, chunk_len, max_chunks = 15, 800, 24
        else:  # deep（quick 已合并入 deep）
            # 4B 模型：极简 chunk 数，避免模型被海量信息淹没后逐条复述
            topk, chunk_len, max_chunks = 6, 500, 6

        self._check_cancel()

        r1 = None
        best_dist = 999.0

        if is_correction:
            search_plan = [(query, None, topk)]
        elif paper_ids:
            # 聚焦搜索：只在指定论文内搜，不做全库回退
            search_plan = [(query, paper_ids, topk)]
        else:
            search_plan = [
                (query, search_ids, topk),
                (query, None, topk),
                (question, None, 5),
            ]

        for q, ids, tk in search_plan:
            raw = _vector_search(q, top_k=tk, paper_ids=ids)
            ranked = _mmr_rerank(raw, top_k=tk) if raw else []
            if ranked and ranked[0]["dist"] < best_dist:
                r1, best_dist = ranked, ranked[0]["dist"]
            if best_dist < SEARCH_DIST_THRESHOLD:
                break

        if not r1 or best_dist >= SEARCH_DIST_THRESHOLD:
            return ""

        # 章节多样性：避免所有 chunks 集中在引言部分
        # 利用 chunk_index 作为章节代理（索引顺序 = 论文顺序: 引言→实验→结论）
        selected = _select_with_section_diversity(r1, max_chunks)
        parts = []
        for i in selected:
            if i < len(r1):
                pid = r1[i]["meta"].get("paper_id", r1[i]["meta"].get("doc_id", "?"))
                fname = r1[i]["meta"].get("filename", "")
                sec = r1[i]["meta"].get("section_type", "?")
                parts.append(f"【来源: Paper#{pid}, {fname}, {sec}】\n{r1[i]['doc'][:chunk_len]}")

        return "\n---\n".join(parts)

    # ── 上下文守卫 ──

    async def _guard_context(self, msg_store: MessageStore) -> None:
        """入口守卫：classify/answer 前若上下文超限，先静默压缩再继续。

        不改变会话状态（保持在 THINKING），不像 _check_compress 那样
        transition 到 COMPRESSING/IDLE。
        """
        compress_engine = self._get_compress_engine()
        active_slice = msg_store.get_active_slice()
        if not compress_engine.should_compress(active_slice):
            return

        logger.info("Context guard: compressing before classify (%d tok)", msg_store.estimate_tokens())
        archived_idx = self._get_archived_index()
        try:
            result = await compress_engine.compress(
                active_slice, msg_store, archived_idx,
                on_progress=None,
                api_key=self._api_key,
                model=self._model,
                base_url=self._base_url,
            )
            # 同步到 conv（兼容持久化）
            if self.conv and result.get("summary"):
                ts = int(time.time() * 1000)
                self.conv.messages.append({
                    "role": "system", "subtype": "compact_boundary",
                    "content": "", "timestamp": ts,
                })
                self.conv.messages.append({
                    "role": "system", "subtype": "summary",
                    "content": result["summary"], "timestamp": ts,
                })
            logger.info("Context guard: compressed %d rounds, new ctx=%d tok",
                        result.get("rounds_compressed", 0), msg_store.estimate_tokens())
        except Exception as e:
            logger.warning("Context guard compress failed (non-fatal): %s", e)

    # ── 压缩 ──

    async def _check_compress(self, msg_store: MessageStore) -> None:
        """回复完成后检查是否需要压缩"""
        compress_engine = self._get_compress_engine()
        active_slice = msg_store.get_active_slice()
        tokens = msg_store.active_slice_tokens()
        should = compress_engine.should_compress(active_slice)
        if config.DEBUG or os.environ.get("DEBUG_WORKFLOW") == "1":
            import sys as _s
            print(f"📦 压缩检查: tokens={tokens} n_ctx={compress_engine._n_ctx} should={should}",
                  file=_s.stderr, flush=True)
        if should:
            # 异步触发压缩（不阻塞回复）
            asyncio.ensure_future(self._do_compress(msg_store))

    async def _do_compress(self, msg_store: MessageStore) -> None:
        """执行后台压缩"""
        try:
            await self._session_state.transition(
                ChatState.COMPRESSING,
                {"verb": "正在优化上下文...", "progress": 0},
            )

            compress_engine = self._get_compress_engine()
            archived_idx = self._get_archived_index()
            active_slice = msg_store.get_active_slice()

            def on_progress(pct: int, verb: str):
                self._session_state.set_compress_progress(pct)

            result = await compress_engine.compress(
                active_slice, msg_store, archived_idx,
                on_progress=on_progress,
                api_key=self._api_key,
                model=self._model,
                base_url=self._base_url,
            )

            # 内存安全监测 (Section 5.1)
            active_tokens = msg_store.active_slice_tokens()
            conv_len = len(self.conv.messages) if self.conv else 0
            archived_count = archived_idx.entry_count
            boundary_count = msg_store.get_boundary_count()
            # RSS 内存（psutil 不可用时跳过）
            mem_mb = 0.0
            try:
                import psutil
                mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)
            except (ImportError, Exception):
                pass
            logger.info(
                "[MEM] process=%.0fMB msgs=%d active_tok=%d archived=%d boundaries=%d",
                mem_mb, conv_len, active_tokens, archived_count, boundary_count,
            )
            # 告警: >500 条消息且没有 compact_boundary
            if conv_len > 500 and boundary_count == 0:
                logger.warning(
                    "[MEM] WARNING: %d messages without compact_boundary — possible memory leak",
                    conv_len,
                )

            # 同步到 conv（兼容持久化）
            if self.conv:
                if result.get("summary"):
                    ts = int(time.time() * 1000)
                    self.conv.messages.append({
                        "role": "system",
                        "subtype": "compact_boundary",
                        "content": "",
                        "timestamp": ts,
                    })
                    self.conv.messages.append({
                        "role": "system",
                        "subtype": "summary",
                        "content": result["summary"],
                        "timestamp": ts,
                    })

            # 处理 pending queue — 逐个执行排队的回调
            pending = await self._session_state.process_pending()
            for proc in pending:
                logger.info("Re-processing queued message after compress")
                try:
                    await proc()
                except Exception as proc_err:
                    logger.error("Failed to re-process queued message: %s", proc_err)

            await self._session_state.transition(ChatState.IDLE)

        except Exception as e:
            logger.error("Compress failed: %s", e)
            # 压缩失败也要处理 pending queue，避免排队消息永久丢失
            pending = await self._session_state.process_pending()
            for proc in pending:
                logger.info("Re-processing queued message after failed compress")
                try:
                    await proc()
                except Exception as proc_err:
                    logger.error("Failed to re-process queued message: %s", proc_err)
            await self._session_state.transition(ChatState.IDLE)

    # ── 辅助方法 ──

    def _detect_correction(self, question: str) -> bool:
        patterns = [
            "不对", "你错了", "搞错了", "错了", "不是这样", "不是这个",
            "你说错了", "不准确", "有误", "错误", "不正确",
            "重新查", "再查", "重新搜", "你再看看", "确认一下",
            "不对吧", "你确定", "更正", "纠正",
        ]
        return any(p in question for p in patterns)

    def _apply_confidence_fallback(self, question: str, mode: str, fail_count: int) -> str:
        # 合并 quick/deep 后简化为二元决策：chat 或 deep
        # fail_count >= 2 或含学术关键词 → 至少 deep
        if fail_count >= 2 and mode == "chat":
            return "deep"

        if self._detect_correction(question):
            return "deep"

        paper_keywords = ["论文", "paper", "文献", "研究", "纳米", "催化", "材料",
                          "合成", "表征", "性能", "机理", "反应", "结构"]
        if any(kw in question for kw in paper_keywords) and mode == "chat":
            return "deep"

        pronoun_keywords = ["它", "这个", "那个", "上面", "前面", "刚才", "之前"]
        if any(kw in question for kw in pronoun_keywords) and mode == "chat":
            return "deep"

        greeting_patterns = ["你好", "hello", "hi ", "在吗", "谢谢", "你是谁", "早", "晚安"]
        if any(g in question.lower() for g in greeting_patterns):
            return "chat"

        return mode

    async def _meta(self, q: str, conv: Conversation, msg_store: MessageStore) -> dict:
        resp = (
            f"**会话信息**\n\n"
            f"- 今日: {_today()}\n"
            f"- 话题: {conv.current_topic or '无'}\n"
            f"- 打开论文: {len(conv.open_papers)} 篇\n"
            f"{conv.paper_titles_str() or '  暂无'}\n"
            f"- 对话轮次: {len(conv.messages) // 2} 轮\n"
        )
        ts = int(time.time() * 1000)
        msg_store.append_user(q)
        msg_store.append_assistant(resp)
        conv.messages.append({"role": "user", "content": q, "timestamp": ts})
        conv.messages.append({"role": "assistant", "content": resp, "timestamp": ts, "mode": self._current_mode, "model": self._model})
        await self._maybe_persist()
        return self._result(response=resp, refused=False, type="meta")

    async def _term(self, q: str, corrected: str, st: list[str], conv: Conversation,
                    msg_store: MessageStore, on_status=None, on_token=None) -> dict:
        self._check_cancel()
        if on_status:
            on_status("searching", f"正在查询术语「{q}」...")
        r = _vector_search(" ".join([s for s in st[:3] if s.strip()]) if st else q, top_k=5, paper_ids=conv.paper_ids())
        r = _mmr_rerank(r, top_k=5, lambda_mmr=0.5) if r else []

        if not r or r[0]["dist"] >= SEARCH_DIST_THRESHOLD:
            # 术语查询无结果 → refused=True，不存消息，由主流程回落 RAG+Call2
            return self._result(response="", refused=True, expanded=False, type="term")

        ctx_text = "\n---\n".join([
            f"【来源: Paper#{r[i]['meta'].get('paper_id', r[i]['meta'].get('doc_id', '?'))}, "
            f"{r[i]['meta'].get('filename', '')}, "
            f"{r[i]['meta'].get('section_type', '?')}】\n{r[i]['doc'][:400]}"
            for i in range(min(3, len(r)))
        ])

        active_slice = msg_store.get_active_slice()
        answer_messages = list(active_slice)
        answer_messages.append({"role": "system", "content": f"【知识库】\n{ctx_text}"})
        answer_messages.append({"role": "user", "content": f"请简洁解释术语「{q}」"})

        if on_token:
            resp_parts = []
            def collect(t, is_thinking: bool = False):
                resp_parts.append(t)
                on_token(t, is_thinking)
            await stream_call_llm(
                SYSTEM_PROMPT, answer_messages[1:],
                max_tokens=400, api_key=self._api_key, model=self._model,
                base_url=self._base_url, on_token=collect,
                reuse_provider=True,
            )
            resp = "".join(resp_parts)
        else:
            resp, _ = sync_call_llm(
                SYSTEM_PROMPT, answer_messages[1:],
                max_tokens=400, api_key=self._api_key, model=self._model,
                base_url=self._base_url, cancel_event=self._cancel_event,
                reuse_provider=True,
            )

        resp = strip_thinking_tags(resp) if resp else resp
        # 空输出说明 LLM 无法基于检索结果解释 → refused，回落 RAG+Call2
        term_failed = not resp.strip()
        if term_failed:
            resp = f"抱歉，无法解释术语「{q}」，请稍后重试。"

        ts = int(time.time() * 1000)
        msg_store.append_user(q)
        msg_store.append_assistant(resp)
        conv.messages.append({"role": "user", "content": q, "timestamp": ts})
        conv.messages.append({"role": "assistant", "content": resp, "timestamp": ts, "mode": self._current_mode, "model": self._model})
        await self._maybe_persist()
        return self._result(response=resp, refused=term_failed, expanded=False, type="term")

    async def _ref(self, n: int, st: list[str], q: str, conv: Conversation,
                   msg_store: MessageStore, mt: int | None,
                   on_status=None, on_token=None) -> dict:
        self._check_cancel()
        if on_status:
            on_status("searching", f"正在检索参考文献 [{n}]...")

        search_queries = [f"reference {n}", f"[{n}]", f"参考文献 {n}"]
        if st:
            search_queries = st[:3] + search_queries

        r, best_dist = None, SEARCH_DIST_THRESHOLD
        for sq in search_queries:
            for paper_scope in [conv.paper_ids(), None]:
                raw = _vector_search(sq, top_k=5, paper_ids=paper_scope)
                if raw and raw[0]["dist"] < best_dist:
                    r, best_dist = _mmr_rerank(raw, top_k=5, lambda_mmr=0.5), raw[0]["dist"]
                if best_dist < 0.35:
                    break
            if best_dist < 0.35:
                break

        if not r or best_dist >= SEARCH_DIST_THRESHOLD:
            resp = f"知识库中未检索到参考文献 [{n}]。"
            ts = int(time.time() * 1000)
            msg_store.append_user(q)
            msg_store.append_assistant(resp)
            conv.messages.append({"role": "user", "content": q, "timestamp": ts})
            conv.messages.append({"role": "assistant", "content": resp, "timestamp": ts, "mode": self._current_mode, "model": self._model})
            await self._maybe_persist()
            return self._result(response=resp, refused=True, type="ref")

        ctx_text = "\n---\n".join([
            f"【来源: Paper#{r[i]['meta'].get('paper_id', r[i]['meta'].get('doc_id', '?'))}, "
            f"{r[i]['meta'].get('filename', '')}】\n{r[i]['doc'][:500]}"
            for i in range(min(2, len(r)))
        ])

        active_slice = msg_store.get_active_slice()
        answer_messages = list(active_slice)
        answer_messages.append({"role": "system", "content": f"【知识库】\n{ctx_text}"})
        answer_messages.append({"role": "user", "content": f"参考文献[{n}]的详细信息"})

        if on_token:
            resp_parts = []
            def collect(t, is_thinking: bool = False):
                resp_parts.append(t)
                on_token(t, is_thinking)
            await stream_call_llm(
                SYSTEM_PROMPT, answer_messages[1:],
                max_tokens=mt or config.LLM_MAX_TOKENS, api_key=self._api_key, model=self._model,
                base_url=self._base_url, on_token=collect,
                reuse_provider=True,
            )
            resp = "".join(resp_parts)
        else:
            resp, _ = sync_call_llm(
                SYSTEM_PROMPT, answer_messages[1:],
                max_tokens=mt or config.LLM_MAX_TOKENS, api_key=self._api_key, model=self._model,
                base_url=self._base_url, cancel_event=self._cancel_event,
                reuse_provider=True,
            )

        resp = strip_thinking_tags(resp) if resp else resp
        if not resp.strip():
            resp = f"抱歉，无法获取参考文献 [{n}] 的信息，请稍后重试。"

        ts = int(time.time() * 1000)
        msg_store.append_user(q)
        msg_store.append_assistant(resp)
        conv.messages.append({"role": "user", "content": q, "timestamp": ts})
        conv.messages.append({"role": "assistant", "content": resp, "timestamp": ts, "mode": self._current_mode, "model": self._model})
        await self._maybe_persist()
        return self._result(response=resp, refused=False, type="ref")

    async def _recommend(self, q: str, c: str, conv: Conversation,
                         msg_store: MessageStore, mt: int | None,
                         on_status=None, on_token=None) -> dict:
        self._check_cancel()
        if on_status:
            on_status("searching", "正在分析论文关联，生成推荐...")
        r1 = _vector_search(c, top_k=10, paper_ids=conv.paper_ids())
        r1 = _mmr_rerank(r1, top_k=8, lambda_mmr=0.5) if r1 else []

        papers_map: dict = defaultdict(list)
        for r_ in r1:
            pid = r_["meta"].get("paper_id", r_["meta"].get("doc_id", "?"))
            papers_map[pid].append(r_)

        ctx_parts = []
        for pid, chunks in list(papers_map.items())[:5]:
            chunk_lines = [f"  [{ch['meta'].get('section_type', '?')}] {ch['doc'][:200]}" for ch in chunks[:2]]
            fname = chunks[0]['meta'].get('filename', '') if chunks else ''
            ctx_parts.append(f"Paper#{pid} ({fname}):\n" + "\n".join(chunk_lines))
        ctx_text = "\n---\n".join(ctx_parts)

        active_slice = msg_store.get_active_slice()
        answer_messages = list(active_slice)
        answer_messages.append({"role": "system", "content": f"【知识库】\n{ctx_text}"})
        answer_messages.append({"role": "user", "content": f"问: {q}\n论文:\n{conv.paper_titles_str()}\n请做结构化推荐。"})

        if on_token:
            resp_parts = []
            def collect(t, is_thinking: bool = False):
                resp_parts.append(t)
                on_token(t, is_thinking)
            await stream_call_llm(
                SYSTEM_PROMPT, answer_messages[1:],
                max_tokens=mt or config.LLM_MAX_TOKENS, api_key=self._api_key, model=self._model,
                base_url=self._base_url, on_token=collect,
                reuse_provider=True,
            )
            resp = "".join(resp_parts)
        else:
            resp, _ = sync_call_llm(
                SYSTEM_PROMPT, answer_messages[1:],
                max_tokens=mt or config.LLM_MAX_TOKENS, api_key=self._api_key, model=self._model,
                base_url=self._base_url, cancel_event=self._cancel_event,
                reuse_provider=True,
            )

        resp = strip_thinking_tags(resp) if resp else resp
        if not resp.strip():
            resp = "抱歉，无法生成论文推荐，请稍后重试。"

        ts = int(time.time() * 1000)
        msg_store.append_user(q)
        msg_store.append_assistant(resp)
        conv.messages.append({"role": "user", "content": q, "timestamp": ts})
        conv.messages.append({"role": "assistant", "content": resp, "timestamp": ts, "mode": self._current_mode, "model": self._model})
        await self._maybe_persist()
        return self._result(response=resp, refused=False, type="recommend", papers_count=len(papers_map))

    # ── 两轮检索辅助 ──

    @staticmethod
    def _extract_paper_ids_from_context(search_context: str) -> list[int]:
        """从 RAG 上下文（【来源: Paper#8, ...】格式）提取论文 ID 列表。"""
        ids = set()
        for m in re.findall(r"Paper#(\d+)", search_context):
            try:
                ids.add(int(m))
            except ValueError:
                pass
        return list(ids)[:10]  # 最多 10 篇

    async def _eval_search_gaps(
        self, question: str, search_context: str, search_terms: list[str],
    ) -> dict | None:
        """让 LLM 评估首轮检索结果的缺口。

        返回 None 表示结果充分；返回 dict 包含需要补搜的不清楚术语和缺失概念。
        仅本地模式执行。
        """
        if not is_local_mode(self._base_url):
            return None

        ctx_short = search_context[:1200] if len(search_context) > 1200 else search_context

        prompt = f"""分析以下检索结果能否充分回答用户问题。

用户问题: {question}
当前搜索词: {", ".join(search_terms) if search_terms else "无"}

检索结果:
{ctx_short}

判断标准:
- 结果中是否有你不理解的专业术语/缩写/专有名词？→ unclear_terms
- 回答用户问题是否还缺少关键概念或逻辑环节？→ missing_concepts
- 如果检索结果已经足够回答问题，返回 null

输出 JSON:
null  ← 结果充分
{{"unclear_terms": ["术语1", "术语2"], "missing_concepts": ["缺失概念1"]}}  ← 有缺口

仅输出 null 或 JSON，不输出其他内容。"""

        try:
            resp, _usage = await asyncio.to_thread(
                sync_call_llm,
                SYSTEM_PROMPT,
                [{"role": "user", "content": prompt}],
                max_tokens=120,
                api_key=self._api_key, model=self._model,
                base_url=self._base_url,
                thinking=False,
                cancel_event=self._cancel_event,
            )
            resp = resp.strip()
            if not resp or resp.lower() == "null":
                return None
            result = _extract_json(resp, "eval_gaps")
            if not isinstance(result, dict):
                return None
            # 确保字段存在且为列表
            unclear = result.get("unclear_terms", [])
            missing = result.get("missing_concepts", [])
            if not isinstance(unclear, list):
                unclear = []
            if not isinstance(missing, list):
                missing = []
            unclear = [str(t) for t in unclear if t]
            missing = [str(t) for t in missing if t]
            if unclear or missing:
                return {"unclear_terms": unclear, "missing_concepts": missing}
            return None
        except asyncio.CancelledError:
            raise
        except Exception:
            return None  # 评估失败不阻塞主流程

    # ── 会话关闭 ──

    async def close_conversation(self, conv_id: str) -> None:
        """关闭会话：生成总结 + 归档 + 持久化 + 释放资源"""
        if conv_id not in self._message_stores:
            return

        msg_store = self._message_stores[conv_id]
        compress_engine = self._get_compress_engine()
        archived_idx = self._get_archived_index()

        try:
            await compress_engine.close_session(msg_store, archived_idx, conv_id)
        except Exception:
            pass

        # 释放
        self._message_stores.pop(conv_id, None)
        self._l2_cache.pop(conv_id, None)
        clear_provider_cache()


# Singleton
engine = ZhibanV10()
