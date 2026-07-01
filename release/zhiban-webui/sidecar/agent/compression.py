"""Agent 上下文压缩 — KV Cache 感知的压缩策略。

三种压缩模式（仿 cc-haha）:
  1. Auto Compact: token 超限时用 LLM 总结早期轮次
  2. Snip Compact: 直接剪除超出边界的早期消息（不依赖 LLM，快速）
  3. Micro Compact: 压缩单个工具结果为摘要

KV Cache 友好设计:
  - 只在压缩边界之后追加新消息，保持前缀不变
  - 用紧凑摘要替换详细对话 → 缩短前缀但不改变 L0
  - Token 估算基于字符统计（中文 ~1.5 char/tok, 英文 ~4 char/tok）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

try:
    from ..llm.kv_cache_config import COMPRESS_BUFFER, COMPRESS_TRIGGER_RATIO
except ImportError:
    COMPRESS_BUFFER = 2000
    COMPRESS_TRIGGER_RATIO = 0.8

logger = logging.getLogger("zhiban.agent.compression")


# ── Token 估算 ──

def estimate_tokens(text: str) -> int:
    """混合中英文 token 估算。"""
    if not text:
        return 0
    cn_chars = sum(1 for c in text if '一' <= c <= '鿿'
                   or '㐀' <= c <= '䶿')
    en_chars = max(0, len(text) - cn_chars)
    return int(cn_chars / 1.5 + en_chars / 4.0)


def estimate_messages_tokens(messages: list[dict]) -> int:
    """估算消息列表总 token 数。"""
    return sum(
        estimate_tokens(m.get("content", ""))
        for m in messages
    )


# ── 压缩配置 ──

@dataclass
class CompressionConfig:
    """压缩阈值配置"""
    n_ctx: int = 32768          # 模型上下文窗口大小
    buffer: int = COMPRESS_BUFFER  # 剩余不足时触发压缩
    trigger_ratio: float = COMPRESS_TRIGGER_RATIO  # 占用超过此比例触发
    keep_recent_rounds: int = 3  # 保留最近 N 轮
    max_summary_tokens: int = 512  # 摘要最大 token 数

    @property
    def auto_threshold(self) -> int:
        """自动压缩触发阈值"""
        return int(self.n_ctx * self.trigger_ratio)

    @property
    def block_threshold(self) -> int:
        """硬拦截阈值（超过此值拒绝新消息）"""
        return self.n_ctx - self.buffer


# ── Snip 压缩 ──

def snip_compact(
    messages: list[dict],
    config: CompressionConfig,
    system_prompt_idx: int = 0,
) -> tuple[list[dict], int]:
    """剪除早期消息，保留 system prompt + 最近 N 轮。

    不依赖 LLM，纯字符串操作。适合快速清理。
    返回 (压缩后的消息列表, 剪除的消息数)

    消息格式: [L0_system, user1, asst1, user2, asst2, ..., userN, asstN]
    """
    if len(messages) <= config.keep_recent_rounds * 2 + 2:
        return messages, 0

    # 保留: system prompt + 最近 N 轮
    keep_count = config.keep_recent_rounds * 2  # N user + N assistant
    if system_prompt_idx >= 0 and messages and messages[0].get("role") == "system":
        l0 = messages[0:1]
        rest = messages[1:]
    else:
        l0 = []
        rest = messages

    removed = len(rest) - keep_count
    if removed <= 0:
        return messages, 0

    kept = rest[-keep_count:]
    result = l0 + kept

    logger.info("Snip compact: removed %d messages, keeping %d", removed, len(kept))
    return result, removed


# ── 工具结果压缩 ──

def micro_compact_tool_result(content: str, max_chars: int = 500) -> str:
    """压缩单个工具结果：过长的结果只保留首尾。

    这避免工具结果占用过多 KV cache 空间。
    """
    if len(content) <= max_chars:
        return content

    half = max_chars // 2
    lines = content.split("\n")
    if len(lines) <= 5:
        # 行数少但很长：截断每行
        return content[:half] + "\n... (已截断) ...\n" + content[-half:]

    # 保留前 3 行和后 2 行
    head = "\n".join(lines[:3])[:half]
    tail = "\n".join(lines[-2:])[-half:]
    return head + f"\n... (共 {len(lines)} 行，已截断) ...\n" + tail


# ── 摘要压缩（需要 LLM） ──

async def auto_compact_summarize(
    messages: list[dict],
    summarize_fn,
    max_summary_tokens: int = 512,
    paper_mode: bool = False,
) -> str:
    """调用 LLM 生成对话摘要。

    summarize_fn: async (prompt: str) -> str
    paper_mode: 论文伴读模式，提示侧重保留论文信息
    """
    # 取待压缩的对话文本
    dialog_parts = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content", "")[:300]
        if role == "system":
            continue
        dialog_parts.append(f"[{role}]: {content}")

    dialog_text = "\n".join(dialog_parts[-20:])
    if not dialog_text.strip():
        return ""

    if paper_mode:
        prompt = (
            "总结以下论文讨论的要点。必须保留:\n"
            "1. 用户问过的所有问题\n"
            "2. 提到的论文名称、作者、关键发现和数据\n"
            "3. AI 给出的核心结论\n"
            "忽略过渡语和重复内容。只输出摘要，不加前缀。\n\n"
            f"{dialog_text}"
        )
    else:
        prompt = (
            "请用 2-3 句话总结以下对话的核心内容和关键结论。"
            "只输出摘要，不要加任何前缀或解释。\n\n"
            f"{dialog_text}"
        )

    try:
        summary = await summarize_fn(prompt)
        if len(summary) > max_summary_tokens * 4:
            summary = summary[:max_summary_tokens * 4]
        return summary.strip()
    except Exception as e:
        logger.warning("Auto compact summarize failed: %s", e)
        return ""


# ── 上下文守卫 ──

def check_context_overflow(
    messages: list[dict],
    config: CompressionConfig,
) -> tuple[bool, str]:
    """检查上下文是否超限，返回 (需要压缩, 原因)。"""
    total = estimate_messages_tokens(messages)

    if total >= config.block_threshold:
        return True, f"critical: {total} tokens >= {config.block_threshold} (hard limit)"

    if total >= config.auto_threshold:
        return True, f"warning: {total} tokens >= {config.auto_threshold} (auto compact)"

    return False, "ok"
