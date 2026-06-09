"""后台压缩引擎 — 独立 context slot，不污染主会话 KV cache。

触发条件:
  1. active_slice tokens > n_ctx - COMPRESS_BUFFER (剩余不足 2000 tok)
  2. active_slice tokens > n_ctx * COMPRESS_TRIGGER_RATIO (占用超过 80%)

压缩流程:
  1. 取最早的 3-5 轮活跃对话
  2. 独立 LLM 调用生成结构化摘要（使用专用轻量 SP）
  3. 向量化原始对话文本（jina embeddings）
  4. 存入 archived_index（关键词倒排 + embedding）
  5. 追加 compact_boundary + summary 到 messages
  6. 物理清理（如超 500 条）
  7. 通知前端

API 模式 (DeepSeek/Ollama HTTP):
  不碰服务端 KV cache（无法控制）
  只靠 get_active_slice() 切片，下次请求自然发送更短内容

本地模式 (llama-cpp-python 进程内):
  需主动清除被压缩掉的旧 KV entry: kv_cache_seq_rm()
  可选：prefill 摘要文本到释放的空位
"""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Any

from .kv_cache_config import (
    COMPRESS_BUFFER,
    COMPRESS_TRIGGER_RATIO,
    WARN_BUFFER,
)

logger = logging.getLogger("zhiban.compress")

# 压缩摘要专用 System Prompt（轻量，不包含主 SP 的业务逻辑）
COMPRESS_SYSTEM_PROMPT = """你是一个对话摘要工具。将以下对话提炼为结构化摘要。

输出格式:
【历史摘要】
- 研究方向: <研究的核心主题>
- 关键结论: <对话中得出的重要结论>
- 引用论文: <提到的论文编号>
- 关键词: <5-8个核心技术关键词>
"""

# 每次压缩的轮次数
COMPRESS_ROUNDS_MIN = 3
COMPRESS_ROUNDS_MAX = 5

# 唤醒相似度阈值
WAKE_COSINE_THRESHOLD = 0.75  # 摘要文本与提问的相似度
WAKE_ARCHIVE_THRESHOLD = 0.55  # 原文向量与提问向量的相似度


class CompressEngine:
    """后台压缩引擎。

    使用独立的 provider 实例（或独立 context slot），
    主会话的 KV cache 不受影响。
    """

    def __init__(
        self,
        sync_llm: Callable[..., tuple[str, dict]] | None = None,
        embed_fn: Callable[[str], list[float]] | None = None,
        n_ctx: int = 32768,
    ):
        self._sync_llm = sync_llm
        self._embed_fn = embed_fn
        self._n_ctx = n_ctx

    # ── 触发判断 ──

    def should_compress(self, active_slice: list[dict], token_counter: Callable[[str], int] | None = None) -> bool:
        """判断当前活跃切片是否需要压缩。

        Args:
          active_slice: get_active_slice() 的结果
          token_counter: 精确 token 计数函数，None 时用 heuristic 估算

        Returns: 是否需要压缩
        """
        if not active_slice:
            return False

        text = " ".join(m.get("content", "") for m in active_slice)
        if token_counter:
            current_tokens = token_counter(text)
        else:
            cn_chars = sum(1 for c in text if '一' <= c <= '鿿')
            en_chars = len(text) - cn_chars
            current_tokens = int(cn_chars / 1.5 + en_chars / 4)

        remaining = self._n_ctx - current_tokens
        usage_ratio = current_tokens / self._n_ctx if self._n_ctx > 0 else 0

        if remaining < COMPRESS_BUFFER:
            logger.info("Compress triggered: remaining=%d < buffer=%d", remaining, COMPRESS_BUFFER)
            return True
        if usage_ratio > COMPRESS_TRIGGER_RATIO:
            logger.info("Compress triggered: usage=%.1f%% > threshold=%.1f%%",
                        usage_ratio * 100, COMPRESS_TRIGGER_RATIO * 100)
            return True

        return False

    # ── 压缩执行 ──

    async def compress(
        self,
        active_slice: list[dict],
        message_store,  # MessageStore
        archived_index,  # ArchivedIndex
        on_progress: Callable[[int, str], Any] | None = None,
        *,
        api_key: str = "",
        model: str = "",
        base_url: str = "",
    ) -> dict:
        """执行压缩流程。

        Args:
          active_slice: 当前活跃切片
          message_store: 消息管理器
          archived_index: 压缩历史索引
          on_progress: 进度回调 (progress_pct, verb)
          api_key/model/base_url: 透传给 LLM（跟随用户当前配置）

        Returns: {summary, keywords, archived_count, rounds_compressed}
        """
        # Step 1: 取最早的 3-5 轮对话（不含 system prompt 和 L2 论文内容）
        rounds = self._extract_compressible_rounds(active_slice)

        if len(rounds) < COMPRESS_ROUNDS_MIN * 2:  # 每轮 = user + assistant
            logger.info("Compress skipped: only %d messages, need >= %d",
                        len(rounds), COMPRESS_ROUNDS_MIN * 2)
            return {"summary": "", "keywords": [], "archived_count": 0, "rounds_compressed": 0}

        if on_progress:
            on_progress(10, "正在分析对话...")

        # Step 2: 独立 LLM 调用生成结构化摘要
        summary = ""
        keywords: list[str] = []
        if self._sync_llm:
            rounds_text = "\n".join(
                f"{'用户' if m['role'] == 'user' else '助手'}: {m['content'][:300]}"
                for m in rounds
            )
            compress_prompt = (
                f"请将以下对话提炼为结构化摘要：\n\n{rounds_text}"
            )

            try:
                summary_text, _ = await asyncio.to_thread(
                    self._sync_llm,
                    COMPRESS_SYSTEM_PROMPT,
                    [{"role": "user", "content": compress_prompt}],
                    512,  # max_tokens — 足够写完研究方向+结论+论文+关键词，不过度占用上下文
                    api_key or "",  # 跟随用户 API Key
                    model or "",    # 跟随用户模型
                    base_url or "", # 跟随用户 base URL
                )
                summary = summary_text.strip()
                # 提取关键词
                from .archived_index import _extract_keywords
                keywords = _extract_keywords(summary, max_kw=8)
            except Exception as e:
                logger.warning("Compress LLM call failed: %s", e)
                # 降级：不生成摘要，仍然归档原始对话
                from .archived_index import _extract_keywords
                rounds_text = " ".join(m.get("content", "") for m in rounds)
                keywords = _extract_keywords(rounds_text, max_kw=8)

        if on_progress:
            on_progress(40, "正在向量化对话...")

        # Step 3: 向量化原始对话 + 归档
        rounds_text = " ".join(m.get("content", "") for m in rounds)
        embedding = None
        if self._embed_fn:
            try:
                embedding = await asyncio.to_thread(self._embed_fn, rounds_text)
            except Exception as e:
                logger.warning("Embedding failed: %s", e)

        archive_id = archived_index.add(
            text=rounds_text,
            embedding=embedding,
            keywords=keywords,
        )

        if on_progress:
            on_progress(70, "正在更新消息历史...")

        # Step 4: 追加标记到 messages
        message_store.insert_boundary()
        if summary:
            message_store.insert_summary(summary)

        # Step 5: 物理清理
        if on_progress:
            on_progress(85, "正在清理旧记录...")

        deleted = message_store.clean_old_rounds()
        if deleted:
            logger.info("Physical cleanup: removed %d messages before earliest boundary", deleted)

        if on_progress:
            on_progress(100, "压缩完成")

        return {
            "summary": summary,
            "keywords": keywords,
            "archived_count": archived_index.entry_count,
            "rounds_compressed": len(rounds) // 2,
            "messages_cleaned": deleted,
            "archive_id": archive_id,
        }

    # ── 提取可压缩轮次 ──

    def _extract_compressible_rounds(self, active_slice: list[dict]) -> list[dict]:
        """从活跃切片中提取最早的 3-5 轮（不含系统提示、摘要、boundary 标记）。

        只取 role 为 user/assistant 的消息。
        """
        rounds: list[dict] = []
        for m in active_slice:
            role = m.get("role", "")
            subtype = m.get("subtype", "")
            if role in ("user", "assistant") and not subtype:
                rounds.append(m)

        if len(rounds) < COMPRESS_ROUNDS_MIN * 2:
            return []

        # 取最早的 3-5 轮 (每轮 2 条: user + assistant)
        max_messages = min(COMPRESS_ROUNDS_MAX * 2, len(rounds) - 2)  # 至少保留最后 1 轮
        return rounds[:max_messages]

    # ── 唤醒检查 ──

    def check_wake_needed(
        self,
        question: str,
        active_slice: list[dict],
        archived_index,  # ArchivedIndex
        embed_fn: Callable[[str], list[float]] | None = None,
    ) -> list[dict]:
        """检查是否需要从压缩历史中唤醒对话记录。

        Step 1: 检查当前 L1 摘要是否够用（cosine similarity > 0.75）
        Step 2: 不够时查 archived_index

        Returns: [{text, archive_id, score}, ...] 需要注入 L3 的唤醒记录
        """
        # 提取 L1 中的摘要文本
        summary_text = ""
        for m in active_slice:
            if m.get("subtype") == "summary":
                summary_text += m.get("content", "") + " "

        # Step 1: 检查摘要是否够用
        if summary_text.strip() and embed_fn:
            try:
                q_vec = embed_fn(question)
                s_vec = embed_fn(summary_text.strip())
                sim = self._cosine_sim(q_vec, s_vec)
                if sim > WAKE_COSINE_THRESHOLD:
                    return []  # 摘要已覆盖，不需要唤醒
            except Exception:
                pass

        # Step 2: 检索存档
        q_vec = None
        if embed_fn:
            try:
                q_vec = embed_fn(question)
            except Exception:
                pass

        results = archived_index.search(question, q_vec)
        return results

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ── 会话关闭 ──

    async def close_session(
        self,
        message_store,  # MessageStore
        archived_index,  # ArchivedIndex
        conv_id: str,
    ) -> str:
        """会话关闭流程。

        1. 取出最后 N 轮 → 生成结构化总结
        2. 向量化原始对话文本
        3. 提取关键词 → 存入 archived_index
        4. 持久化到磁盘
        5. 释放资源

        Returns: 关闭时的摘要文本
        """
        last_rounds = message_store.get_last_n_rounds(5)
        if not last_rounds:
            return ""

        rounds_text = " ".join(m.get("content", "") for m in last_rounds)
        summary_text = ""

        if self._sync_llm:
            try:
                summary_text, _ = await asyncio.to_thread(
                    self._sync_llm,
                    COMPRESS_SYSTEM_PROMPT,
                    [{"role": "user", "content": f"请总结这段对话：\n{rounds_text}"}],
                    256,
                )
                summary_text = summary_text.strip()
            except Exception:
                pass

        embedding = None
        if self._embed_fn:
            try:
                embedding = await asyncio.to_thread(self._embed_fn, rounds_text)
            except Exception:
                pass

        from .archived_index import _extract_keywords
        keywords = _extract_keywords(rounds_text)

        archived_index.add(
            text=rounds_text,
            embedding=embedding,
            keywords=keywords,
        )

        archived_index.persist(conv_id)

        return summary_text or rounds_text[:200]
