"""双视图消息管理 — conv.messages 完整视图 + get_active_slice() LLM 切片

完整视图 (conv.messages)：保存全量历史，只增不减，用于 UI 回溯。
活跃切片 (get_active_slice)：只取最后一个 compact_boundary 之后的内容，发给 LLM。

compact_boundary 标记本身不发给 LLM（normalize 时过滤）。
最早 boundary 之前的原始轮次可以物理删除（摘要已覆盖关键信息）。
"""

from __future__ import annotations

from .kv_cache_config import MAX_MESSAGES_BEFORE_CLEANUP

# 消息子类型常量
SUBTYPE_COMPACT_BOUNDARY = "compact_boundary"
SUBTYPE_SUMMARY = "summary"


class MessageStore:
    """双视图消息管理器。

    conv.messages 完整视图:
      [L0_SP,
       user_T1, asst_T1, ..., user_T10, asst_T10,
       compact_boundary_1,     ← 第一次压缩标记
       summary_message_1,      ← 第一次压缩摘要
       user_T11, asst_T11, ...]

    get_active_slice() 切片:
      [L0_SP,
       summary_message_2,      ← 最近的摘要
       user_T21, asst_T21, ...] ← boundary 之后的活跃轮次
    """

    def __init__(self, system_prompt: str = ""):
        self._messages: list[dict] = []
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})

    # ── 属性 ──

    @property
    def messages(self) -> list[dict]:
        """完整消息列表（含 boundary 标记和摘要）"""
        return self._messages

    @property
    def raw_count(self) -> int:
        return len(self._messages)

    @property
    def user_rounds(self) -> int:
        """用户提问轮次数"""
        return sum(1 for m in self._messages if m.get("role") == "user")

    # ── 追加 ──

    def append_user(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def append_assistant(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})

    def append_system(self, content: str, subtype: str = "") -> None:
        msg: dict = {"role": "system", "content": content}
        if subtype:
            msg["subtype"] = subtype
        self._messages.append(msg)

    # ── 压缩标记 ──

    def insert_boundary(self) -> None:
        """插入 compact_boundary 标记（仅客户端可见，不发 LLM）"""
        self._messages.append({
            "role": "system",
            "subtype": SUBTYPE_COMPACT_BOUNDARY,
            "content": "",
        })

    def insert_summary(self, summary_text: str) -> None:
        """插入压缩摘要消息"""
        self._messages.append({
            "role": "system",
            "subtype": SUBTYPE_SUMMARY,
            "content": summary_text,
        })

    # ── 切片 ──

    def get_active_slice(self) -> list[dict]:
        """取最后一个 compact_boundary 之后的消息（发给 LLM 的切片）。

        L0 (system prompt) 始终保留在切片头部。
        过滤掉 boundary 标记本身，保留摘要和后续轮次。
        """
        # L0 is always at index 0 (the system prompt)
        l0 = self._messages[0] if self._messages and self._messages[0].get("role") == "system" else None

        last_boundary_idx = -1
        for i in range(len(self._messages) - 1, -1, -1):
            if self._messages[i].get("subtype") == SUBTYPE_COMPACT_BOUNDARY:
                last_boundary_idx = i
                break

        if last_boundary_idx == -1:
            # 无压缩标记，返回全部（过滤可能的旧格式 boundary）
            result = [m for m in self._messages if m.get("subtype") != SUBTYPE_COMPACT_BOUNDARY]
            return result

        # 取 boundary 之后的消息，过滤掉 boundary 标记本身
        post_boundary = [
            m for m in self._messages[last_boundary_idx + 1:]
            if m.get("subtype") != SUBTYPE_COMPACT_BOUNDARY
        ]

        # L0 always prepended (summary messages also have role="system" but with subtype="summary")
        if l0 and l0.get("subtype") != SUBTYPE_SUMMARY:
            # Check if L0 is already the first item
            if not post_boundary or post_boundary[0] is not l0:
                return [l0] + post_boundary
        return post_boundary

    def get_full_history(self) -> list[dict]:
        """返回完整历史（用于 UI 展示，含 boundary 标记但不含其空内容）"""
        return list(self._messages)

    # ── 物理清理 ──

    def clean_old_rounds(self) -> int:
        """物理删除最早 compact_boundary 之前的原始轮次。

        触发条件: len(messages) > MAX_MESSAGES_BEFORE_CLEANUP (500)
        只清理第一个 boundary 之前的内容，保留 boundary 和之后的所有消息。

        Returns: 删除的消息条数
        """
        if len(self._messages) <= MAX_MESSAGES_BEFORE_CLEANUP:
            return 0

        # 找到第一个 compact_boundary
        first_boundary_idx = -1
        for i, m in enumerate(self._messages):
            if m.get("subtype") == SUBTYPE_COMPACT_BOUNDARY:
                first_boundary_idx = i
                break

        if first_boundary_idx <= 1:
            return 0  # 没有 boundary 或 boundary 在开头，无需清理

        # 删除 boundary 之前的原始轮次（保留 L0 system prompt）
        deleted = first_boundary_idx - 1  # -1 保留 L0
        if deleted > 0:
            # 保留 messages[0] (L0_SP) 和 boundary 之后的所有消息
            self._messages = (
                self._messages[:1] +  # L0
                self._messages[first_boundary_idx:]  # boundary + after
            )

        return deleted

    def get_boundary_count(self) -> int:
        """返回到目前为止插入的 compact_boundary 数量"""
        return sum(
            1 for m in self._messages
            if m.get("subtype") == SUBTYPE_COMPACT_BOUNDARY
        )

    # ── Token 估算 ──

    def estimate_tokens(self, text: str | None = None) -> int:
        """粗略估算 token 数（中文 ~1.5 char/tok, 英文 ~4 char/tok）"""
        if text is None:
            text = " ".join(
                m.get("content", "") for m in self.get_active_slice()
            )
        if not text:
            return 0
        # 混合估算: 中文字符 ~1.5 char/tok, 英文 ~4 char/tok
        cn_chars = sum(1 for c in text if '一' <= c <= '鿿')
        en_chars = len(text) - cn_chars
        return int(cn_chars / 1.5 + en_chars / 4)

    def active_slice_tokens(self) -> int:
        return self.estimate_tokens()

    # ── 会话关闭 ──

    def get_last_n_rounds(self, n: int = 5) -> list[dict]:
        """获取最后 N 轮用户-助手对话（用于关闭时生成总结）"""
        slice_msgs = self.get_active_slice()
        rounds: list[dict] = []
        for m in slice_msgs:
            if m.get("role") in ("user", "assistant"):
                rounds.append(m)
        return rounds[-(n * 2):]  # N 轮 = 2N 条消息

    def get_all_user_rounds(self) -> list[dict]:
        """获取所有用户和助手的原始对话轮次（用于 embedding 归档）"""
        return [
            m for m in self._messages
            if m.get("role") in ("user", "assistant")
            and m.get("subtype") != SUBTYPE_COMPACT_BOUNDARY
        ]
