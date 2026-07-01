"""ChatState 状态机 + pending queue + 并发控制

状态: IDLE → THINKING → COMPRESSING → IDLE
              ↘                ↙
              → → → IDLE ← ← ←

IDLE:        可接收新消息
THINKING:    LLM 正在生成回答，前端显示动画，忽略重复提交
COMPRESSING: 后台压缩中，前端可输入但消息入 pending queue
             压缩完成后自动出队处理

WebSocket 通知格式:
  {"type":"status","state":"thinking","verb":"正在生成回答..."}
  {"type":"status","state":"compressing","verb":"正在优化上下文...","progress":60}
  {"type":"status","state":"idle"}
"""

from __future__ import annotations

import asyncio
import enum
import time
from typing import Any, Callable, Awaitable


class ChatState(enum.Enum):
    IDLE = "idle"
    THINKING = "thinking"
    COMPRESSING = "compressing"


# 状态对应的动词描述
STATE_VERBS: dict[ChatState, str] = {
    ChatState.IDLE: "",
    ChatState.THINKING: "正在生成回答...",
    ChatState.COMPRESSING: "正在优化上下文...",
}


class SessionState:
    """会话状态管理器。

    职责:
      - 维护当前 ChatState（IDLE / THINKING / COMPRESSING）
      - COMPRESSING 期间暂存用户消息到 pending queue
      - 压缩完成后自动处理排队消息
      - WebSocket 状态通知回调
    """

    def __init__(self):
        self._state: ChatState = ChatState.IDLE
        self._pending: asyncio.Queue = asyncio.Queue()  # stores async callables
        self._on_state_change: Callable[[ChatState, dict | None], Awaitable[None]] | None = None
        self._compress_progress: int = 0

    # ── 状态读写 ──

    @property
    def state(self) -> ChatState:
        return self._state

    @property
    def is_idle(self) -> bool:
        return self._state == ChatState.IDLE

    @property
    def is_thinking(self) -> bool:
        return self._state == ChatState.THINKING

    @property
    def is_compressing(self) -> bool:
        return self._state == ChatState.COMPRESSING

    def set_state(self, new_state: ChatState) -> None:
        """切换状态（内部使用，不触发回调）。需要通知前端时用 transition()"""
        self._state = new_state

    async def transition(self, new_state: ChatState, extra: dict | None = None) -> None:
        """切换状态并通知前端。

        extra 可包含: verb (覆盖默认动词), progress (压缩进度 0-100)
        """
        old_state = self._state
        self._state = new_state

        if self._on_state_change:
            payload: dict[str, Any] = {
                "state": new_state.value,
                "verb": extra.get("verb", STATE_VERBS.get(new_state, "")) if extra else STATE_VERBS.get(new_state, ""),
                "prev_state": old_state.value,
                "timestamp": int(time.time() * 1000),
            }
            if extra:
                payload.update(extra)
            result = self._on_state_change(new_state, payload)
            if result is not None:
                await result

    def set_on_state_change(self, callback: Callable[[ChatState, dict | None], Awaitable[None]]) -> None:
        """注册状态变更回调（用于发送 WebSocket 通知）"""
        self._on_state_change = callback

    # ── pending queue ──

    async def enqueue(self, processor: Callable[[], Awaitable[None]]) -> None:
        """将消息处理回调放入 pending queue（仅在 COMPRESSING 状态时使用）。

        传入一个 async callable，压缩完成后会被自动调用。
        用法: await state.enqueue(lambda: handle_user_query(ws, msg, conv_id))
        """
        await self._pending.put(processor)

    def pending_size(self) -> int:
        return self._pending.qsize()

    async def process_pending(self) -> list[Callable[[], Awaitable[None]]]:
        """取出所有排队回调（非阻塞）"""
        items: list[Callable[[], Awaitable[None]]] = []
        while not self._pending.empty():
            try:
                items.append(self._pending.get_nowait())
            except asyncio.QueueEmpty:
                break
        return items

    # ── 消息入口 ──

    def can_accept(self) -> tuple[bool, str]:
        """检查当前状态是否可接收新消息。

        Returns: (可接收, 原因)
        """
        if self._state == ChatState.IDLE:
            return True, ""
        if self._state == ChatState.THINKING:
            return False, "正在生成回答中，请稍候"
        if self._state == ChatState.COMPRESSING:
            return True, "queued"  # 可接收但入队
        return False, "未知状态"

    # ── 压缩进度 ──

    @property
    def compress_progress(self) -> int:
        return self._compress_progress

    def set_compress_progress(self, progress: int) -> None:
        self._compress_progress = max(0, min(100, progress))

    # ── 健康状态 ──

    def health_payload(self) -> dict:
        return {
            "state": self._state.value,
            "pending_count": self.pending_size(),
            "compress_progress": self._compress_progress,
        }
