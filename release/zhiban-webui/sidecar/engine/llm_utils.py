"""LLM 调用工具 — 缓存 Provider 实例以复用 KV cache。

同一轮 Call 1 (classify) → Call 2 (answer) 之间复用 provider 实例，
使得 L0+L1 的 KV cache 跨调用命中。

压缩引擎的 LLM 调用也走这里提供的底层 sync_call_llm()。
llm_utils.py 保持底层工具角色，不包含业务逻辑。

Local 模式 (base_url="__local__") 时路由到 LocalChatEngine。
"""

import asyncio
import re
import time
from typing import Callable

from .. import config
from ..llm.deepseek_proxy import build_provider
from ..llm.providers.base import LLMProvider

# ── Thinking 标签过滤 ──

# Qwen 系列模型的 <think> / <opti-q-think> XML 块
_THINK_TAG_PATTERN = re.compile(
    r'<\s*(?:think|opti-q-think)[^>]*>.*?<\s*/\s*(?:think|opti-q-think)\s*>\s*',
    re.DOTALL | re.IGNORECASE,
)
# 未闭合的 think 块：从 <think> 到文本末尾（流式截断产物）
_THINK_TRUNCATED = re.compile(
    r'<\s*(?:think|opti-q-think)[^>]*>.*',
    re.DOTALL | re.IGNORECASE,
)
# 孤立的开/闭标签
_THINK_OPEN_TAG = re.compile(r'<\s*(?:think|opti-q-think)[^>]*>', re.IGNORECASE)
_THINK_CLOSE_TAG = re.compile(r'<\s*/\s*(?:think|opti-q-think)\s*>', re.IGNORECASE)
# OptiQ / 部分模型将 thinking 输出为纯文本 "Thinking Process:" 段落
# 匹配从 "Thinking Process:" 到第一个非结构化行之前的所有内容
# 结构化行 = 以数字/星号/井号/短横/空格开头 或 空行
_THINKING_PROCESS_PATTERN = re.compile(
    r'(?:^|\n)Thinking Process:[\s\S]*?(?=\n(?![\d\*\-#\s]|\*\*)|$)',
    re.IGNORECASE,
)


def strip_thinking_tags(text: str) -> str:
    """移除模型输出中的 thinking 内容。

    支持两种格式：
      1. XML 标签: <think>...</think> 或 <opti-q-think>...</opti-q-think>
      2. 纯文本: Thinking Process: ... （OptiQ 模型在 thinking 关闭时仍可能输出）
    只保留实际回答部分。
    """
    if not text:
        return text
    text = _THINK_TAG_PATTERN.sub('', text)
    text = _THINK_TRUNCATED.sub('', text)
    text = _THINKING_PROCESS_PATTERN.sub('', text)
    return text.strip()


def _extract_answer_from_reasoning(text: str) -> str | None:
    """从 Qwopus 模型的 reasoning 文本中提取纯答案部分。

    段落级方法：分类每个段落是"思考"还是"内容"，只保留内容段落。
    比边界匹配更稳健，不依赖具体的文本模式。
    """
    if not text or len(text) < 50:
        return None

    # 思考段落特征（模型自我对话/规划/指令复述）
    _thinking_line_patterns = [
        r'^用户(?:要求|问的是|希望|询问|让我)',
        r'^(?:我(?:需要|应该|可以|来|将)|让我|首先[，,])',
        r'^(?:需要|应该|可以)(?:.{0,5})(?:提取|总结|梳理|搜索|检索|回答)',
        r'^(?:根据|按照)(?:.{0,5})(?:规则|格式|要求)',
        r'^[\d一二三四五六七八九十]+[\.、]\s*(?:摘要|引言|方法|结果|讨论|结论|背景)',
        r'^💭',
        r'^(?:摘要|引言|结果与讨论|结论|方法)(?:要点|部分)?[：:]?',
        r'^(?:现在|好的|让我|首先).{0,10}(?:组织|开始|提取|总结)',
        r'^".*?"\s*$',  # 纯引号行
    ]
    _thinking_pattern = re.compile('|'.join(_thinking_line_patterns))

    # 段落正文标题（真正的答案内容，必须有"："且后面跟实质性描述）
    _content_header_pattern = re.compile(
        r'^(?:摘要与引言|引言与背景|结果与讨论|实验部分|结论与展望|'
        r'结论|材料与方法|研究背景|核心发现|主要贡献)[：:]\s*\S'
    )

    lines = text.split('\n')
    # 找第一个内容段落：必须是显式的正文标题
    content_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if _content_header_pattern.match(stripped):
            content_start = i
            break

    # 找最后一个内容段落的位置（从后往前扫，找到第一个非思考行作为结尾）
    content_end = len(lines)
    for i in range(len(lines) - 1, content_start, -1):
        stripped = lines[i].strip()
        if not stripped:
            continue
        if not (_thinking_pattern.match(stripped) or stripped.startswith('💭')):
            content_end = i + 1  # 保留这一行
            break
        content_end = i

    if content_start > 0 or content_end < len(lines):
        result_lines = lines[content_start:content_end]
        result = '\n'.join(result_lines).strip()
        if result and len(result) > 20:
            return result

    return None


# 检测 thinking 开始的标记：XML <think> 标签 或 纯文本 "Thinking Process:"
def _has_thinking_start(text: str) -> bool:
    return bool(_THINK_OPEN_TAG.search(text) or "Thinking Process:" in text)


def _has_thinking_end(text: str) -> bool:
    """检测 thinking 块是否结束。XML 看 </think> 标签；纯文本看是否已出现
    非结构化的响应行（以中文或普通英文开头，不以数字/星号/井号/空格开头）。"""
    if _THINK_CLOSE_TAG.search(text):
        return True
    # 纯文本 "Thinking Process:" 格式：检查是否有非结构化的响应内容
    if "Thinking Process:" in text:
        clean = _THINKING_PROCESS_PATTERN.sub('', text)
        if clean.strip():
            return True
    return False


class ThinkingStreamFilter:
    """流式 thinking 过滤器。始终启用，支持两种格式：
    1. XML: <think>...</think> 或 <opti-q-think>...</opti-q-think>
    2. 纯文本: Thinking Process: ...（OptiQ 模型）

    检测到 thinking 后抑制输出。自然结束时剥离 thinking 只输出 clean text。
    硬上限：缓冲超过 MAX_CHARS 仍未结束 → 强制放行，保留原文（小模型可能
    全部输出都是 thinking，剥离后反而为空）。
    """

    _NO_THINK_THRESHOLD_TOKENS = 80
    _MAX_THINKING_CHARS = 2000

    def __init__(self):
        self._buf = ""
        self._clean_cursor = 0
        self._think_ended = False
        self._think_detected = False
        self._forced_end = False  # 硬上限触发，不剥离原文
        self._token_count = 0

    def feed(self, token: str) -> str | None:
        self._token_count += 1
        self._buf += token

        if self._think_ended:
            if self._forced_end:
                # 强制结束：原样输出，不剥离
                if len(self._buf) > self._clean_cursor:
                    emit = self._buf[self._clean_cursor:]
                    self._clean_cursor = len(self._buf)
                    return emit
                return None
            clean = strip_thinking_tags(self._buf)
            if len(clean) > self._clean_cursor:
                emit = clean[self._clean_cursor:]
                self._clean_cursor = len(clean)
                return emit
            return None

        if not self._think_detected:
            if _has_thinking_start(self._buf):
                self._think_detected = True
                return None
            elif self._token_count >= self._NO_THINK_THRESHOLD_TOKENS:
                self._think_ended = True
                self._clean_cursor = len(self._buf)
                return self._buf
            return None

        # 自然结束 或 硬上限强制结束
        if _has_thinking_end(self._buf):
            self._think_ended = True
        elif len(self._buf) > self._MAX_THINKING_CHARS:
            self._think_ended = True
            self._forced_end = True
        return None

    def finalize(self) -> str | None:
        if not self._think_detected:
            if self._think_ended:
                return None  # fallback 已输出全部，不重复
            return self._buf if self._buf else None
        if self._forced_end:
            if len(self._buf) > self._clean_cursor:
                return self._buf[self._clean_cursor:]
            return None
        clean = strip_thinking_tags(self._buf)
        if len(clean) > self._clean_cursor:
            emit = clean[self._clean_cursor:]
            self._clean_cursor = len(clean)
            return emit
        return None

    @property
    def full_text(self) -> str:
        return self._buf


class ThinkTagStreamSplitter:
    """流式 <think> 标签解析器。

    在 API 模式的 token 流中检测 <think>/<opti-q-think> 标签，
    即使标签被 tokenizer 拆成多个 fragment 也能正确识别。
    对 <think> 标签内的 token 标记 is_thinking=True，标签外的标记 is_thinking=False。

    用法:
        splitter = ThinkTagStreamSplitter()
        for chunk in stream:
            if chunk["type"] == "token":
                for is_thinking, text in splitter.feed(chunk["token"]):
                    on_token(text, is_thinking)
        for is_thinking, text in splitter.flush():
            on_token(text, is_thinking)
    """

    _OPEN_TAG_RE = re.compile(r'<(think|opti-q-think)>', re.IGNORECASE)
    _CLOSE_TAG_RE = re.compile(r'</(think|opti-q-think)>', re.IGNORECASE)
    _NO_TAG_THRESHOLD = 80
    _MAX_TAG_LEN = 20

    def __init__(self):
        self._buf = ""
        self._cursor = 0
        self._in_think = False
        self._tag_detected = False
        self._token_count = 0

    def feed(self, token: str) -> list[tuple[bool, str]]:
        """喂入一个 token，返回 [(is_thinking, text), ...] 列表。"""
        self._token_count += 1
        self._buf += token
        return self._emit_available()

    def flush(self) -> list[tuple[bool, str]]:
        """清空剩余缓冲区。"""
        results = []
        if self._cursor < len(self._buf):
            remaining = self._buf[self._cursor:]
            self._cursor = len(self._buf)
            if remaining:
                results.append((self._in_think, remaining))
        return results

    def _emit_available(self) -> list[tuple[bool, str]]:
        """从缓冲区中尽可能多地 emit 已确定分类的文本。"""
        results = []

        while self._cursor < len(self._buf):
            remaining = self._buf[self._cursor:]

            if not self._tag_detected:
                m_open = self._OPEN_TAG_RE.search(remaining)
                if m_open:
                    self._tag_detected = True
                    if m_open.start() > 0:
                        results.append((False, remaining[:m_open.start()]))
                    self._cursor += m_open.end()
                    self._in_think = True
                    continue

                safe_end = self._safe_end(remaining)
                if safe_end > 0:
                    results.append((False, remaining[:safe_end]))
                    self._cursor += safe_end

                if self._token_count >= self._NO_TAG_THRESHOLD:
                    self._tag_detected = True
                    tail = self._buf[self._cursor:]
                    if tail:
                        results.append((False, tail))
                        self._cursor = len(self._buf)
                break

            elif self._in_think:
                m_close = self._CLOSE_TAG_RE.search(remaining)
                if m_close:
                    if m_close.start() > 0:
                        results.append((True, remaining[:m_close.start()]))
                    self._cursor += m_close.end()
                    self._in_think = False
                    continue

                safe_end = self._safe_end(remaining)
                if safe_end > 0:
                    results.append((True, remaining[:safe_end]))
                    self._cursor += safe_end
                break

            else:
                m_open = self._OPEN_TAG_RE.search(remaining)
                if m_open:
                    if m_open.start() > 0:
                        results.append((False, remaining[:m_open.start()]))
                    self._cursor += m_open.end()
                    self._in_think = True
                    continue

                results.append((False, remaining))
                self._cursor = len(self._buf)
                break

        return results

    def _safe_end(self, text: str) -> int:
        """返回可以安全 emit 的字符数，保留末尾可能的部分标签。

        例如 text="前面内容</thi" → 返回 4（只 emit "前面内容"），
        保留 "</thi" 等下一个 token 来之后确认是否形成完整标签。
        """
        last_lt = text.rfind('<')
        if last_lt < 0:
            return len(text)

        suffix = text[last_lt:last_lt + self._MAX_TAG_LEN]
        known_tags = ['<think>', '<opti-q-think>', '</think>', '</opti-q-think>']
        for tag in known_tags:
            if tag.startswith(suffix) and len(suffix) < len(tag):
                return last_lt

        return len(text)


# ── 重试 ──

_RETRYABLE_PATTERNS = [
    "connection", "timeout", "timed out", "reset by peer",
    "broken pipe", "no route to host", "service unavailable",
    "try again", "too many requests", "rate limit",
    "internal server error", "bad gateway", "gateway timeout",
]


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(p in msg for p in _RETRYABLE_PATTERNS)


# ── Provider 实例缓存（API 模式用，同一轮内复用）──
_provider_cache: dict[str, LLMProvider] = {}

# ── 本地引擎单例 ──
_local_engine = None  # LocalChatEngine | None
_local_engine_path: str = ""  # 当前加载的模型路径
_local_engine_loading = False  # 是否正在加载中


def _cache_key(api_key: str, base_url: str) -> str:
    return f"{api_key}@{base_url}"


def get_or_create_provider(
    api_key: str = "",
    base_url: str = "",
    max_tokens: int = 4096,
) -> LLMProvider:
    """获取或创建 provider 实例。同一 (api_key, base_url) 组合复用实例。"""
    key = _cache_key(api_key, base_url)
    if key not in _provider_cache:
        _provider_cache[key] = build_provider(
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
        )
    return _provider_cache[key]


def clear_provider_cache() -> None:
    """清除 provider 缓存（会话切换时调用）。不影响本地引擎。"""
    _provider_cache.clear()


# ── 本地引擎管理 ──

_LOCAL_MARKER = "__local__"


def is_local_mode(base_url: str) -> bool:
    return base_url == _LOCAL_MARKER


def get_local_engine():
    """获取 LocalChatEngine 单例。"""
    return _local_engine


def get_local_engine_path() -> str:
    """获取当前加载的模型路径（解析后的精确路径）。"""
    return _local_engine_path


def is_local_engine_loading() -> bool:
    """本地引擎是否正在加载中（前端可用于禁用查询输入）。"""
    return _local_engine_loading


def _resolve_model_path(model_path: str) -> str:
    """将用户提供的模型路径解析为精确的模型文件路径。

    目录搜索优先级：GGUF (.gguf) > MLX (model.safetensors)。
    返回解析后的路径字符串。
    """
    from pathlib import Path

    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"模型路径不存在: {model_path}")

    if p.is_dir():
        if (p / "model.safetensors").exists() or (p / "model.safetensors.index.json").exists():
            return str(p)  # 已经是 MLX 目录
        # 搜索 GGUF
        all_ggufs = sorted(
            [f for f in p.rglob("*.gguf") if not f.name.startswith("mmproj-")]
        )
        if all_ggufs:
            return str(all_ggufs[0])
        # 搜索 MLX
        mlx_dirs = list(p.rglob("model.safetensors"))
        if mlx_dirs:
            return str(mlx_dirs[0].parent)
        raise FileNotFoundError(f"目录中未找到模型文件（已递归搜索）: {model_path}")

    return str(p)


def load_local_engine(model_path: str, n_ctx: int | None = None) -> str:
    """加载本地模型引擎。返回状态描述。

    目录搜索优先级：GGUF (.gguf) > MLX (model.safetensors)。
    Apple Silicon 上 MLX 原生加速，性能远优于 GGUF。
    加载成功后将 _local_engine_path 设为解析后的精确路径（非搜索根目录），
    以便持久化后重启时精确命中。
    """
    global _local_engine, _local_engine_path, _local_engine_loading

    # 解析路径（目录 → 精确文件），确保与 _local_engine_path 可比对
    resolved = _resolve_model_path(model_path)

    # 同一路径不重复加载
    if _local_engine is not None and _local_engine_path == resolved and _local_engine.is_loaded:
        return f"模型已加载: {_local_engine.model_path.name}"

    # 卸载旧的
    unload_local_engine()

    from ..llm.local_chat_engine import LocalChatEngine

    _local_engine_loading = True
    try:
        from .. import config as _cfg
        _local_engine = LocalChatEngine(
            resolved, n_ctx=n_ctx,
            flash_attn=_cfg.LLM_FLASH_ATTN,
            use_mmap=_cfg.LLM_USE_MMAP,
            n_batch=_cfg.LLM_N_BATCH,
            n_ubatch=_cfg.LLM_N_UBATCH,
        )
        _local_engine.load()
        # 保存解析后的精确路径，非用户提供的搜索根目录
        _local_engine_path = resolved
        return f"已加载: {_local_engine.model_path.name} ({_local_engine.n_ctx} ctx)"
    except Exception:
        _local_engine = None
        _local_engine_path = ""
        raise
    finally:
        _local_engine_loading = False


def unload_local_engine() -> None:
    """卸载本地引擎，释放显存。"""
    global _local_engine, _local_engine_path
    if _local_engine is not None:
        try:
            _local_engine.unload()
        except Exception:
            pass
        _local_engine = None
        _local_engine_path = ""


# ── HealthSnapshot → usage dict ──

def _health_to_usage(hs) -> dict:
    """将 HealthSnapshot 转为引擎期望的 usage dict 格式。"""
    d = hs.to_dict() if hasattr(hs, 'to_dict') else {}
    return {
        "input": getattr(hs, 'prefill_tokens', 0),
        "output": getattr(hs, 'output_tokens', 0),
        "elapsed": getattr(hs, 'total_ms', 0) / 1000 if hasattr(hs, 'total_ms') else 0,
        "health": d,
    }


# ── 同步 LLM 调用 ──

def _parse_stop_tokens(raw: str) -> list[str] | None:
    """解析逗号分隔的停止词字符串为列表。"""
    if not raw or not raw.strip():
        return None
    return [t.strip() for t in raw.split(",") if t.strip()]


def sync_call_llm(
    system: str,
    messages: list[dict],
    max_tokens: int = 1024,
    api_key: str = "",
    model: str = "",
    base_url: str = "",
    temperature: float = 0.0,
    thinking: bool | None = None,
    cancel_event=None,
    reuse_provider: bool = False,
    repeat_penalty: float = 1.08,
    top_k: int = 0,
    top_p: float = 0.95,
    stop: list[str] | None = None,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> tuple[str, dict]:
    """同步 LLM 调用。

    reuse_provider=True 时复用缓存的 provider 实例（Call 1 → Call 2），
    使 L0+L1 的 KV cache 跨调用命中。

    base_url="__local__" 时走本地引擎。
    thinking=None 时不发送 thinking 参数（模型默认行为）。

    反重复参数：
      repeat_penalty: 本地引擎用, 1.0=无惩罚, 1.08 推荐
      frequency_penalty/presence_penalty: API 用, -2.0~2.0
      stop: 停止词列表
      top_k: 小模型推荐 40-50, 0=禁用
    """
    # ── 本地模式 ──
    if is_local_mode(base_url):
        engine = get_local_engine()
        if engine is None:
            return "", {"input": 0, "output": 0, "elapsed": 0, "health": {}}

        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        try:
            text, health = engine.chat(
                full_messages,
                max_tokens=max_tokens,  # 0 = 不限制，透传至后端
                temperature=temperature or 0.0,
                repeat_penalty=repeat_penalty,
                top_p=top_p,
                stop=stop,
                thinking=thinking,
            )
            # 仅 thinking 开启时才剥离 <think> 标签；关闭时模型不输出标签
            if thinking:
                text = strip_thinking_tags(text)
            return text.strip(), _health_to_usage(health)
        except Exception as e:
            print(f"  [engine] Local LLM error: {type(e).__name__}: {e}")
            return "", {"input": 0, "output": 0, "elapsed": 0, "health": {}}

    # ── API 模式 ──
    if reuse_provider:
        provider = get_or_create_provider(
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
        )
    else:
        provider = build_provider(
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
        )

    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    last_err = None
    for attempt in range(2):  # 1 initial + 1 retry
        try:
            return provider.chat_sync(
                full_messages,
                model=model,
                max_tokens=max_tokens or config.LLM_MAX_TOKENS,
                temperature=temperature or 0.0,
                top_p=top_p,
                thinking=thinking,
                cancel_event=cancel_event,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
            )
        except Exception as e:
            last_err = e
            if attempt == 0 and _is_retryable(e):
                print(f"  [engine] LLM call failed ({type(e).__name__}), retrying in 2s...")
                time.sleep(2)
                # Rebuild provider on retry (connection may have been reset)
                clear_provider_cache()
                if reuse_provider:
                    provider = get_or_create_provider(
                        api_key=api_key, base_url=base_url, max_tokens=max_tokens,
                    )
                else:
                    provider = build_provider(
                        api_key=api_key, base_url=base_url, max_tokens=max_tokens,
                    )
            else:
                break
    raise last_err  # type: ignore[misc]


# ── 流式 LLM 调用 ──

async def stream_call_llm(
    system: str,
    messages: list[dict],
    max_tokens: int = 4096,
    api_key: str = "",
    model: str = "",
    base_url: str = "",
    temperature: float = 0.0,
    thinking: bool | None = None,
    on_token: Callable | None = None,
    reuse_provider: bool = False,
    repeat_penalty: float = 1.08,
    top_k: int = 0,
    top_p: float = 0.95,
    stop: list[str] | None = None,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    _loop_cancel: list[bool] | None = None,
) -> dict:
    """流式 LLM 调用。

    reuse_provider=True 时复用缓存的 provider 实例（Call 1 → Call 2）。
    base_url="__local__" 时走本地引擎（同步转异步）。

    反重复参数：
      repeat_penalty: 本地引擎用, 1.0=无惩罚, 1.08 推荐
      frequency_penalty/presence_penalty: API 用, -2.0~2.0
      stop: 停止词列表
      top_k: 小模型推荐 40-50, 0=禁用
    """
    # ── 本地模式（同步生成器 → 异步回调）──
    if is_local_mode(base_url):
        engine = get_local_engine()
        if engine is None:
            return {"input": 0, "output": 0, "elapsed": 0}

        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        t0 = time.time()
        total_tokens = 0
        final_health = None

        def _run_local_stream():
            """在后台线程中运行同步流式生成。"""
            nonlocal total_tokens, final_health
            _stream = None
            try:
                _stream = engine.chat_stream(
                    full_messages,
                    max_tokens=max_tokens,  # 0 = 不限制，透传至后端
                    temperature=temperature or 0.0,
                    repeat_penalty=repeat_penalty,
                    top_p=top_p,
                    stop=stop,
                    thinking=thinking,
                )
                for item in _stream:
                    if _loop_cancel and _loop_cancel[0]:
                        break
                    if hasattr(item, 'cache_hit_rate'):
                        final_health = item
                    elif isinstance(item, tuple):
                        is_thinking, text = item
                        total_tokens += 1
                        if on_token:
                            try:
                                on_token(text, is_thinking)
                            except Exception:
                                pass
                    else:
                        total_tokens += 1
                        if on_token:
                            try:
                                on_token(str(item), False)
                            except Exception:
                                pass
            finally:
                if _stream is not None:
                    try:
                        _stream.close()
                    except Exception:
                        pass

        try:
            await asyncio.to_thread(_run_local_stream)
            health_dict = _health_to_usage(final_health) if final_health else {}
            return {
                "input": health_dict.get("input", 0),
                "output": total_tokens,
                "elapsed": round(time.time() - t0, 2),
                "health": health_dict.get("health", {}),
                "finish_reason": getattr(final_health, 'finish_reason', 'unknown') if final_health else 'unknown',
            }
        except Exception as e:
            print(f"  [engine] Local LLM stream error: {type(e).__name__}: {e}")
            return {"input": 0, "output": 0, "elapsed": round(time.time() - t0, 2)}

    # ── API 模式 ──
    if reuse_provider:
        provider = get_or_create_provider(
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
        )
    else:
        provider = build_provider(api_key=api_key, base_url=base_url)

    full_messages = []
    if system:
        full_messages.append({"role": "system", "content": system})
    full_messages.extend(messages)

    t0 = time.time()
    total_tokens = 0
    finish_reason = "unknown"

    # <think> 标签流式解析器：API 模式下 content 流中的 <think> 标签
    # 通过此解析器分离 thinking/content，统一不同模型的输出格式
    _think_splitter = ThinkTagStreamSplitter()

    try:
        has_content_token = False
        reasoning_buffer = []

        async for chunk in provider.chat_stream(
            full_messages,
            model=model,
            max_tokens=max_tokens or config.LLM_MAX_TOKENS,
            temperature=temperature,
            top_p=top_p,
            thinking=thinking,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
        ):
            if _loop_cancel and _loop_cancel[0]:
                break
            if chunk["type"] == "token":
                # 通过 <think> 标签解析器分离 thinking/content
                if on_token:
                    for is_thinking, text in _think_splitter.feed(chunk["token"]):
                        if is_thinking:
                            reasoning_buffer.append(text)
                        else:
                            has_content_token = True
                        total_tokens += 1
                        on_token(text, is_thinking)
                else:
                    total_tokens += 1
            elif chunk["type"] == "reasoning_token":
                total_tokens += 1
                reasoning_buffer.append(chunk["token"])
                if on_token:
                    on_token(chunk["token"], True)
            elif chunk["type"] == "done":
                total_tokens = chunk.get("total_tokens", total_tokens)
                finish_reason = chunk.get("finish_reason", finish_reason)

        # 清空 splitter 缓冲区中剩余的部分标签文本
        if on_token:
            for is_thinking, text in _think_splitter.flush():
                if not is_thinking:
                    has_content_token = True
                total_tokens += 1
                on_token(text, is_thinking)

        # Fallback: 如果模型只输出了 reasoning 但没有 content（Qwen OptiQ 等），
        # 将 reasoning 内容作为兜底输出。需要剥离<think>标签和非答案文本。
        if not has_content_token and reasoning_buffer and on_token:
            reasoning_text = "".join(reasoning_buffer)
            clean = strip_thinking_tags(reasoning_text)
            if not clean:
                for token in reasoning_buffer:
                    on_token(token)
            else:
                # 尝试从 reasoning 中提取答案部分（去掉思考规划前缀和尾部思考）
                answer = _extract_answer_from_reasoning(clean)
                if answer:
                    on_token(answer)
                else:
                    on_token(clean)

        return {
            "input": 0,
            "output": total_tokens,
            "elapsed": round(time.time() - t0, 2),
            "finish_reason": finish_reason,
        }
    except Exception as e:
        print(f"  [engine] LLM stream error: {type(e).__name__}: {e}")
        return {"input": 0, "output": 0, "elapsed": round(time.time() - t0, 2), "finish_reason": "error"}
