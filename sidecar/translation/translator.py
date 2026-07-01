"""LLM 流式翻译：逐句翻译 + KV cache 感知 + API Context Caching。

本地 GGUF 模式：
  Prefill system+原文一次进 KV cache，truncate_cache() 每句后清除
  instruction+output，逐句间原文缓存复用。仅 GGUF 支持精确截断；
  MLX fallback 到全量 reset。

API 模式：
  每句请求携带完整原文，利用 DeepSeek Context Caching 跨请求缓存
  system_prompt+原文前缀，仅 prefill 变化的指令后缀。
  thinking 强制禁用（翻译不需要推理）。
"""

import asyncio
import time
import traceback
from typing import Callable, Awaitable

from .extractor import Block


def _ensure_local_engine():
    """获取翻译用的本地引擎。

    优先使用 ModelManager 管理翻译独立模型切换；
    如果未配置独立翻译模型，退回复用伴读模型单例。
    """
    from ..llm.model_manager import model_manager
    from ..engine.llm_utils import get_local_engine, load_local_engine
    from .. import config as global_config

    # 有独立翻译模型 → ModelManager 已在 handler 完成 switch_to_translation
    if model_manager.has_translation_model and model_manager._translation_loaded:
        engine = get_local_engine()
        if engine is not None and engine.is_loaded:
            return engine

    # 无独立翻译模型 → 复用伴读模型
    engine = get_local_engine()
    if engine is not None and engine.is_loaded:
        return engine

    if not global_config.LLM_MODEL_PATH:
        raise RuntimeError("本地模型路径未配置 (LLM_MODEL_PATH 为空)")
    load_local_engine(global_config.LLM_MODEL_PATH)
    return get_local_engine()


# 精简 system prompt，减少 prefill 量
TRANSLATION_SYSTEM_PROMPT = (
    "你是论文学术翻译器。将任意语言的学术句子翻译为简体中文。"
    "必须只输出中文译文，禁止输出英文或原文，禁止加任何解释、注释、前缀。"
    "禁止输出句子序号（如 [9]、[10]），序号仅用于标记原文位置。"
    "专业术语保留英文原文并括号标注中文，如 \"XRD (X射线衍射)\"。"
    "保留文献引用标记如 [69] 和 LaTeX 公式如 $H_2O_2$ 不变。"
    "如果原文已经是中文，直接输出原文。"
)


async def translate_blocks(
    blocks: list[Block],
    *,
    concurrency: int = 3,
    on_token: Callable[[str, str, bool], Awaitable[None]],
    api_key: str = "",
    model: str = "",
    base_url: str = "",
    thinking: bool = False,
    use_local: bool = False,
    cancel_event: asyncio.Event | None = None,
) -> dict:
    """
    Translate all sentences in blocks with KV cache optimization.

    on_token(sentence_id, token, is_first) is called for every LLM token.

    use_local=True → 本地 GGUF/MLX：
      GGUF: 利用 truncate_cache() 每句后清除 instruction+output，保留
      system+原文的 KV cache 供下句复用，原文仅 prefill 一次。
      MLX: fallback 到全量 reset。

    use_local=False → API 模式：
      每句请求携带完整原文上下文，利用 DeepSeek Context Caching
      使 system_prompt+原文前缀跨请求缓存，仅 prefill 变化的指令后缀。
    """
    from ..engine.llm_utils import ThinkingStreamFilter, strip_thinking_tags

    local_engine = None
    if use_local:
        local_engine = _ensure_local_engine()

    sem = asyncio.Semaphore(concurrency if not use_local else 1)
    t0 = time.time()
    tokens_sent = 0

    ws_dead = False

    async def counted_on_token(sid: str, token: str, is_first: bool):
        nonlocal tokens_sent, ws_dead
        if ws_dead:
            return
        tokens_sent += 1
        try:
            await on_token(sid, token, is_first)
        except RuntimeError as e:
            if "after sending 'websocket.close'" in str(e) or "already completed" in str(e):
                ws_dead = True
            else:
                raise

    # ── 本地模式：稳定前缀 + raw_generate_stream（修复版）──
    # 修复：去掉 chat template + truncate_cache 组合导致的 RoPE 位置错位。
    # 策略：构造稳定前缀 {SYSTEM}\n\n{全文原文}\n\n，每句附加变化指令，
    # 依靠 llama.cpp 原生 prefix caching 自动复用前缀部分的 KV cache。
    if use_local and local_engine:
        n_ctx = local_engine.n_ctx
        max_ctx = int(n_ctx * 0.85)

        all_sents = []
        for block in blocks:
            for s in block.sentences:
                all_sents.append(s)

        full_original = "\n\n".join(s.text for s in all_sents)
        full_original_tok = local_engine.count_tokens(full_original)
        sys_tok = local_engine.count_tokens(TRANSLATION_SYSTEM_PROMPT)
        base_tok = sys_tok + full_original_tok

        print(f"\n[翻译] 本地 {local_engine.backend_name}, {len(all_sents)}句, "
              f"n_ctx={n_ctx}, base={base_tok} tok", flush=True)

        # 构建稳定前缀（system + 全文原文）
        cache_prefix = f"{TRANSLATION_SYSTEM_PROMPT}\n\n以下是待翻译论文原文：\n\n{full_original}\n\n"

        # 逐句翻译：利用 llama.cpp prefix caching 自动复用前缀
        # 首句翻译自然完成 prefill，无需独立 warmup（之前 160s 空跑）
        for idx, s in enumerate(all_sents):
            if cancel_event and cancel_event.is_set():
                break

            prompt_text = cache_prefix + f"请翻译第{idx+1}句：\n{s.text}"
            prompt_tok = local_engine.count_tokens(prompt_text)

            if prompt_tok >= max_ctx:
                # 超出上下文：只用简洁 prompt（无全文上下文）
                prompt_text = f"{TRANSLATION_SYSTEM_PROMPT}\n\n翻译为中文：\n{s.text}"

            t_start = time.time()
            full_translation = []
            is_first = True
            _aborted = False

            try:
                # Run blocking generator in thread to keep event loop alive for WS pings
                def _generate():
                    tokens = []
                    for item in local_engine.raw_generate_stream(
                        prompt_text, max_tokens=256, temperature=0.3,
                    ):
                        if cancel_event and cancel_event.is_set():
                            break
                        if isinstance(item, str):
                            tokens.append(str(item))
                    return tokens

                loop = asyncio.get_running_loop()
                token_list = await loop.run_in_executor(None, _generate)

                if cancel_event and cancel_event.is_set():
                    _aborted = True
                else:
                    for tok_str in token_list:
                        full_translation.append(tok_str)
                        await counted_on_token(s.id, tok_str, is_first)
                        is_first = False

            except Exception as e:
                import traceback
                print(f"  [翻译] 第{idx+1}句生成失败: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                full_translation = [f"[翻译中断: {type(e).__name__}]"]

            if _aborted or ws_dead:
                break

            elapsed = time.time() - t_start
            raw = "".join(full_translation)
            translation = strip_thinking_tags(raw).strip()
            s.translation = translation

            if (idx + 1) % 20 == 0 or idx == 0:
                print(f"  [{idx+1}/{len(all_sents)}] {elapsed:.1f}s, "
                      f"{prompt_tok} prompt tok → {translation[:60]}", flush=True)

        return {"total_sentences": len(all_sents), "duration_sec": round(time.time() - t0, 1), "tokens_sent": tokens_sent}

    else:
        # ── API 模式：带 DeepSeek Context Caching 的并发翻译 ──
        # 每句请求都包含完整原文上下文，利用 DeepSeek Context Caching
        # 使 system_prompt + 原文前缀跨请求缓存，每句仅 prefill 指令后缀。
        # thinking 已强制禁用（翻译不需要推理）。
        all_sents = []
        for block in blocks:
            for s in block.sentences:
                all_sents.append(s)

        full_original = "\n\n".join(s.text for s in all_sents)

        async def translate_one(sentence_id: str, text: str, block: Block, sent_idx: int):
            async with sem:
                # 每句都包含完整原文作为上下文，使 DeepSeek Context Caching
                # 在 service 侧缓存 system_prompt + 原文前缀，仅继续 prefill 指令后缀
                user_msg = (
                    f"以下是待翻译论文原文：\n\n{full_original}\n\n"
                    f"请翻译第{sent_idx+1}句：\n{text}"
                )

                from ..llm.deepseek_proxy import llm_proxy

                is_first = True
                has_content = False
                reasoning_buffer = []
                finish_reason = ""

                try:
                    async for chunk in llm_proxy.chat_stream(
                        query=user_msg,
                        context="",
                        system_prompt=TRANSLATION_SYSTEM_PROMPT,
                        thinking=False,
                        api_key=api_key,
                        model=model or "deepseek-v4-flash",
                        base_url=base_url,
                    ):
                        if cancel_event and cancel_event.is_set():
                            return
                        if chunk["type"] == "token":
                            has_content = True
                            await counted_on_token(sentence_id, chunk["token"], is_first)
                            is_first = False
                        elif chunk["type"] == "reasoning_token":
                            reasoning_buffer.append(chunk["token"])
                        elif chunk["type"] == "done":
                            finish_reason = chunk.get("finish_reason", "")
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    raise RuntimeError(
                        f"第{sent_idx+1}句翻译失败: {type(e).__name__}: {e}"
                    ) from e

                if not has_content and finish_reason and finish_reason not in ("stop", "", "length"):
                    raise RuntimeError(
                        f"第{sent_idx+1}句翻译被拒绝 (finish_reason={finish_reason})"
                    )

                # Fallback: 某些模型仍可能输出 reasoning_content 而非 content
                if not has_content and reasoning_buffer:
                    reasoning_text = "".join(reasoning_buffer)
                    clean = strip_thinking_tags(reasoning_text)
                    if clean:
                        await counted_on_token(sentence_id, clean, True)

        tasks = []
        for block in blocks:
            for si, s in enumerate(block.sentences):
                tasks.append(asyncio.create_task(translate_one(s.id, s.text, block, si)))

        pending = set(tasks)
        while pending:
            if cancel_event and cancel_event.is_set():
                for t in pending:
                    t.cancel()
                break
            done, pending = await asyncio.wait(pending, timeout=2.0)

        errors = []
        for t in tasks:
            if t.done() and not t.cancelled():
                exc = t.exception()
                if exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
        if errors:
            print(f"[translator] {len(errors)}/{len(tasks)} sentences failed:", flush=True)
            for msg in errors[:3]:
                print(f"  {msg}", flush=True)

        total = sum(len(b.sentences) for b in blocks)
        return {
            "total_sentences": total,
            "duration_sec": round(time.time() - t0, 1),
            "tokens_sent": tokens_sent,
            "errors": errors,
        }

    total = sum(len(b.sentences) for b in blocks)
    return {"total_sentences": total, "duration_sec": round(time.time() - t0, 1), "tokens_sent": tokens_sent}
