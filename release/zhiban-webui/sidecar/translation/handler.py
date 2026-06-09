"""翻译请求 WebSocket 处理：提取 → 翻译 → 完成"""

import json
import time
import asyncio
from dataclasses import replace
from pathlib import Path
from fastapi import WebSocket

from .. import config
from ..config import DEBUG
from .extractor import extract_blocks
from .translator import translate_blocks

# Per-session cancel flag
_translation_cancel: asyncio.Event | None = None


async def handle_cancel_translation(ws: WebSocket, msg: dict):
    """Cancel an ongoing translation."""
    global _translation_cancel
    if _translation_cancel is None:
        _translation_cancel = asyncio.Event()
    _translation_cancel.set()
    await ws.send_json({
        "type": "status", "level": "info",
        "code": "translation_cancelled", "message": "翻译已取消",
    })


async def handle_translation_request(ws: WebSocket, msg: dict):
    global _translation_cancel
    """Process translation_request: extract PDF structure, stream translation."""
    file_path = msg.get("filePath", "")
    scope = msg.get("scope", "full")
    raw_api_key = msg.get("apiKey", "")
    if DEBUG:
        print(f"[DEBUG-TRANS] Received: filePath={file_path[:80] if file_path else '(empty)'}, scope={scope}, hasApiKey={bool(raw_api_key)}, apiKeyLen={len(raw_api_key)}", flush=True)

    if not file_path:
        await ws.send_json({
            "type": "status", "level": "error",
            "code": "translation_error", "message": "文件路径为空",
        })
        return

    # Phase 1: Extract with progress
    # Create cancel event before extraction for extraction-phase cancellation support.
    # Preserve cancelled state from previous event to avoid race conditions.
    old_event = _translation_cancel
    _translation_cancel = asyncio.Event()
    if old_event is not None and old_event.is_set():
        _translation_cancel.set()

    await ws.send_json({
        "type": "status", "level": "info",
        "code": "translation_extracting", "message": "正在提取文档结构...",
    })

    last_progress_sent = [0]  # mutable to update from callback
    loop = asyncio.get_running_loop()

    def on_extract_progress(current_page: int, total_pages: int):
        """Send progress every 5% to avoid flooding the WS.

        Runs in thread pool executor → must use run_coroutine_threadsafe
        to schedule WS send on the event loop.
        """
        pct = int(current_page / total_pages * 100)
        if pct - last_progress_sent[0] >= 5 or current_page == total_pages:
            last_progress_sent[0] = pct
            asyncio.run_coroutine_threadsafe(ws.send_json({
                "type": "status", "level": "info",
                "code": "translation_extracting",
                "message": f"提取文档结构 {current_page}/{total_pages} 页 ({pct}%)",
            }), loop)

    try:
        blocks = await asyncio.wait_for(
            loop.run_in_executor(None, extract_blocks, file_path, on_extract_progress),
            timeout=300,
        )
    except asyncio.TimeoutError:
        await ws.send_json({
            "type": "status", "level": "error",
            "code": "translation_error",
            "message": "文档提取超时。PDF 文件可能过大或包含复杂页面，请尝试分割文件后重试。",
        })
        return
    except Exception as e:
        await ws.send_json({
            "type": "status", "level": "error",
            "code": "translation_error",
            "message": f"文档提取失败: {e}",
        })
        return

    if not blocks:
        await ws.send_json({
            "type": "status", "level": "warn",
            "code": "translation_empty", "message": "未检测到文本内容",
        })
        return

    # Apply scope filter
    if scope == "selection":
        if "selectionRects" in msg and msg["selectionRects"]:
            # Coordinate-based selection: find sentences whose bboxes overlap the user's drag rects
            sels = msg["selectionRects"]
            selected_ids: set[str] = set()
            for sel in sels:
                pi = sel["pageIndex"]
                sx, sy, sw, sh = sel["x"], sel["y"], sel["w"], sel["h"]
                sel_right = sx + sw
                sel_bottom = sy + sh
                for b in blocks:
                    if b.page_num != pi:
                        continue
                    for s in b.sentences:
                        for r in s.rects:
                            if r.x < sel_right and r.x + r.w > sx and r.y < sel_bottom and r.y + r.h > sy:
                                selected_ids.add(s.id)
                                break
            # Filter blocks: only keep sentences that intersect at least one selection rect
            filtered: list = []
            for b in blocks:
                kept = [s for s in b.sentences if s.id in selected_ids]
                if kept:
                    filtered.append(replace(b, sentences=kept))
            if not filtered:
                total_rects = sum(1 for b in blocks for s in b.sentences for _ in s.rects)
                await ws.send_json({
                    "type": "status", "level": "warn",
                    "code": "translation_empty",
                    "message": f"框选范围未匹配到任何句子。文档共有 {len(blocks)} 个文本块，尝试调整框选区域。",
                })
                return
            blocks = filtered
        elif "selectionRange" in msg:
            # Legacy: page-range based selection
            sr = msg["selectionRange"]
            blocks = [b for b in blocks if sr["startPage"] <= b.page_num <= sr["endPage"]]
    elif scope == "page" and "page" in msg:
        page = msg["page"]
        filtered = [b for b in blocks if b.page_num == page]
        if not filtered:
            total_pages = max(b.page_num for b in blocks) + 1 if blocks else 0
            await ws.send_json({
                "type": "status", "level": "warn",
                "code": "translation_empty", "message": f"第 {page + 1} 页没有可翻译的文本内容（共 {total_pages} 页）",
            })
            return
        blocks = filtered

    total_sentences = sum(len(b.sentences) for b in blocks)

    # Send blocks to frontend (sentences without translations)
    blocks_payload = []
    for b in blocks:
        blocks_payload.append({
            "id": b.id,
            "type": b.type,
            "level": b.level,
            "sentences": [
                {
                    "id": s.id, "text": s.text, "translation": "", "isComplete": False,
                    "rects": [{"x": r.x, "y": r.y, "w": r.w, "h": r.h} for r in s.rects],
                }
                for s in b.sentences
            ],
            "pageNum": b.page_num,
            "bbox": {"x": b.bbox.x, "y": b.bbox.y, "w": b.bbox.w, "h": b.bbox.h} if b.bbox else None,
        })
    await ws.send_json({
        "type": "translation_blocks",
        "blocks": blocks_payload,
        "totalSentences": total_sentences,
    })
    if DEBUG:
        _with_rects = sum(1 for b in blocks_payload for s in b["sentences"] if s.get("rects"))
        print(f"[DEBUG-TRANS] Sent {len(blocks_payload)} blocks, {total_sentences} sentences, {_with_rects} with rects", flush=True)

    # Phase 2: Translate with streaming
    t0 = time.time()
    api_key = msg.get("apiKey", "")
    if api_key:
        api_key = api_key.encode("ascii", errors="ignore").decode("ascii").strip()
    model = msg.get("model", "") or config.LLM_MODEL
    base_url = msg.get("baseUrl", "") or config.LLM_BASE_URL
    use_local = msg.get("useLocal", False)  # 本地推理切换

    # 翻译强制禁用 thinking — 翻译是确定性任务，不需要推理
    thinking = False

    # Provider 自动路由: __local__ base_url 或 use_local → 本地引擎
    from ..llm.deepseek_proxy import llm_proxy
    from ..engine.llm_utils import is_local_mode as _is_local
    _using_local = use_local or _is_local(base_url)

    # 本地模式: 切换到翻译独立模型（等待伴读、检测内存、加载翻译模型）
    _model_switched = False
    if _using_local:
        from ..llm.model_manager import model_manager
        if model_manager.has_translation_model:
            _model_switched = await model_manager.switch_to_translation()

    if _using_local:
        pass  # translate_blocks handles local engine
    elif not api_key and not llm_proxy.is_available:
        if DEBUG:
            print(f"[DEBUG-TRANS] FAIL: no apiKey and no global key", flush=True)
        await ws.send_json({
            "type": "status", "level": "error",
            "code": "translation_error",
            "message": "未配置 LLM API Key，请在设置中填入 API Key 或在 sidecar/.env 中设置 LLM_API_KEY",
        })
        return
    # Estimate prefill context for user feedback
    total_chars = sum(len(s.text) for b in blocks for s in b.sentences)
    est_prefill_tokens = max(1, int(total_chars / 3 + 200))  # rough: 3 chars/tok + prompt overhead

    # 检测当前使用的翻译模型
    if _using_local:
        from ..llm.model_manager import model_manager as _mm
        from ..engine.llm_utils import get_local_engine as _gle
        _eng = _gle()
        if _mm.has_translation_model and _mm._translation_loaded:
            _model_label = f"本地 · {Path(_mm._translation_path).stem[:30]}"
        elif _eng and _eng.is_loaded:
            _model_label = f"本地 · {Path(str(_eng.model_path)).stem[:30]}"
        else:
            _model_label = "本地模型"
    else:
        _model_label = model or "API"

    await ws.send_json({
        "type": "status", "level": "info",
        "code": "translation_prefilling",
        "message": f"翻译 {total_sentences} 句 · ~{est_prefill_tokens:,} tok 上下文 · {_model_label}",
    })
    if DEBUG:
        print(f"[DEBUG-TRANS] Starting translation: {total_sentences} sentences, model={model}, thinking={thinking}", flush=True)

    total_tokens = 0
    last_speed_update = t0
    first_token_sent = False
    first_token_time = 0.0

    async def send_token(sentence_id: str, token: str, is_first: bool):
        nonlocal total_tokens, last_speed_update, first_token_sent, first_token_time
        total_tokens += 1
        now = time.time()
        # Track time-to-first-token (prefill latency)
        if not first_token_sent:
            first_token_sent = True
            first_token_time = now - t0
        # Send speed update every 2 seconds with translation_token
        speed_info = {}
        if now - last_speed_update > 2.0:
            elapsed = now - t0
            speed_info = {
                "elapsed": round(elapsed, 1),
                "tokensPerSec": round(total_tokens / elapsed, 1) if elapsed > 0 else 0,
            }
            last_speed_update = now
        if first_token_sent and first_token_time > 0:
            speed_info["firstTokenMs"] = round(first_token_time * 1000)
        await ws.send_json({
            "type": "translation_token",
            "sentenceId": sentence_id,
            "token": token,
            "isFirst": is_first,
            **speed_info,
        })
        # Force event loop I/O flush so tokens reach the frontend
        # immediately instead of all at once at translation end.
        await asyncio.sleep(0)

    try:
        result = await translate_blocks(
            blocks, concurrency=8, on_token=send_token,
            api_key=api_key, model=model, base_url=base_url,
            thinking=thinking, use_local=use_local,
            cancel_event=_translation_cancel,
        )
    except asyncio.CancelledError:
        await ws.send_json({
            "type": "status", "level": "info",
            "code": "translation_cancelled", "message": "翻译已取消",
        })
        return
    except Exception as e:
        if DEBUG:
            print(f"[DEBUG-TRANS] Exception during translate_blocks: {type(e).__name__}: {e}", flush=True)
        await ws.send_json({
            "type": "status", "level": "error",
            "code": "translation_error",
            "message": f"翻译失败: {e}",
        })
        return
    finally:
        _was_cancelled = _translation_cancel.is_set() if _translation_cancel else False
        _translation_cancel = None
        # 恢复伴读模型（如果之前切换了翻译模型）
        if _model_switched:
            await model_manager.switch_to_companion()

    # If no tokens were sent at all, report error
    if result.get("tokens_sent", 0) == 0:
        if _was_cancelled:
            return
        if DEBUG:
            print(f"[DEBUG-TRANS] 0 tokens sent! result={result}", flush=True)
        if _using_local:
            await ws.send_json({
                "type": "status", "level": "info",
                "code": "translation_info",
                "message": "翻译未产生结果，请检查本地模型是否正常加载",
            })
        else:
            await ws.send_json({
                "type": "status", "level": "error",
                "code": "translation_error",
                "message": "翻译未产生任何结果，请检查 API Key 是否有效",
            })
        return

    # Phase 3: Done
    if DEBUG:
        print(f"[DEBUG-TRANS] Done: {result['total_sentences']} sentences, {result.get('tokens_sent',0)} tokens, {result['duration_sec']}s", flush=True)
    await ws.send_json({
        "type": "translation_done",
        "totalBlocks": len(blocks),
        "totalSentences": total_sentences,
        "duration": result["duration_sec"],
    })
