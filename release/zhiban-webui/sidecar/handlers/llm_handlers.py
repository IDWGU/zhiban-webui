"""LLM-related WebSocket message handlers."""
import asyncio
import time
from pathlib import Path

from fastapi import WebSocket

from .. import config
from ..engine import engine as workflow_engine
from ..engine import WorkflowError
from ..engine.llm_utils import (
    clear_provider_cache, load_local_engine, unload_local_engine,
    get_local_engine, get_local_engine_path, is_local_mode, is_local_engine_loading,
)
from ..llm.deepseek_proxy import llm_proxy, build_provider
from ..rag.engine import rag_engine
from ..rag.graph_store import graph_store
from ..rag.embeddings import embedding_engine
from .query_utils import (
    sanitize_api_key, extract_doc_ids, build_rag_fallback, build_citations,
    parse_citation_refs, build_citation_context,
)


async def handle_llm_test(ws: WebSocket, msg: dict):
    """Test LLM connection with a simple API call."""
    api_key = sanitize_api_key(msg.get("apiKey", "") or config.LLM_API_KEY)
    model = sanitize_api_key(msg.get("model", "")).strip() or config.LLM_MODEL
    base_url = sanitize_api_key(msg.get("baseUrl", "")).strip() or config.LLM_BASE_URL

    if not api_key and base_url == config.LLM_BASE_URL and "deepseek" in (base_url or "").lower():
        await ws.send_json({
            "type": "llm_test_result",
            "success": False,
            "error": "API Key not configured",
        })
        return

    try:
        provider = build_provider(api_key=api_key, base_url=base_url)
        ok = await provider.test_connection(model=model)
        await ws.send_json({
            "type": "llm_test_result",
            "success": ok,
            "model": model,
        })
    except Exception as e:
        err_msg = str(e).encode("ascii", errors="replace").decode("ascii")
        await ws.send_json({
            "type": "llm_test_result",
            "success": False,
            "error": f"{type(e).__name__}: {err_msg}",
        })


async def handle_llm_list_models(ws: WebSocket, msg: dict):
    """Fetch available models from the LLM provider."""
    api_key = sanitize_api_key(msg.get("apiKey", "") or config.LLM_API_KEY)
    base_url = sanitize_api_key(msg.get("baseUrl", "")).strip() or config.LLM_BASE_URL

    try:
        provider = build_provider(api_key=api_key, base_url=base_url)
        models = await provider.list_models()
        if models:
            await ws.send_json({
                "type": "llm_models_result",
                "success": True,
                "models": models,
            })
        else:
            await ws.send_json({
                "type": "llm_models_result",
                "success": False,
                "error": "No models returned or endpoint does not support listing",
            })
    except Exception as e:
        err_msg = str(e).encode("ascii", errors="replace").decode("ascii")
        await ws.send_json({
            "type": "llm_models_result",
            "success": False,
            "error": f"{type(e).__name__}: {err_msg}",
        })


# ── 本地模型名 → 路径解析缓存 ──
_local_model_registry: dict[str, str] = {}  # model_name → resolved_path


def _scan_for_model(name_hint: str) -> str | None:
    """在配置的模型目录和 LM Studio 默认目录中搜索匹配 name_hint 的模型。

    匹配规则：name_hint 是模型名的子串（大小写不敏感）。
    返回匹配到的模型路径，未找到返回 None。
    """
    from pathlib import Path
    search_roots: list[Path] = []

    # 优先搜索项目内置模型目录
    bundled_llm = config._BUNDLED / "llm" if hasattr(config, '_BUNDLED') else None
    if bundled_llm and bundled_llm.exists():
        search_roots.append(bundled_llm)

    if config.LLM_MODEL_PATH:
        p = Path(config.LLM_MODEL_PATH)
        search_roots.append(p if p.is_dir() else p.parent)
    lmstudio = Path.home() / ".lmstudio" / "models"
    if lmstudio.exists():
        search_roots.append(lmstudio)

    # 去掉前端显示名中附加的后缀（如 " (MLX)"），提取核心模型名
    hint_lower = name_hint.lower()
    for suffix in (" (mlx)", " (gguf)"):
        hint_lower = hint_lower.replace(suffix, "")

    for root in search_roots:
        # MLX 目录
        for safetensors in root.rglob("model.safetensors"):
            model_dir = safetensors.parent
            dir_lower = model_dir.name.lower()
            if hint_lower in dir_lower or dir_lower in hint_lower:
                return str(model_dir)
        # GGUF 文件
        for gguf in root.rglob("*.gguf"):
            if "mmproj-" in gguf.name:
                continue
            stem_lower = gguf.stem.lower()
            parent_lower = gguf.parent.name.lower()
            if hint_lower in stem_lower or stem_lower in hint_lower or \
               hint_lower in parent_lower or parent_lower in hint_lower:
                return str(gguf)
    return None


async def _auto_switch_local_model(requested_model: str, base_url: str) -> None:
    """如果前端请求的模型名与当前加载的不同，自动查找路径并切换。

    本地模式下引擎是单例，模型名通过 user_query 的 model 字段传入。
    下拉选模型触发此逻辑，无需重启 Electron 即可生效。
    异步执行模型加载，避免阻塞事件循环。
    """
    _log = (Path(__file__).parent.parent.parent / ".conversations" / "auto-switch.log")
    if not requested_model:
        return
    engine = get_local_engine()
    if engine is None:
        return
    current_path = str(engine.model_path)

    # 检查当前路径是否已匹配请求的模型名
    if requested_model.lower() in current_path.lower():
        return

    # 检查缓存
    if requested_model in _local_model_registry:
        cached = _local_model_registry[requested_model]
        if cached.lower() in current_path.lower():
            return
        # 缓存命中 → 异步加载
        try:
            await asyncio.to_thread(load_local_engine, cached)
            _local_model_registry[requested_model] = cached
            _log.write_text(f"loaded from cache: {cached}")
            return
        except Exception as e:
            _log.write_text(f"cache load failed: {e}")

    # 搜索匹配的模型路径
    resolved = _scan_for_model(requested_model)
    if resolved and resolved.lower() not in current_path.lower():
        try:
            await asyncio.to_thread(load_local_engine, resolved)
            _local_model_registry[requested_model] = resolved
            _log.write_text(f"loaded: {resolved}")
        except Exception as e:
            _log.write_text(f"load failed: {e}")
    elif not resolved:
        _log.write_text(f"not found: {requested_model} (current={current_path})")


async def handle_user_query(ws: WebSocket, msg: dict):
    """Handle user query: V10 workflow engine — classify→search→MMR→filter→R2→answer"""
    query_text = msg.get("queryText", "")
    context = msg.get("context", {})
    active_doc = context.get("activeDoc", "")
    active_paragraph = context.get("activeParagraph", "")

    if not query_text.strip():
        await ws.send_json({
            "type": "status", "level": "warn",
            "code": "empty_query", "message": "Input is empty",
        })
        return

    api_key = sanitize_api_key(msg.get("apiKey", "") or config.LLM_API_KEY)
    llm_model = sanitize_api_key(msg.get("model", "")) or config.LLM_MODEL
    llm_base_url = sanitize_api_key(msg.get("baseUrl", "")).strip() or config.LLM_BASE_URL
    thinking_enabled = msg.get("thinking", False)  # 默认关：小模型 thinking 会吃掉全部 token
    history = msg.get("history", [])

    # Auto-detect local mode: local engine loaded → prefer it over API
    # 优先级：本地模型已加载 > API key 可用 > fallback
    local_check = get_local_engine()
    if local_check and local_check.is_loaded and not is_local_mode(llm_base_url):
        llm_base_url = "__local__"
        llm_model = str(local_check.model_path)
    elif not api_key and not is_local_mode(llm_base_url) and not llm_proxy.is_available:
        await _handle_query_fallback(ws, msg, query_text, active_doc, active_paragraph)
        return

    # 前端可能发送 baseUrl="__local__" 但实际上模型通过 llama-server API 提供。
    # 此时 __local__ 指向的是不存在的 Python 本地引擎，需要回退到 config 中的 API URL。
    if is_local_mode(llm_base_url) and not (local_check and local_check.is_loaded):
        if config.LLM_BASE_URL and not is_local_mode(config.LLM_BASE_URL):
            llm_base_url = config.LLM_BASE_URL
        await _handle_query_fallback(ws, msg, query_text, active_doc, active_paragraph)
        return

    # Local mode → check engine is loaded + auto-switch model if user selected different one
    if is_local_mode(llm_base_url):
        engine = get_local_engine()
        if engine is None or not engine.is_loaded:
            await ws.send_json({
                "type": "status", "level": "error",
                "code": "llm_error",
                "message": "本地模型未加载，请先在设置中设置并加载模型路径",
            })
            return
        # 用户在下拉中选了不同模型 → 自动查找路径并切换
        await _auto_switch_local_model(llm_model, llm_base_url)

    # Get or create conversation
    conv_id = context.get("conversationId", "default")
    workflow_engine.get_or_create_conversation(conv_id)

    # Sync open papers to engine
    open_papers = msg.get("openPapers", [])
    for p in open_papers:
        raw_pid = p.get("paperId") or p.get("paper_id", "")
        title = p.get("title", "")
        filename = p.get("filename", "")
        filepath = p.get("filepath", "")
        extracted_ids = extract_doc_ids(str(raw_pid)) if raw_pid else []
        if extracted_ids:
            for did in extracted_ids:
                workflow_engine.add_paper_to_conv(did, title, filename, filepath)
        elif raw_pid:
            workflow_engine.add_paper_to_conv(raw_pid, title, filename, filepath)

    # Extract paper IDs from activeDoc
    doc_ids = extract_doc_ids(active_doc)
    for did in doc_ids:
        try:
            pid = int(did)
            graph_info = graph_store.get_paper_info(pid)
            if graph_info:
                workflow_engine.add_paper_to_conv(pid, graph_info.get("title", f"Paper #{pid}"), "")
        except (ValueError, TypeError):
            pass

    # Build screen context
    screen_ctx = ""
    if active_paragraph:
        screen_ctx = f"Current reading: {active_doc}\nParagraph content: {active_paragraph}"

    # Parse citation refs (@Paper#N(section)) from user message and inject chunk context
    citation_refs = parse_citation_refs(query_text)
    if citation_refs:
        citation_ctx = build_citation_context(citation_refs)
        if citation_ctx:
            screen_ctx = screen_ctx + "\n\n" + citation_ctx if screen_ctx else citation_ctx

    # Inject history messages into engine
    if history and not workflow_engine.conv.messages:
        for m in history:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role in ("user", "assistant") and content.strip():
                workflow_engine.conv.messages.append({
                    "role": role, "content": content,
                    "timestamp": m.get("timestamp", int(time.time() * 1000)),
                    "mode": m.get("mode", ""),
                    "model": m.get("model", ""),
                })

    msg_id = f"msg_{int(time.time() * 1000)}"

    # Capture the event loop for thread-safe WS writes.
    # stream_call_llm runs the local model in asyncio.to_thread(), so all
    # on_* callbacks may fire from a background thread without an event loop.
    _loop = asyncio.get_running_loop()

    def _ws_send(data: dict) -> None:
        """Thread-safe wrapper: schedule ws.send_json on the event loop."""
        try:
            asyncio.run_coroutine_threadsafe(ws.send_json(data), _loop)
        except Exception:
            pass  # WS 已关闭时静默忽略，避免后台线程崩溃

    # Status callback → WebSocket
    def on_status(code: str, message: str):
        _ws_send({
            "type": "workflow_status",
            "code": code,
            "message": message,
            "conversationId": conv_id,
            "timestamp": int(time.time() * 1000),
        })

    # Token callback → streaming output
    is_first_token = [True]

    def on_token(token: str, is_thinking: bool = False):
        _ws_send({
            "type": "llm_token",
            "token": token,
            "messageId": msg_id,
            "isFirst": is_first_token[0],
            "isThinking": is_thinking,
            "conversationId": conv_id,
            "timestamp": int(time.time() * 1000),
        })
        is_first_token[0] = False

    # Health callback → structured health data per LLM call
    def on_health(data: dict):
        _ws_send({
            "type": "llm_health",
            "messageId": msg_id,
            "conversationId": conv_id,
            "call": data.get("call", ""),
            "timing": data.get("timing", {}),
            "tokens": data.get("tokens", {}),
            "memory": data.get("memory", {}),
            "debug_text": data.get("debug_text", ""),
            "timestamp": int(time.time() * 1000),
        })

    # Agent step callback → 前端思考过程面板
    def on_agent_step(data: dict):
        _ws_send({
            "type": "agent_step",
            "messageId": msg_id,
            "conversationId": conv_id,
            "stepIndex": data.get("stepIndex", 0),
            "phase": data.get("phase", "thinking"),
            "content": data.get("content", ""),
            "toolName": data.get("toolName", ""),
            "toolArgs": data.get("toolArgs", ""),
            "toolResult": data.get("toolResult", ""),
            "timestamp": int(time.time() * 1000),
        })

    # Create cancel event, inject into engine (keyed by conv_id 避免并发竞态)
    cancel_event = asyncio.Event()
    workflow_engine._cancel_events[conv_id] = cancel_event

    async def _run_query():
        return await workflow_engine.run(
            query_text,
            screen_ctx=screen_ctx,
            history_hint="",
            api_key=api_key,
            model=llm_model,
            base_url=llm_base_url,
            thinking=thinking_enabled,
            on_status=on_status,
            on_token=on_token,
            on_health=on_health,
            on_agent_step=on_agent_step,
        )

    ws_id = id(ws)
    task = asyncio.ensure_future(_run_query())
    # Store task for cancellation (managed by server module-level dict)

    usage: dict = {}
    total_tokens = 0
    result_refused = False
    result_expanded = False
    result_type = ""
    result_mode = ""
    result_loop_detected = False
    result_error = ""
    result_cancelled = False

    try:
        result = await task

        # 非流式回答兜底：如果引擎没有通过 on_token 流式推送过任何 token，
        # 则把整个 response 作为单次 token 发送（兼容旧的非流式路径）。
        usage = result.get("usage", {})
        total_tokens = usage.get("output", 0) if isinstance(usage, dict) else 0
        resp_text = result.get("response", "")
        if resp_text and total_tokens == 0 and is_first_token[0]:
            await ws.send_json({
                "type": "llm_token",
                "token": resp_text,
                "messageId": msg_id,
                "isFirst": True,
                "conversationId": conv_id,
                "timestamp": int(time.time() * 1000),
            })
            total_tokens = len(resp_text)

        result_refused = result.get("refused", False)
        result_expanded = result.get("expanded", False)
        result_type = result.get("type", "")
        result_mode = result.get("mode", "")
        result_loop_detected = result.get("loop_detected", False)

        # Send related papers (knowledge graph)
        if doc_ids:
            related = []
            seen = set()
            for pid_str in doc_ids[:2]:
                try:
                    pid = int(pid_str)
                except ValueError:
                    continue
                neighbors = graph_store.get_neighbors(pid, hops=1)[:3]
                for nb in neighbors:
                    if nb["paper_id"] not in seen:
                        seen.add(nb["paper_id"])
                        related.append({
                            "paperId": nb["paper_id"],
                            "title": nb.get("title", "")[:30],
                            "relationType": nb["relation"],
                            "year": nb.get("year", 0),
                            "relevance": 0.8,
                        })
            if related:
                await ws.send_json({
                    "type": "llm_related_papers",
                    "messageId": msg_id,
                    "conversationId": conv_id,
                    "papers": related,
                })

        # Send citations from RAG search results
        try:
            raw_results = await rag_engine.search(
                query_text, context_doc_ids=doc_ids or None, top_k=5,
            )
            if raw_results:
                citations = build_citations(raw_results)
                if citations:
                    await ws.send_json({
                        "type": "llm_citation",
                        "messageId": msg_id,
                        "conversationId": conv_id,
                        "citations": citations,
                    })
        except Exception:
            pass  # citation is best-effort, don't fail the whole response

    except asyncio.CancelledError:
        result_cancelled = True
    except WorkflowError as e:
        result_error = f"{e.step}失败"
    except Exception as e:
        result_error = f"AI 回答生成失败: {e}"

    finally:
        workflow_engine._cancel_events.pop(conv_id, None)

        # 始终发送 llm_done，确保前端停止流式 UI
        try:
            await ws.send_json({
                "type": "llm_done",
                "messageId": msg_id,
                "conversationId": conv_id,
                "totalTokens": total_tokens,
                "duration": usage.get("elapsed", 0) if isinstance(usage, dict) else 0,
                "usage": {
                    "input": usage.get("input", 0) if isinstance(usage, dict) else 0,
                    "output": usage.get("output", 0) if isinstance(usage, dict) else 0,
                },
                "refused": result_refused,
                "expanded": result_expanded,
                "responseType": result_type,
                "mode": result_mode,
                "loopDetected": result_loop_detected,
                "model": llm_model,
                "cancelled": result_cancelled,
                "error": result_error,
            })
        except Exception:
            pass  # 如果 WS 已断开，忽略发送失败

        # 发送错误状态（在 llm_done 之后，确保前端先停止流式再显示错误）
        if result_cancelled:
            try:
                await ws.send_json({
                    "type": "status", "level": "info",
                    "code": "cancelled", "message": "查询已取消",
                })
            except Exception:
                pass
        elif result_error:
            try:
                await ws.send_json({
                    "type": "status", "level": "error",
                    "code": "llm_error",
                    "message": result_error,
                })
            except Exception:
                pass


async def _handle_query_fallback(
    ws: WebSocket, msg: dict, query_text: str,
    active_doc: str, active_paragraph: str,
):
    """RAG retrieval fallback when no API Key is available."""
    context = msg.get("context", {})
    top_k = max(1, min(20, msg.get("topK", 5)))
    msg_id = f"msg_{int(time.time() * 1000)}"

    doc_ids = extract_doc_ids(active_doc)
    try:
        results = await rag_engine.search(query_text, doc_ids, top_k=top_k)
    except Exception:
        results = []

    fallback_text = build_rag_fallback(
        query_text, results, top_k,
        screen_doc=active_doc, screen_text=active_paragraph,
    )
    await ws.send_json({
        "type": "llm_token", "token": fallback_text,
        "messageId": msg_id, "isFirst": True,
        "conversationId": context.get("conversationId", "default"),
        "timestamp": int(time.time() * 1000),
    })
    await ws.send_json({
        "type": "llm_done", "messageId": msg_id,
        "conversationId": context.get("conversationId", "default"),
        "totalTokens": len(fallback_text), "duration": 0,
    })
    if results:
        citations = build_citations(results)
        if citations:
            await ws.send_json({
                "type": "llm_citation", "messageId": msg_id,
                "conversationId": context.get("conversationId", "default"),
                "citations": citations,
            })


async def handle_model_config(ws: WebSocket, msg: dict):
    """管理模型配置（本地模型路径、嵌入模型、缓存目录）。

    支持的操作:
      {"type": "model_config", "action": "get"}  — 获取当前配置
      {"type": "model_config", "action": "set_local_model", "path": "/path/to/model.gguf"}  — 设置本地模型路径
      {"type": "model_config", "action": "set_embedding_model", "model": "jinaai/jina-embeddings-v5-text-nano"}  — 切换嵌入模型
    """
    action = msg.get("action", "")

    if action == "get":
        local_eng = get_local_engine()
        # 返回实际加载的模型路径（精确路径），非 config 中的搜索根目录
        actual_path = ""
        if local_eng and local_eng.is_loaded:
            actual_path = str(local_eng.model_path)
        elif config.LLM_MODEL_PATH:
            actual_path = config.LLM_MODEL_PATH
        await ws.send_json({
            "type": "model_config_result",
            "action": "get",
            "config": {
                "llm_model_path": actual_path if is_local_mode(config.LLM_BASE_URL) else "",
                "llm_base_url": config.LLM_BASE_URL,
                "translation_model_path": config.TRANSLATION_MODEL_PATH,
                "embedding_model": embedding_engine.model_name,
                "model_cache_dir": str(config.MODEL_CACHE),
                "chroma_dir": str(config.CHROMA_PERSIST_DIR),
                "debug": config.DEBUG,
                "embedding_available": embedding_engine.is_available,
                "local_engine_loaded": local_eng is not None and local_eng.is_loaded if local_eng else False,
                "local_engine_backend": (
                    Path(local_eng.model_path).name if local_eng and local_eng.is_loaded else ""
                ),
                "local_engine_loading": is_local_engine_loading(),
                "llm_flash_attn": config.LLM_FLASH_ATTN,
                "llm_use_mmap": config.LLM_USE_MMAP,
                "llm_n_batch": config.LLM_N_BATCH,
                "llm_n_ubatch": config.LLM_N_UBATCH,
            },
        })

    elif action == "set_local_model":
        model_path = (msg.get("path") or "").strip()
        if model_path:
            p = Path(model_path)
            if not p.exists():
                await ws.send_json({
                    "type": "model_config_result",
                    "action": "set_local_model",
                    "success": False,
                    "error": f"路径不存在: {model_path}",
                })
                return

        # 更新环境变量和配置
        import os as _os
        _os.environ["LLM_MODEL_PATH"] = model_path
        config.LLM_MODEL_PATH = model_path

        if not model_path:
            # 空路径 → 卸载
            unload_local_engine()
            await ws.send_json({
                "type": "model_config_result",
                "action": "set_local_model",
                "success": True,
                "path": "",
                "status": "已卸载",
            })
            return

        # 先持久化路径（加载之前），确保即使加载失败路径也不丢失
        try:
            from ..server import save_sidecar_settings
            save_sidecar_settings(LLM_MODEL_PATH=model_path)
        except Exception as e:
            print(f"   ⚠️  persist model path failed: {e}")

        # 检查模型是否已加载（解析后再比对，避免重复加载）
        from ..engine.llm_utils import _resolve_model_path
        try:
            resolved_check = _resolve_model_path(model_path)
        except FileNotFoundError:
            resolved_check = model_path  # 路径不存在时仍然尝试加载，让 load_local_engine 报错

        existing = get_local_engine()
        already_loaded = (
            existing is not None
            and existing.is_loaded
            and get_local_engine_path() == resolved_check
        )

        # 只有确实需要加载时才发 loading_model 状态
        if not already_loaded:
            await ws.send_json({
                "type": "status", "code": "loading_model", "level": "info",
                "message": f"正在加载本地模型...",
            })

        try:
            loop = asyncio.get_running_loop()
            status = await loop.run_in_executor(None, load_local_engine, model_path)
            # 持久化解析后的精确路径（非用户提供的搜索根目录）
            eng = get_local_engine()
            resolved = str(eng.model_path) if eng else resolved_check
            await ws.send_json({
                "type": "model_config_result",
                "action": "set_local_model",
                "success": True,
                "path": resolved,
                "status": status,
                "config": {
                    "local_engine_loaded": True,
                    "local_engine_backend": Path(resolved).name if resolved else "",
                },
            })
            try:
                from ..server import save_sidecar_settings
                save_sidecar_settings(LLM_MODEL_PATH=resolved)
            except Exception as e:
                print(f"   ⚠️  persist resolved path failed: {e}")
        except Exception as e:
            await ws.send_json({
                "type": "model_config_result",
                "action": "set_local_model",
                "success": False,
                "error": f"模型加载失败: {e}",
            })

    elif action == "set_translation_model":
        model_path = (msg.get("path") or "").strip()
        import os as _os
        _os.environ["TRANSLATION_MODEL_PATH"] = model_path
        config.TRANSLATION_MODEL_PATH = model_path
        # 配置 ModelManager
        from ..llm.model_manager import model_manager
        model_manager.configure(translation=model_path)
        await ws.send_json({
            "type": "model_config_result",
            "action": "set_translation_model",
            "success": True,
            "path": model_path,
        })
        # 持久化
        try:
            from ..server import save_sidecar_settings
            save_sidecar_settings(TRANSLATION_MODEL_PATH=model_path)
        except Exception:
            pass

    elif action == "set_debug":
        enabled = msg.get("enabled", False)
        config.DEBUG = bool(enabled)
        import os as _os
        _os.environ["SIDECAR_DEBUG"] = "1" if enabled else "0"
        await ws.send_json({
            "type": "model_config_result",
            "action": "set_debug",
            "success": True,
            "enabled": config.DEBUG,
        })
        # 持久化
        try:
            from ..server import save_sidecar_settings
            save_sidecar_settings(SIDECAR_DEBUG="1" if enabled else "0")
        except Exception:
            pass
        # 触发热重启：touch sidecar/__init__.py 让 uvicorn reloader 重启 worker
        try:
            _touch = Path(__file__).resolve().parent.parent / "__init__.py"
            _touch.touch()
        except Exception:
            pass
        # 延时退出，让响应有时间发送
        loop = asyncio.get_running_loop()
        loop.call_later(1.0, _os._exit, 0)

    elif action == "set_llm_params":
        params = msg.get("params", {})
        changed = []
        import os as _os

        # flash_attn (bool)
        if "flash_attn" in params:
            val = bool(params["flash_attn"])
            config.LLM_FLASH_ATTN = val
            _os.environ["LLM_FLASH_ATTN"] = "1" if val else "0"
            changed.append("flash_attn")

        # use_mmap (bool)
        if "use_mmap" in params:
            val = bool(params["use_mmap"])
            config.LLM_USE_MMAP = val
            _os.environ["LLM_USE_MMAP"] = "1" if val else "0"
            changed.append("use_mmap")

        # n_batch (int)
        if "n_batch" in params:
            val = max(256, int(params["n_batch"]))
            config.LLM_N_BATCH = val
            _os.environ["LLM_N_BATCH"] = str(val)
            changed.append("n_batch")

        # n_ubatch (int)
        if "n_ubatch" in params:
            val = max(128, int(params["n_ubatch"]))
            config.LLM_N_UBATCH = val
            _os.environ["LLM_N_UBATCH"] = str(val)
            changed.append("n_ubatch")

        await ws.send_json({
            "type": "model_config_result",
            "action": "set_llm_params",
            "success": True,
            "changed": changed,
            "status": f"已更新: {', '.join(changed)}，即将重启以应用新参数",
        })

        # 持久化
        try:
            from ..server import save_sidecar_settings
            save_sidecar_settings(
                LLM_FLASH_ATTN=config.LLM_FLASH_ATTN,
                LLM_USE_MMAP=config.LLM_USE_MMAP,
                LLM_N_BATCH=config.LLM_N_BATCH,
                LLM_N_UBATCH=config.LLM_N_UBATCH,
            )
        except Exception:
            pass

        # 触发热重启
        try:
            _touch = Path(__file__).resolve().parent.parent / "__init__.py"
            _touch.touch()
        except Exception:
            pass
        loop = asyncio.get_running_loop()
        loop.call_later(1.0, _os._exit, 0)

    elif action == "set_embedding_model":
        model_name = (msg.get("model") or "").strip()
        if not model_name:
            await ws.send_json({
                "type": "model_config_result",
                "action": "set_embedding_model",
                "success": False,
                "error": "模型名称不能为空",
            })
            return

        await ws.send_json({
            "type": "status", "code": "loading_embedding", "level": "info",
            "message": f"正在加载嵌入模型 {model_name}...",
        })

        # 设置进度回调（从工作线程发回 WS）
        def on_progress(pct: int, msg_text: str):
            asyncio.ensure_future(ws.send_json({
                "type": "embedding_progress",
                "percent": pct,
                "message": msg_text,
            }))
        embedding_engine.set_progress_callback(on_progress)

        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, embedding_engine.reload, model_name)
            await ws.send_json({
                "type": "model_config_result",
                "action": "set_embedding_model",
                "success": True,
                "model": model_name,
                "dim": embedding_engine.dim,
            })
            # 持久化 embedding 模型选择
            try:
                from ..server import save_sidecar_settings
                save_sidecar_settings(EMBEDDING_MODEL=model_name)
            except Exception:
                pass
        except Exception as e:
            await ws.send_json({
                "type": "model_config_result",
                "action": "set_embedding_model",
                "success": False,
                "error": f"加载失败: {e}",
        })
        finally:
            embedding_engine.set_progress_callback(None)

    elif action == "scan_local_models":
        """扫描本地可用的模型文件（.gguf 文件和 MLX 目录）。

        返回 [{name, path}]，path 可直接用于 set_local_model。
        扫描路径:
          - models/bundled/llm/ (项目内置模型)
          - LLM_MODEL_PATH 指向的文件或目录
          - MODEL_CACHE 目录
          - ~/.lmstudio/models/ (LM Studio 默认存放位置, publisher/model 二级目录)
          - ~/models/
        """
        models: list[dict] = []
        scan_paths: list[Path] = []

        # 优先扫描项目内置模型目录
        bundled_llm = config._BUNDLED / "llm" if hasattr(config, '_BUNDLED') else None
        if bundled_llm and bundled_llm.exists():
            scan_paths.append(bundled_llm)

        # 从 LLM_MODEL_PATH 扫描
        if config.LLM_MODEL_PATH:
            p = Path(config.LLM_MODEL_PATH)
            if p.is_dir():
                scan_paths.append(p)
            elif p.is_file():
                models.append({"name": p.stem, "path": str(p)})

        # 从 MODEL_CACHE 扫描
        cache = config.MODEL_CACHE
        if cache.exists():
            scan_paths.append(cache)

        # LM Studio 默认模型目录: ~/.lmstudio/models/publisher/model_name/
        lmstudio = Path.home() / ".lmstudio" / "models"
        if lmstudio.exists():
            scan_paths.append(lmstudio)

        # 其他常见目录
        for home_dir in [Path.home() / "models"]:
            if home_dir.exists():
                scan_paths.append(home_dir)

        scanned_paths: set[str] = set()

        def _should_skip(name: str) -> bool:
            """跳过 HuggingFace 缓存目录和无关目录"""
            return (
                name.startswith(".") or
                name.startswith("models--") or  # HF 缓存: models--BAAI--bge-m3
                name == "snapshots" or           # HF 快照目录
                name == "blobs" or               # HF 大文件缓存
                name in ("tokens", ".locks", ".no_exist")
            )

        def _scan_dir(dir_path: Path, depth: int = 0):
            """递归扫描目录，depth 0=顶级，1= publisher 级，2+=模型级"""
            if depth > 3:
                return
            try:
                for entry in dir_path.iterdir():
                    if _should_skip(entry.name):
                        continue
                    if entry.is_dir():
                        # 检查 MLX 模型目录
                        if (entry / "model.safetensors").exists():
                            p = str(entry)
                            if p not in scanned_paths:
                                models.append({"name": f"{entry.name} (MLX)", "path": p})
                                scanned_paths.add(p)
                        else:
                            _scan_dir(entry, depth + 1)
                    elif entry.is_file() and entry.suffix == ".gguf" and "mmproj-" not in entry.name:
                        p = str(entry)
                        if p not in scanned_paths:
                            # 用文件名 stem 做展示名（去掉后缀），加上父目录区分同名
                            name = entry.stem
                            if name in scanned_paths:
                                name = f"{entry.parent.name}/{name}"
                            models.append({"name": name, "path": p})
                            scanned_paths.add(p)
            except PermissionError:
                pass

        for scan_dir in scan_paths:
            _scan_dir(scan_dir)

        models.sort(key=lambda m: m["name"])
        await ws.send_json({
            "type": "llm_models_result",
            "success": True,
            "models": [m["name"] for m in models],          # 旧前端兼容
            "model_entries": models,                         # 新前端使用 {name, path}
        })

    elif action == "test_local_model":
        """测试本地引擎是否已加载并可工作。"""
        engine = get_local_engine()
        if engine is None or not engine.is_loaded:
            await ws.send_json({
                "type": "llm_test_result",
                "success": False,
                "error": "本地模型未加载，请先在「本地模型路径」中设置并点击「应用」",
            })
            return

        snap = engine.health_snapshot()
        await ws.send_json({
            "type": "llm_test_result",
            "success": True,
            "model": f"{snap['backend']} ({snap['n_ctx']} ctx)",
        })

    elif action == "set_sampling_params":
        """从前端接收采样参数并写入 config。"""
        params = msg.get("params", {})
        import os as _os
        updated = {}
        for key, val in params.items():
            if hasattr(config, key):
                setattr(config, key, val)
                _os.environ[key] = str(val)
                updated[key] = val
        await ws.send_json({
            "type": "model_config_result",
            "action": "set_sampling_params",
            "success": True,
            "updated": updated,
        })

    else:
        await ws.send_json({
            "type": "model_config_result",
            "action": action,
            "success": False,
            "error": f"未知操作: {action}",
        })


async def handle_swap_model(ws: WebSocket, msg: dict):
    """热切换模型：卸载当前模型并加载新模型，不重启 sidecar。"""
    model_name = (msg.get("model") or "").strip()
    if not model_name:
        await ws.send_json({
            "type": "status", "level": "error",
            "code": "swap_failed", "message": "模型名称为空",
        })
        return

    # 通知前端开始加载
    await ws.send_json({
        "type": "status", "code": "loading_model", "level": "info",
        "message": f"正在热切换到 {model_name}...",
    })

    try:
        from ..engine.llm_utils import load_local_engine, unload_local_engine, get_local_engine, get_local_engine_path

        old_name = ""
        old_eng = get_local_engine()
        if old_eng and old_eng.is_loaded:
            old_name = Path(get_local_engine_path() or "").name if get_local_engine_path() else ""

        # 相同模型 → 无需切换
        if old_name and model_name.lower() in old_name.lower():
            await ws.send_json({
                "type": "model_config_result",
                "action": "set_local_model",
                "success": True,
                "path": get_local_engine_path() or "",
                "status": f"模型已是 {old_name}，无需切换",
                "config": {"local_engine_loaded": True, "local_engine_backend": old_name},
            })
            return

        # 卸载旧模型
        if old_eng and old_eng.is_loaded:
            unload_local_engine()

        # 加载新模型
        resolved = _scan_for_model(model_name)
        if not resolved:
            await ws.send_json({
                "type": "status", "level": "error",
                "code": "swap_failed",
                "message": f"未找到模型: {model_name}",
            })
            return

        loop = asyncio.get_running_loop()
        status = await loop.run_in_executor(None, load_local_engine, resolved)

        new_eng = get_local_engine()
        new_name = Path(str(new_eng.model_path)).name if new_eng else resolved

        await ws.send_json({
            "type": "model_config_result",
            "action": "set_local_model",
            "success": True,
            "path": resolved,
            "status": f"已热切换到 {new_name}",
            "config": {
                "local_engine_loaded": True,
                "local_engine_backend": new_name,
            },
        })
    except Exception as e:
        await ws.send_json({
            "type": "status", "level": "error",
            "code": "swap_failed",
            "message": f"模型热切换失败: {e}",
        })


async def handle_regenerate_last(ws: WebSocket, msg: dict):
    """用户确认后重新生成上一条回复：清 KV cache + 提高反重复参数。

    流程：
    1. 获取当前会话的最后一条用户消息
    2. 删除最后一条 assistant 消息
    3. 用更高 repeat_penalty + 更高温度 + 不复用 provider 重新生成
    """
    conv = workflow_engine.conv
    if not conv:
        await ws.send_json({
            "type": "status", "level": "error",
            "code": "no_conversation", "message": "没有活跃的对话",
        })
        return

    # 找到最后一条用户消息
    last_user_msg = None
    for m in reversed(conv.messages):
        if m.get("role") == "user":
            last_user_msg = m
            break

    if not last_user_msg:
        await ws.send_json({
            "type": "status", "level": "error",
            "code": "no_question", "message": "没有找到可重新生成的问题",
        })
        return

    question = last_user_msg.get("content", "")
    if not question.strip():
        await ws.send_json({
            "type": "status", "level": "error",
            "code": "empty_question", "message": "上一条问题为空",
        })
        return

    # 删除最后一组 user-assistant 消息（engine.run 会重新添加）
    if conv.messages and conv.messages[-1].get("role") == "assistant":
        conv.messages.pop()
    if conv.messages and conv.messages[-1].get("role") == "user":
        conv.messages.pop()

    # 清除 KV cache 和 provider 缓存，避免重复模式污染
    clear_provider_cache()
    if is_local_mode(config.LLM_BASE_URL):
        engine = get_local_engine()
        if engine is not None and engine.is_loaded:
            try:
                engine.kv_cache_clear()
            except Exception:
                pass

    # 获取前端传来的参数
    api_key = sanitize_api_key(msg.get("apiKey", "") or config.LLM_API_KEY)
    llm_model = sanitize_api_key(msg.get("model", "")) or config.LLM_MODEL
    llm_base_url = sanitize_api_key(msg.get("baseUrl", "")).strip() or config.LLM_BASE_URL

    # 构建消息
    msg_id = f"msg_{int(time.time() * 1000)}"
    _loop = asyncio.get_running_loop()

    def _ws_send(data: dict) -> None:
        _loop.call_soon_threadsafe(
            lambda d=data: asyncio.ensure_future(ws.send_json(d))
        )

    is_first_token = [True]

    def on_token(token: str, is_thinking: bool = False):
        _ws_send({
            "type": "llm_token",
            "token": token,
            "messageId": msg_id,
            "isFirst": is_first_token[0],
            "isThinking": is_thinking,
            "timestamp": int(time.time() * 1000),
        })
        is_first_token[0] = False

    def on_status(code: str, message: str):
        _ws_send({
            "type": "workflow_status",
            "code": code,
            "message": message,
            "timestamp": int(time.time() * 1000),
        })

    # 使用引擎重新生成，但传入更高的反重复参数
    # 通过临时修改 config 实现（最简方式）
    import os as _os
    _orig_repeat = config.LLM_REPEAT_PENALTY
    _orig_temp = config.LLM_TEMPERATURE
    try:
        config.LLM_REPEAT_PENALTY = min(config.LLM_REPEAT_PENALTY + 0.08, 1.3)
        config.LLM_TEMPERATURE = min(config.LLM_TEMPERATURE + 0.2, 1.0)

        result = await workflow_engine.run(
            question,
            screen_ctx="",
            history_hint="",
            api_key=api_key,
            model=llm_model,
            base_url=llm_base_url,
            thinking=msg.get("thinking", True),
            on_status=on_status,
            on_token=on_token,
        )
    finally:
        config.LLM_REPEAT_PENALTY = _orig_repeat
        config.LLM_TEMPERATURE = _orig_temp

    # 发送完成消息
    usage = result.get("usage", {})
    total_tokens = usage.get("output", 0) if isinstance(usage, dict) else 0
    # 非流式回答推送
    resp_text = result.get("response", "")
    if resp_text and total_tokens == 0 and result.get("refused") != True:
        await ws.send_json({
            "type": "llm_token", "token": resp_text,
            "messageId": msg_id, "isFirst": True,
            "timestamp": int(time.time() * 1000),
        })
        total_tokens = len(resp_text)
    await ws.send_json({
        "type": "llm_done",
        "messageId": msg_id,
        "totalTokens": total_tokens,
        "duration": usage.get("elapsed", 0) if isinstance(usage, dict) else 0,
        "usage": {
            "input": usage.get("input", 0) if isinstance(usage, dict) else 0,
            "output": usage.get("output", 0) if isinstance(usage, dict) else 0,
        },
        "refused": result.get("refused", False),
        "responseType": result.get("type", ""),
        "mode": result.get("mode", ""),
        "regenerated": True,
        "model": result.get("model", msg.get("model", "")),
    })
