"""
知伴 Sidecar — FastAPI + WebSocket 主服务
接收前端查询 → RAG检索 → LLM流式回复

Phase 2: ChatState 状态机 + WebSocket 状态通知
"""

import asyncio
import json
import os
import platform
import signal
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

# 增大线程栈大小避免 llama.cpp C++ 推理时栈溢出崩溃
# 默认 macOS 线程栈 512KB，16MB 用户栈不够 llama.cpp 的深度递归
# SIGBUS 崩溃地址在 STACK GUARD 区域，确认是栈溢出
threading.stack_size(64 * 1024 * 1024)  # 64MB for llama.cpp inference threads

import shutil

import psutil
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from . import config
from .engine import engine as workflow_engine
from .llm.session_state import ChatState
from .model_downloader import ensure_models
from .rag.embeddings import embedding_engine
from .rag.graph_store import graph_store
from .rag.vector_store import vector_store

try:
    from .translation.handler import handle_translation_request, handle_cancel_translation
except ImportError:
    handle_translation_request = None  # type: ignore
    handle_cancel_translation = None    # type: ignore

from .handlers.llm_handlers import (
    handle_llm_test,
    handle_llm_list_models,
    handle_user_query,
    handle_model_config,
    handle_regenerate_last,
    handle_swap_model,
)
# Voice handlers removed — STT module deleted
from .handlers.index_handlers import (
    handle_import_vector_store,
    handle_build_index,
    handle_build_control,
    handle_import_paper,
    handle_add_papers,
    handle_list_library,
    handle_clear_vector_store,
    handle_remove_paper_vectors,
    handle_list_indexed_papers,
    handle_delete_library_papers,
)

_ready = False

# Startup progress — polled by frontend via /startup-status
_startup_status: dict = {"step": "initializing", "message": "正在启动...", "progress": 0}

# Active WebSocket clients for state broadcast
_ws_clients: set[WebSocket] = set()


def _load_embedding_background():
    """后台加载 embedding 模型"""
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_load_embedding_async())
    except Exception:
        pass
    finally:
        loop.close()


async def _load_embedding_async():
    try:
        await asyncio.wait_for(
            asyncio.to_thread(embedding_engine.load),
            timeout=60,
        )
        print(f"   ✅ Embedding 就绪: {embedding_engine.model_name}, dim={embedding_engine.dim}")
    except Exception as e:
        print(f"   Embeddings: unavailable ({e})")


def _load_llm_background(model_path: str):
    """后台加载 LLM 引擎（llama-server 或本地引擎）"""
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_load_llm_async(model_path))
    except Exception:
        pass
    finally:
        loop.close()


async def _load_llm_async(model_path: str):
    # 尝试 llama-server
    _mtp_pid = await _start_llama_server(model_path)
    if _mtp_pid:
        os.environ["LLM_BASE_URL"] = "http://127.0.0.1:18923/v1"
        config.LLM_BASE_URL = "http://127.0.0.1:18923/v1"
        config.LLM_MODEL_PATH = ""
        return

    # 回退到本地引擎
    print(f"   ⏳ Auto-loading companion model: {Path(model_path).name[:50]}...")
    await _auto_load_local_model(model_path)


# ===== sidecar 设置持久化（JSON 文件，存放本地模型路径等） =====

_SIDECAR_SETTINGS_KEYS = {
    "LLM_MODEL_PATH", "LLM_BASE_URL", "TRANSLATION_MODEL_PATH", "EMBEDDING_MODEL",
    "SIDECAR_DEBUG",
    "LLM_FLASH_ATTN", "LLM_USE_MMAP", "LLM_N_BATCH", "LLM_N_UBATCH",
}
_SIDECAR_BOOL_KEYS = {"SIDECAR_DEBUG", "LLM_FLASH_ATTN", "LLM_USE_MMAP"}
_SIDECAR_INT_KEYS = {"LLM_N_BATCH", "LLM_N_UBATCH"}


def load_sidecar_settings() -> dict:
    """启动时加载 sidecar settings，写入 config 模块和 os.environ。"""
    sf = config.SIDECAR_SETTINGS
    if not sf.exists():
        return {}
    try:
        data = json.loads(sf.read_text())
        for key in _SIDECAR_SETTINGS_KEYS:
            val = data.get(key)
            if val is None:
                continue
            # 类型转换：JSON 中可能存 bool/int/str，统一映射到 Python 类型
            if key in _SIDECAR_BOOL_KEYS:
                py_val = bool(val) if not isinstance(val, bool) else val
                os.environ[key] = "1" if py_val else "0"
                if key == "SIDECAR_DEBUG":
                    config.DEBUG = py_val
                else:
                    setattr(config, key, py_val)
            elif key in _SIDECAR_INT_KEYS:
                py_val = int(val)
                os.environ[key] = str(py_val)
                setattr(config, key, py_val)
            else:
                os.environ[key] = str(val)
                setattr(config, key, val)
        # 确保模型路径是绝对路径：相对路径可能因 CWD 不同而找不到文件
        for _pk in ("LLM_MODEL_PATH", "TRANSLATION_MODEL_PATH"):
            _pv = getattr(config, _pk, "")
            if _pv and not os.path.isabs(str(_pv)):
                _abs = str(config.MODEL_CACHE.parent / _pv)
                if os.path.exists(_abs):
                    os.environ[_pk] = _abs
                    setattr(config, _pk, _abs)
        print(f"   Sidecar settings loaded: {sf}")
        return data
    except Exception as e:
        print(f"   Sidecar settings: load failed ({e})")
        return {}


def save_sidecar_settings(**kwargs) -> None:
    """持久化 sidecar 设置到 JSON 文件，覆盖写入。"""
    sf = config.SIDECAR_SETTINGS
    try:
        sf.parent.mkdir(parents=True, exist_ok=True)
        prev = {}
        if sf.exists():
            prev = json.loads(sf.read_text())
        # bool 值统一存为 JSON bool（非字符串），int 存为 JSON number
        cleaned = {}
        for k, v in kwargs.items():
            if k in _SIDECAR_BOOL_KEYS:
                cleaned[k] = bool(v)
            elif k in _SIDECAR_INT_KEYS:
                cleaned[k] = int(v)
            else:
                cleaned[k] = str(v)
        prev.update(cleaned)
        prev = {k: v for k, v in prev.items() if k in _SIDECAR_SETTINGS_KEYS}
        sf.write_text(json.dumps(prev, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"   Sidecar settings: save failed ({e})")


async def _broadcast_state(state: ChatState, payload: dict | None = None) -> None:
    """Broadcast state change to all connected WebSocket clients."""
    if payload is None:
        payload = {
            "state": state.value,
            "timestamp": int(time.time() * 1000),
        }
    msg = {"type": "status", **payload}

    dead: list[WebSocket] = []
    for ws in _ws_clients:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)

    for ws in dead:
        _ws_clients.discard(ws)


async def _broadcast_conv_list() -> None:
    """Broadcast updated conversation list to all connected clients."""
    convs = []
    for cid, c in workflow_engine.conversations.items():
        convs.append({
            "id": cid, "name": c.name,
            "messageCount": len(c.messages),
            "paperCount": len(c.open_papers),
            "topic": c.current_topic,
            "isActive": cid == workflow_engine.active_conv_id,
            "updatedAt": c.updated_at,
        })
    msg = {"type": "conversation_list", "conversations": convs}
    dead: list[WebSocket] = []
    for ws in _ws_clients:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.discard(ws)


async def _auto_load_local_model(path: str) -> None:
    """启动后自动加载持久化的本地模型（不阻断 lifespan）。"""
    p = Path(path)
    if not p.exists():
        print(f"   ❌ Companion model not found: {path}")
        return
    try:
        from .engine.llm_utils import load_local_engine, get_local_engine
        import psutil, time
        t0 = time.time()
        mem_before = psutil.virtual_memory().percent
        loop = asyncio.get_running_loop()
        status = await loop.run_in_executor(None, load_local_engine, path)
        elapsed = time.time() - t0
        eng = get_local_engine()
        mem_after = psutil.virtual_memory().percent
        model_name = Path(path).name
        print(f"   ✅ 伴读模型就绪: {status} ({elapsed:.0f}s, 内存 {mem_before}%→{mem_after}%)")
        # 通知所有已连接的客户端模型加载完成
        for ws in list(_ws_clients):
            try:
                await ws.send_json({
                    "type": "model_config_result",
                    "action": "set_local_model",
                    "success": True,
                    "path": str(p),
                    "status": status,
                    "config": {"local_engine_loaded": True, "local_engine_backend": model_name},
                })
            except Exception:
                pass
    except Exception as e:
        print(f"   ❌ 伴读模型加载失败: {e}")
        # 通知失败
        for ws in list(_ws_clients):
            try:
                await ws.send_json({
                    "type": "model_config_result",
                    "action": "set_local_model",
                    "success": False,
                    "error": f"模型加载失败: {e}",
                })
            except Exception:
                pass


async def _start_llama_server(model_path: str) -> int | None:
    """启动 llama-server（LM Studio 内置版本，支持 MTP 等新架构）。

    成功返回 PID，后续 LLM 请求走 http://127.0.0.1:18923/v1。
    失败返回 None，回退到 llama-cpp-python 本地引擎。
    """
    if not model_path:
        return None
    p = Path(model_path)
    if not p.exists():
        print(f"   llama-server: model not found: {model_path}")
        return None

    # 优先用项目内置的 llama-server，否则回退到 LM Studio 安装的
    bundled = Path(__file__).parent.parent / "sidecar-dist" / "llama-server"
    if bundled.exists():
        server_bin = str(bundled)
    else:
        candidates = sorted(
            Path.home().glob(".lmstudio/extensions/backends/llama.cpp-*/llama-server"),
            key=lambda x: x.parent.name,
        )
        if not candidates:
            print("   llama-server: not found, falling back to local engine")
            return None
        server_bin = str(candidates[-1])
    args = [
        server_bin, "--model", str(p),
        "--port", "18923", "--host", "127.0.0.1",
        "--n-gpu-layers", "-1", "--ctx-size", "32768",
        "--alias", "companion",
    ]

    print(f"   ⏳ llama-server: starting with {p.name[:50]}...")
    import subprocess
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 轮询等待 /health 就绪（最长 30s）
    for _ in range(30):
        await asyncio.sleep(1)
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", 18923)
            writer.write(b"GET /health HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
            await writer.drain()
            data = await asyncio.wait_for(reader.read(1024), timeout=2)
            writer.close()
            if b"200" in data or b"ok" in data:
                print(f"   ✅ llama-server ready (PID {proc.pid})")
                os.environ["LLM_BASE_URL"] = "http://127.0.0.1:18923/v1"
                config.LLM_BASE_URL = "http://127.0.0.1:18923/v1"
                return proc.pid
        except Exception:
            pass
    print(f"   ❌ llama-server: startup timed out")
    proc.kill()
    return None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _ready

    # 恢复持久化的 sidecar 设置
    _startup_status.update(step="loading_settings", message="正在加载设置...", progress=5)
    load_sidecar_settings()
    print("   Sidecar settings: loaded")

    # ── /ready 立即返回，不再等待模型加载 ──
    _ready = True
    _startup_status.update(step="ready", message="就绪", progress=10)

    # ── 以下全部后台执行 ──
    async def _background_init():
        try:
            # 配置模型管理器
            from .llm.model_manager import model_manager
            model_manager.configure(
                companion=config.LLM_MODEL_PATH,
                translation=config.TRANSLATION_MODEL_PATH,
            )
            _companion_name = Path(config.LLM_MODEL_PATH).name if config.LLM_MODEL_PATH else "(未配置)"
            _trans_name = Path(config.TRANSLATION_MODEL_PATH).name if config.TRANSLATION_MODEL_PATH else "(未配置)"
            print(f"   Models configured: 伴读={_companion_name[:50]} 翻译={_trans_name[:50]}")

            # LLM 后端选择
            from .engine.llm_utils import is_local_mode as _is_local
            if config.LLM_BASE_URL and not _is_local(config.LLM_BASE_URL):
                print(f"   API mode: using {config.LLM_BASE_URL}")
                config.LLM_MODEL_PATH = ""
            elif config.LLM_MODEL_PATH:
                _model_path = config.LLM_MODEL_PATH
                await _load_llm_async(_model_path)

            # 知识图谱
            _startup_status.update(step="loading_graph", message="正在加载知识图谱...", progress=40)
            try:
                graph_store.load()
                print(f"   Knowledge graph: {graph_store.paper_count} papers")
            except Exception as e:
                print(f"   Knowledge graph: unavailable ({e})")

            # 嵌入模型
            _startup_status.update(step="loading_embeddings", message="正在加载嵌入模型...", progress=60)
            import threading as _thr
            _thr.Thread(target=_load_embedding_background, daemon=True, name="embedding-loader").start()

            await ensure_models()
            print("   Models: ready")

            # 会话持久化
            _startup_status.update(step="restoring_sessions", message="正在恢复会话...", progress=80)
            from .persistence import ConversationStore
            _persistence = ConversationStore(str(config.CONVERSATIONS_DB))
            db_dir = config.CONVERSATIONS_DB.parent
            db_dir.mkdir(parents=True, exist_ok=True)
            await _persistence.initialize()
            workflow_engine.set_persistence(_persistence)
            workflow_engine.session_state.set_on_state_change(
                lambda state, payload: _broadcast_state(state, payload)
            )
            try:
                convs, active_id = await _persistence.load_all()
                if convs:
                    workflow_engine.conversations = convs
                    workflow_engine.active_conv_id = active_id or next(iter(convs))
                    print(f"   Conversations: {len(convs)} loaded")
                else:
                    workflow_engine.new_conversation(name="新对话")
                    print("   Conversations: initialized default")
            except Exception as e:
                print(f"   Conversations: load failed ({e})")
                workflow_engine.new_conversation(name="新对话")

            _startup_status.update(step="ready", message="就绪", progress=100)
            print("   === 知伴就绪 ===")
        except Exception as e:
            print(f"   ❌ Background init error: {e}")
            import traceback
            traceback.print_exc()

    asyncio.ensure_future(_background_init())

    yield

    _ready = False


app = FastAPI(title="知伴 Sidecar", version="0.2.0", lifespan=lifespan)

# CORS: 允许前端（Electron / Vite dev server）通过 fetch 访问 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebUI: static file serving for distribution build
# 注意：此为占位声明，实际 mount 操作需在所有 API 路由注册之后执行
if getattr(sys, 'frozen', False):
    _STATIC_DIR = Path(sys.executable).parent / "dist-web"
else:
    _STATIC_DIR = Path(__file__).resolve().parent.parent / "dist-web"
_STATIC_MOUNTED = False

# Track running query tasks per WebSocket for cancellation
_active_queries: dict[int, asyncio.Task] = {}


try:
    import multipart  # noqa: F401
    _HAS_MULTIPART = True
except ImportError:
    _HAS_MULTIPART = False


if _HAS_MULTIPART:
    from fastapi import UploadFile, File

    @app.post("/upload")
    async def upload_file(file: UploadFile = File(...)):
        """文件上传端点 — WebUI 模式下前端通过此接口上传论文"""
        if not file.filename:
            return JSONResponse(status_code=400, content={"error": "文件名不能为空"})
        ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext not in ("pdf", "docx", "txt", "md"):
            return JSONResponse(status_code=400, content={"error": f"不支持的文件类型: .{ext}"})
        temp_dir = config.KNOWLEDGE_BASE / "_uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)
        dest = temp_dir / file.filename
        if dest.exists():
            stem = dest.stem
            counter = 1
            while dest.exists():
                dest = temp_dir / f"{stem}_{counter}.{ext}"
                counter += 1
        try:
            with open(dest, "wb") as f:
                shutil.copyfileobj(file.file, f)
        finally:
            file.file.close()
        return {"filePath": str(dest), "filename": file.filename, "size": dest.stat().st_size}
else:
    @app.post("/upload")
    async def upload_file_unavailable(request: Request):
        """文件上传端点（降级模式）"""
        return JSONResponse(
            status_code=501,
            content={"error": "文件上传功能不可用：缺少 python-multipart 依赖，请运行: pip install python-multipart"}
        )


@app.get("/system-info")
async def system_info():
    """返回精确的硬件和系统信息，用于前端兼容性检查。"""
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(str(config.KNOWLEDGE_BASE)) if config.KNOWLEDGE_BASE.exists() else None

    # Apple Silicon 检测
    machine = platform.machine()
    is_apple_silicon = machine in ("arm64",)
    is_apple = platform.system() == "Darwin"

    # MPS (GPU) 可用性
    mps_available = False
    gpu_name = ""
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            mps_available = True
            # macOS 上尝试读取 GPU 名称（仅 Apple Silicon）
            try:
                import subprocess
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0 and result.stdout.strip():
                    gpu_name = result.stdout.strip()
                else:
                    # 尝试通过 system_profiler 获取 GPU 信息
                    result = subprocess.run(
                        ["system_profiler", "SPHardwareDataType"],
                        capture_output=True, text=True, timeout=5
                    )
                    for line in result.stdout.split("\n"):
                        if "Chip" in line or "Apple" in line:
                            gpu_name = line.split(":")[-1].strip()
                            break
            except Exception:
                pass
    except ImportError:
        pass

    # 内存类型说明
    if is_apple_silicon:
        memory_type = "Apple Silicon 统一内存"
    elif is_apple:
        memory_type = "Mac 物理内存"
    else:
        memory_type = "物理内存"

    return {
        "platform": platform.system(),
        "machine": machine,
        "is_apple_silicon": is_apple_silicon,
        "cpu": {
            "physical_cores": psutil.cpu_count(logical=False) or os.cpu_count() or 0,
            "logical_cores": os.cpu_count() or 0,
        },
        "memory": {
            "total_gb": round(mem.total / (1024**3), 1),
            "available_gb": round(mem.available / (1024**3), 1),
            "type": memory_type,
            "note": "macOS Apple Silicon 上为 CPU+GPU 共享的统一内存"
                if is_apple_silicon else "",
        },
        "gpu": {
            "available": mps_available,
            "name": gpu_name,
            "backend": "MPS (Metal Performance Shaders)" if mps_available else "N/A",
        },
        "disk": {
            "total_gb": round(disk.total / (1024**3), 1) if disk else 0,
            "free_gb": round(disk.free / (1024**3), 1) if disk else 0,
        },
    }


@app.get("/file-content")
async def file_content(path: str = ""):
    """WebUI 模式：前端通过此端点获取已上传文件的内容"""
    from fastapi.responses import FileResponse
    from urllib.parse import unquote
    file_path = Path(unquote(path))
    if not file_path.exists():
        return JSONResponse(status_code=404, content={"error": "文件不存在"})
    # 安全检查：只允许访问 KNOWLEDGE_BASE 下的文件
    try:
        file_path.resolve().relative_to(config.KNOWLEDGE_BASE.resolve())
    except ValueError:
        return JSONResponse(status_code=403, content={"error": "禁止访问"})
    return FileResponse(str(file_path))

@app.get("/ready")
async def ready():
    """Returns 200 only after lifespan has completed and _ready is True.
    Used by Electron/frontend to gate WebSocket connection attempts."""
    if _ready:
        return {"ready": True}
    return JSONResponse(
        status_code=503,
        content={"ready": False, "message": "Server is starting up..."}
    )


@app.get("/startup-status")
async def startup_status():
    """Returns current startup progress for frontend polling during waitForReady."""
    return _startup_status


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "graphs_papers": graph_store.paper_count,
        "graphs_edges": graph_store.edge_count,
        "vectors_chunks": vector_store.chunk_count,
        "embedding_available": embedding_engine.is_available,
        "stt_available": False,
        "session_state": workflow_engine.session_state.health_payload(),
    }


@app.websocket("/ws")
async def websocket_handler(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)

    if not _ready:
        await ws.send_json({
            "type": "status",
            "code": "unready",
            "message": "Server is still initializing, please wait",
        })
        await ws.close(code=4001, reason="server_not_ready")
        _ws_clients.discard(ws)
        return

    conversation_id = f"conv_{int(time.time() * 1000)}"
    await ws.send_json({
        "type": "status", "state": "idle",
        "code": "ready", "message": "知伴就绪",
    })

    # 连接后立即推送会话列表
    convs = []
    for cid, c in workflow_engine.conversations.items():
        convs.append({
            "id": cid, "name": c.name,
            "messageCount": len(c.messages),
            "paperCount": len(c.open_papers),
            "topic": c.current_topic,
            "isActive": cid == workflow_engine.active_conv_id,
            "updatedAt": c.updated_at,
        })
    try:
        await ws.send_json({"type": "conversation_list", "conversations": convs})
    except Exception as e:
        print(f"  ERROR: conversation_list send failed: {e}", flush=True)

    try:
        async for raw_msg in ws.iter_text():
            try:
                msg = json.loads(raw_msg)
            except json.JSONDecodeError:
                continue

            if msg.get("type") == "ping":
                await ws.send_json({"type": "pong", "timestamp": int(time.time() * 1000)})
                continue

            if msg.get("type") == "user_query":
                ws_id = id(ws)

                # Phase 2: Check session state
                state = workflow_engine.session_state
                can_accept, reason = state.can_accept()

                if reason == "queued":
                    # COMPRESSING: enqueue message as a callable for later re-processing
                    await state.enqueue(lambda m=msg, w=ws: handle_user_query(w, m))
                    await ws.send_json({
                        "type": "status", "state": "compressing",
                        "code": "queued",
                        "message": "消息已排队，压缩完成后自动处理",
                    })
                    continue

                if not can_accept:
                    # THINKING: reject duplicate
                    await ws.send_json({
                        "type": "status", "state": state.state.value,
                        "code": "busy", "message": reason,
                    })
                    continue

                if ws_id in _active_queries and not _active_queries[ws_id].done():
                    await ws.send_json({
                        "type": "status", "level": "warn",
                        "code": "busy", "message": "上一个查询仍在处理中",
                    })
                else:
                    task = asyncio.ensure_future(handle_user_query(ws, msg))
                    task._conv_id = msg.get("context", {}).get("conversationId", "")
                    _active_queries[ws_id] = task

            elif msg.get("type") == "new_conversation":
                conv_id = workflow_engine.new_conversation(
                    name=msg.get("name", "新对话")
                )
                await ws.send_json({
                    "type": "conversation_created",
                    "conversationId": conv_id,
                    "name": workflow_engine.conv.name if workflow_engine.conv else "新对话",
                })

            elif msg.get("type") == "switch_conversation":
                target_id = msg.get("conversationId", "")
                if target_id and target_id in workflow_engine.conversations:
                    # 如果旧会话有查询在运行，不关闭它，让查询在后台继续
                    old_id = workflow_engine.active_conv_id
                    has_running_query = any(
                        old_id == getattr(t, '_conv_id', None) and not t.done()
                        for t in _active_queries.values()
                    ) if old_id and old_id != target_id else False
                    if old_id and old_id != target_id and not has_running_query:
                        await workflow_engine.close_conversation(old_id)
                    workflow_engine.active_conv_id = target_id
                    conv = workflow_engine.conv
                    await ws.send_json({
                        "type": "conversation_switched",
                        "conversationId": target_id,
                        "messages": conv.messages if conv else [],
                        "openPapers": conv.open_papers if conv else [],
                        "currentTopic": conv.current_topic if conv else "",
                    })
                    if config.DEBUG:
                        for p in (conv.open_papers if conv else []):
                            print(f"  [switch] paper: {p.get('title','')} path={p.get('filepath','(无)')}")

            elif msg.get("type") == "list_conversations":
                convs = []
                for cid, c in workflow_engine.conversations.items():
                    convs.append({
                        "id": cid,
                        "name": c.name,
                        "messageCount": len(c.messages),
                        "paperCount": len(c.open_papers),
                        "topic": c.current_topic,
                        "isActive": cid == workflow_engine.active_conv_id,
                        "updatedAt": c.updated_at,
                    })
                await ws.send_json({
                    "type": "conversation_list",
                    "conversations": convs,
                })

            elif msg.get("type") == "delete_conversation":
                target_id = msg.get("conversationId", "")
                if target_id and target_id in workflow_engine.conversations:
                    workflow_engine.delete_conversation(target_id)
                    convs = []
                    for cid, c in workflow_engine.conversations.items():
                        convs.append({
                            "id": cid, "name": c.name,
                            "messageCount": len(c.messages),
                            "paperCount": len(c.open_papers),
                            "topic": c.current_topic,
                            "isActive": cid == workflow_engine.active_conv_id,
                            "updatedAt": c.updated_at,
                        })
                    await ws.send_json({
                        "type": "conversation_list",
                        "conversations": convs,
                    })

            elif msg.get("type") == "rename_conversation":
                target_id = msg.get("conversationId", "")
                new_name = msg.get("name", "").strip()
                if target_id and new_name and target_id in workflow_engine.conversations:
                    workflow_engine.rename_conversation(target_id, new_name)
                    await ws.send_json({
                        "type": "conversation_renamed",
                        "conversationId": target_id,
                        "name": new_name,
                    })

            elif msg.get("type") == "branch_conversation":
                source_id = msg.get("conversationId", "")
                message_index = msg.get("messageIndex", -1)
                new_id = workflow_engine.branch_conversation(
                    source_id, message_index,
                    name=msg.get("name", "").strip(),
                )
                if new_id:
                    # 推送新对话到列表
                    convs = []
                    for cid, c in workflow_engine.conversations.items():
                        convs.append({
                            "id": cid, "name": c.name,
                            "messageCount": len(c.messages),
                            "paperCount": len(c.open_papers),
                            "topic": c.current_topic,
                            "isActive": cid == new_id,
                            "updatedAt": c.updated_at,
                        })
                    await ws.send_json({
                        "type": "conversation_list",
                        "conversations": convs,
                    })
                    await ws.send_json({
                        "type": "conversation_branched",
                        "conversationId": new_id,
                        "sourceConversationId": source_id,
                        "name": workflow_engine.conversations[new_id].name,
                    })
                else:
                    await ws.send_json({
                        "type": "status", "level": "error",
                        "code": "branch_failed",
                        "message": "无法创建分支对话",
                    })

            elif msg.get("type") == "translation_request":
                if handle_translation_request is not None:
                    asyncio.ensure_future(handle_translation_request(ws, msg))
                else:
                    await ws.send_json({"type": "status", "level": "error", "code": "translation_error", "message": "翻译功能不可用（缺少 PyMuPDF）"})

            elif msg.get("type") == "cancel_translation":
                if handle_cancel_translation is not None:
                    await handle_cancel_translation(ws, msg)

            elif msg.get("type") == "compute_file_identity":
                _fp = msg.get("filePath", "")
                if not _fp or not os.path.isfile(_fp):
                    await ws.send_json({
                        "type": "file_identity_result",
                        "filePath": _fp, "error": "文件不存在",
                    })
                else:
                    try:
                        _ident = vector_store._compute_file_identity(Path(_fp))
                        await ws.send_json({
                            "type": "file_identity_result",
                            "filePath": _fp,
                            "sha256": _ident["sha256"],
                            "size": _ident["size"],
                        })
                    except Exception as e:
                        await ws.send_json({
                            "type": "file_identity_result",
                            "filePath": _fp, "error": str(e),
                        })

            elif msg.get("type") == "llm_test":
                await handle_llm_test(ws, msg)

            elif msg.get("type") == "llm_list_models":
                await handle_llm_list_models(ws, msg)

            elif msg.get("type") == "model_config":
                await handle_model_config(ws, msg)

            elif msg.get("type") == "control":
                action = msg.get("action", "")
                if action == "cancel_query":
                    ws_id = id(ws)
                    task = _active_queries.pop(ws_id, None)
                    if task and not task.done():
                        task.cancel()
                        await ws.send_json({
                            "type": "status", "level": "info",
                            "code": "cancelled", "message": "查询已取消",
                        })
                        await ws.send_json({
                            "type": "workflow_status",
                            "code": "cancelled",
                            "message": "查询已取消",
                            "timestamp": int(time.time() * 1000),
                        })
                    else:
                        await ws.send_json({
                            "type": "status", "level": "info",
                            "code": "cancelled", "message": "无运行中的查询",
                        })

            # Voice control handlers removed

            elif msg.get("type") == "import_vector_store":
                await handle_import_vector_store(ws, msg)

            elif msg.get("type") == "build_index":
                await handle_build_index(ws, msg)

            elif msg.get("type") == "build_control":
                await handle_build_control(ws, msg)

            elif msg.get("type") == "import_paper":
                await handle_import_paper(ws, msg)

            elif msg.get("type") == "bind_paper":
                _fp = msg.get("filePath", "")
                _pid = msg.get("paperId", "")
                _title = msg.get("title", "")
                if _fp:
                    try:
                        _p = Path(_fp)
                        if _p.exists():
                            workflow_engine.add_paper_to_conv(
                                pid=str(_pid or _p.stem),
                                title=str(_title or _p.name),
                                filename=_p.name,
                                filepath=str(_p),
                            )
                    except Exception:
                        pass
                # Refresh conversation list for all clients
                await _broadcast_conv_list()

            elif msg.get("type") == "unbind_paper":
                _remove_pid = msg.get("paperId", "")
                if _remove_pid and workflow_engine.conv:
                    workflow_engine.conv.open_papers = [
                        p for p in workflow_engine.conv.open_papers
                        if str(p.get("paper_id", "")) != str(_remove_pid)
                    ]
                    await _broadcast_conv_list()

            elif msg.get("type") == "add_papers":
                await handle_add_papers(ws, msg)

            elif msg.get("type") == "list_library":
                await handle_list_library(ws, msg)

            elif msg.get("type") == "clear_vector_store":
                await handle_clear_vector_store(ws, msg)

            elif msg.get("type") == "remove_paper_vectors":
                await handle_remove_paper_vectors(ws, msg)

            elif msg.get("type") == "list_indexed_papers":
                await handle_list_indexed_papers(ws, msg)

            elif msg.get("type") == "delete_library_papers":
                await handle_delete_library_papers(ws, msg)

            elif msg.get("type") == "set_paper_context":
                # L2 论文全文上下文 — 供"总结全文"等全文档查询使用
                paper_text = msg.get("text", "")
                paper_title = msg.get("title", "")
                if paper_text and paper_title:
                    conv_id = msg.get("conversationId", "")
                    workflow_engine.set_l2_context(conv_id, paper_text)
                    if config.DEBUG:
                        print(f"  [L2] set context: {paper_title[:40]} ({len(paper_text)} chars)", flush=True)

            elif msg.get("type") == "delete_message":
                conv_id = msg.get("conversationId", "")
                msg_idx = msg.get("messageIndex", -1)
                conv = workflow_engine.conversations.get(conv_id)
                if conv and 0 <= msg_idx < len(conv.messages):
                    deleted = conv.messages.pop(msg_idx)
                    # Also remove from msg_store
                    store = workflow_engine._message_stores.get(conv_id)
                    if store:
                        store.messages = [
                            m for m in store.messages
                            if not (m.get("role") == deleted.get("role")
                                    and m.get("content") == deleted.get("content")
                                    and m.get("timestamp") == deleted.get("timestamp"))
                        ]
                    await _broadcast_conv_list()
                    await ws.send_json({
                        "type": "message_deleted",
                        "conversationId": conv_id,
                        "messageIndex": msg_idx,
                    })
                else:
                    await ws.send_json({
                        "type": "status", "level": "error",
                        "code": "delete_failed", "message": "无法删除消息",
                    })

            elif msg.get("type") == "edit_message":
                conv_id = msg.get("conversationId", "")
                msg_idx = msg.get("messageIndex", -1)
                new_content = msg.get("content", "")
                conv = workflow_engine.conversations.get(conv_id)
                if conv and 0 <= msg_idx < len(conv.messages) and new_content.strip():
                    conv.messages[msg_idx]["content"] = new_content
                    # 删除该消息之后的所有消息（编辑后的上下文已失效）
                    conv.messages = conv.messages[:msg_idx + 1]
                    await _broadcast_conv_list()
                    await ws.send_json({
                        "type": "conversation_switched",
                        "conversationId": conv_id,
                        "messages": conv.messages,
                        "openPapers": conv.open_papers,
                        "currentTopic": conv.current_topic,
                    })
                else:
                    await ws.send_json({
                        "type": "status", "level": "error",
                        "code": "edit_failed", "message": "无法编辑消息",
                    })

            elif msg.get("type") == "swap_model":
                await handle_swap_model(ws, msg)

            elif msg.get("type") == "regenerate_last":
                await handle_regenerate_last(ws, msg)

    except WebSocketDisconnect:
        pass
    finally:
        ws_id = id(ws)
        # Cancel any running query for this WS to prevent zombie THINKING state
        task = _active_queries.pop(ws_id, None)
        if task and not task.done():
            task.cancel()
            # Also signal the engine to abort at next checkpoint (conv-scoped)
            conv_id = getattr(task, '_conv_id', None)
            if conv_id:
                evt = workflow_engine._cancel_events.get(conv_id)
                if evt:
                    evt.set()
        _ws_clients.discard(ws)


# ===== WebUI static mount (在所有 API 路由之后) =====
if _STATIC_DIR.exists() and not _STATIC_MOUNTED:
    from fastapi.responses import FileResponse, HTMLResponse

    # index.html 不加 no-cache 会导致浏览器缓存旧版本，刷新后仍加载旧的 JS bundle
    _NO_CACHE = {"Cache-Control": "no-cache, no-store, must-revalidate"}

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_webui(full_path: str):
        """SPA fallback: serve built frontend for non-API routes."""
        if not full_path or full_path == "":
            index = _STATIC_DIR / "index.html"
            if index.exists():
                return FileResponse(str(index), headers=_NO_CACHE)
            return HTMLResponse("知伴 WebUI not built", status_code=404)

        file_path = _STATIC_DIR / full_path
        if file_path.exists() and file_path.is_file():
            # JS/CSS 有 content hash 文件名，可长期缓存；其他文件 no-cache
            ext = file_path.suffix.lower()
            cc = "public, max-age=31536000, immutable" if ext in (".js", ".css", ".woff", ".woff2", ".ttf") else "no-cache"
            return FileResponse(str(file_path), headers={"Cache-Control": cc})

        # SPA fallback: all unknown paths → index.html
        index = _STATIC_DIR / "index.html"
        if index.exists():
            return FileResponse(str(index), headers=_NO_CACHE)

        return HTMLResponse("Not Found", status_code=404)

    print(f"   ✅ WebUI mode: serving static files from {_STATIC_DIR}")
    _STATIC_MOUNTED = True


def _cleanup_models():
    """SIGTERM 时卸载模型，释放显存/内存。"""
    try:
        from .rag.embeddings import embedding_engine
        if embedding_engine.model is not None:
            embedding_engine.unload()
            print("  [shutdown] Embedding model unloaded")
    except Exception as e:
        print(f"  [shutdown] Embedding unload failed: {e}")
    try:
        from .engine.llm_utils import unload_local_engine
        unload_local_engine()
        print("  [shutdown] Local LLM unloaded")
    except Exception as e:
        print(f"  [shutdown] Local LLM unload failed: {e}")


def _handle_shutdown(signum, frame):
    print(f"\n  [shutdown] Received signal {signum}, unloading models...")
    _cleanup_models()
    import sys
    sys.exit(0)


def _force_free_port(port: int) -> None:
    """启动前强制释放端口，杀死旧进程。"""
    import subprocess
    import time
    try:
        out = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=5,
        )
        pids = [p.strip() for p in out.stdout.strip().split() if p.strip()]
        if not pids:
            return
        for pid in pids:
            try:
                os.kill(int(pid), signal.SIGKILL)
                print(f"  Killed stale PID {pid} on port {port}")
            except (OSError, ValueError):
                pass
        time.sleep(1.5)  # Wait for OS to release the socket
    except Exception as e:
        print(f"  Port cleanup: {e}")


def main():
    # 注册信号处理，Ctrl+C 或 SIGTERM 时卸载模型
    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    _force_free_port(config.WS_PORT)

    # 二次确认端口未被占用（防止 Electron 双 spawn 竞态）
    import socket as _socket
    _test = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    _test.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    try:
        _test.bind((config.WS_HOST, config.WS_PORT))
        _test.close()
    except OSError:
        print(f"   ⚠️ 端口 {config.WS_PORT} 已被占用（已有 sidecar 实例在运行），退出")
        import sys as _sys
        _sys.exit(0)  # 退出码 0，阻止 Electron 重启循环

    print(f"🀄 知伴 Sidecar starting on ws://{config.WS_HOST}:{config.WS_PORT}")
    use_reload = os.environ.get("SIDECAR_NO_RELOAD", "0") != "1"
    uvicorn.run(
        "sidecar.server:app",
        host=config.WS_HOST,
        port=config.WS_PORT,
        reload=use_reload,
        reload_dirs=[str(Path(__file__).resolve().parent)] if use_reload else None,
        log_level="info",
        ws_ping_interval=30.0,     # 30s 发一次 WS ping
        ws_ping_timeout=120.0,     # 120s 无 pong 才断开（默认 20s，长推理会被误杀）
        timeout_keep_alive=120,    # HTTP keep-alive 超时
    )


if __name__ == "__main__":
    main()
