"""知伴 WebUI 启动入口 — 使用轻量级 lifespan，适合浏览器使用"""
import os
import sys
import threading
from pathlib import Path

threading.stack_size(64 * 1024 * 1024)  # 64MB for llama.cpp inference threads

# 添加项目根目录到 Python 路径
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from contextlib import asynccontextmanager


def _load_embedding_background():
    """后台加载 embedding 模型，不影响 /ready 响应"""
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_load_embedding_async())
    except Exception:
        pass
    finally:
        loop.close()


async def _load_embedding_async():
    import asyncio as _asyncio
    from sidecar.rag.embeddings import embedding_engine
    try:
        await _asyncio.to_thread(embedding_engine.load)
        print(f"   ✅ Embedding 就绪: {embedding_engine.model_name}, dim={embedding_engine.dim}")
    except Exception as e:
        print(f"   Embedding: unavailable ({e})")


# 快速 lifespan：不加载 embedding / LLM 模型，仅启动必要服务。STT 已移除。
@asynccontextmanager
async def fast_lifespan(app):
    print("[WebUI] Fast lifespan starting...")

    from sidecar import config
    from sidecar.server import load_sidecar_settings
    load_sidecar_settings()

    # 配置模型管理器（伴读 + 独立翻译模型）
    from sidecar.llm.model_manager import model_manager
    model_manager.configure(
        companion=config.LLM_MODEL_PATH,
        translation=config.TRANSLATION_MODEL_PATH,
    )
    _companion = Path(config.LLM_MODEL_PATH).name if config.LLM_MODEL_PATH else "(未配置)"
    _trans = Path(config.TRANSLATION_MODEL_PATH).name if config.TRANSLATION_MODEL_PATH else "(未配置,复用伴读)"
    print(f"   Models: companion={_companion[:40]}, translation={_trans[:40]}")

    # 知识图谱（可缺失）
    from sidecar.rag.graph_store import graph_store
    try:
        graph_store.load()
        print(f"[WebUI] Graph: {graph_store.paper_count} papers, {graph_store.edge_count} edges")
    except Exception as e:
        print(f"[WebUI] Graph: unavailable ({e})")

    # 嵌入模型：后台加载，不阻塞 /ready。
    # embedding 引擎自带懒加载，首次 embed/embed_query 时自动加载。
    # 后台预加载只是预热，即使失败也不影响后续懒加载。
    import threading as _thr
    _thr.Thread(target=_load_embedding_background, daemon=True, name="embedding-loader").start()

    # 会话持久化
    from sidecar.persistence import ConversationStore
    db_dir = config.CONVERSATIONS_DB.parent
    db_dir.mkdir(parents=True, exist_ok=True)
    persistence = ConversationStore(str(config.CONVERSATIONS_DB))
    await persistence.initialize()

    from sidecar.engine import engine as wf
    wf.set_persistence(persistence)

    from sidecar.llm.session_state import ChatState
    async def _noop_state_change(_state, _payload):
        return
    wf.session_state.set_on_state_change(_noop_state_change)

    # 初始化默认会话
    try:
        convs, active_id = await persistence.load_all()
        if convs:
            wf.conversations = convs
            wf.active_conv_id = active_id or next(iter(convs))
        else:
            wf.new_conversation(name="新对话")
    except Exception:
        wf.new_conversation(name="新对话")

    import sidecar.server as _server
    _server._ready = True
    _server._startup_status.update(step="ready", message="就绪", progress=100)
    print("[WebUI] Ready!")
    yield

    # 关闭
    if wf.active_conv_id:
        try:
            await persistence.update_active(wf.active_conv_id)
        except Exception:
            pass
    await persistence.close()
    _server._ready = False


def main():
    import uvicorn
    from sidecar import config as cfg
    from sidecar.server import app

    # 替换 lifespan
    app.router.lifespan_context = fast_lifespan

    # 挂载 Web 前端静态文件（生产模式）
    web_dir = _PROJECT_ROOT / "web"
    if web_dir.is_dir() and (web_dir / "index.html").exists():
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="webui")
        print(f"   WebUI: {web_dir}")

    port = int(os.getenv("WS_PORT", cfg.WS_PORT))
    host = os.getenv("WS_HOST", "127.0.0.1")

    print(f"🀄 知伴 WebUI 启动于 http://{host}:{port}")
    print(f"    WebSocket: ws://{host}:{port}/ws")
    print(f"    按 Ctrl+C 停止")

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
