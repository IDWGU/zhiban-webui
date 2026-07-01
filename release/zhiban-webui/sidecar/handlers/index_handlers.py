"""Index/build/import WebSocket message handlers."""
import asyncio
import shutil
import threading
import time
from pathlib import Path

from fastapi import WebSocket

from .. import config
from ..rag.vector_store import vector_store

# 构建控制信号：支持暂停/恢复/取消
_build_cancel_event: threading.Event | None = None
_build_pause_event: threading.Event | None = None
_build_lock = threading.Lock()


async def handle_build_control(ws: WebSocket, msg: dict):
    """Handle build control: pause / resume / cancel."""
    global _build_cancel_event, _build_pause_event

    action = msg.get("action", "")
    if action == "cancel":
        if _build_cancel_event:
            _build_cancel_event.set()
            await ws.send_json({
                "type": "status", "level": "info",
                "code": "build_cancelling", "message": "正在取消构建...",
            })
        else:
            await ws.send_json({
                "type": "status", "level": "warn",
                "code": "build_not_running", "message": "没有运行中的构建任务",
            })
    elif action == "pause":
        if _build_pause_event and not _build_pause_event.is_set():
            _build_pause_event.set()
            await ws.send_json({
                "type": "build_index_progress",
                "phase": "paused",
                "current": 0, "total": 0,
                "message": "构建已暂停",
            })
        else:
            await ws.send_json({
                "type": "status", "level": "warn",
                "code": "build_not_running", "message": "没有运行中的构建任务",
            })
    elif action == "resume":
        if _build_pause_event and _build_pause_event.is_set():
            _build_pause_event.clear()
            await ws.send_json({
                "type": "status", "level": "info",
                "code": "build_resumed", "message": "构建已恢复",
            })
        else:
            await ws.send_json({
                "type": "status", "level": "warn",
                "code": "build_not_paused", "message": "没有暂停的构建任务",
            })
    else:
        await ws.send_json({
            "type": "status", "level": "error",
            "code": "unknown_action", "message": f"未知控制动作: {action}",
        })


async def handle_import_vector_store(ws: WebSocket, msg: dict):
    """Import external ChromaDB vector store directory."""
    source_path = msg.get("sourcePath", "")
    if not source_path:
        await ws.send_json({
            "type": "import_vector_result",
            "success": False,
            "error": "No directory path specified",
        })
        return

    src = Path(source_path)
    if not src.is_dir():
        await ws.send_json({
            "type": "import_vector_result",
            "success": False,
            "error": f"Directory does not exist: {source_path}",
        })
        return

    if not (src / "chroma.sqlite3").exists():
        await ws.send_json({
            "type": "import_vector_result",
            "success": False,
            "error": "Selected directory is not a valid ChromaDB directory (chroma.sqlite3 not found)",
        })
        return

    await ws.send_json({
        "type": "status", "level": "info",
        "code": "importing", "message": "Importing vector store...",
    })

    try:
        # Backup the current DB before overwriting
        dest = config.CHROMA_PERSIST_DIR
        if dest.exists():
            backup = dest.with_name(dest.name + ".bak")
            if backup.exists():
                shutil.rmtree(backup)
            shutil.copytree(str(dest), str(backup))
            print(f"  [import] backed up current DB to {backup}")

        # Remove existing and copy the imported one
        if dest.exists():
            shutil.rmtree(str(dest))
        shutil.copytree(str(src), str(dest))
        print(f"  [import] copied {src} → {dest}")

        # Reinitialize the vector store client & collection
        vector_store.client = None
        vector_store.collection = None
        vector_store._indexed = False

        # Verify the imported collection actually has data
        client = vector_store._get_client()
        try:
            collections = client.list_collections()
            target_name = "paper_chunks"
            found = [c for c in collections if c.name == target_name]
            if not found:
                raise ValueError(f"Collection '{target_name}' not found in imported vector store")
            chunks = found[0].count()
        except Exception as e:
            vector_store.client = None
            # Restore backup
            dest = config.CHROMA_PERSIST_DIR
            backup = dest.with_name(dest.name + ".bak")
            if backup.exists():
                if dest.exists():
                    shutil.rmtree(str(dest))
                shutil.copytree(str(backup), str(dest))
                print(f"  [import] restored backup due to: {e}")
            raise RuntimeError(f"Imported data is invalid: {e}")

        await ws.send_json({
            "type": "import_vector_result",
            "success": True,
            "chunks": chunks,
        })
        print(f"  [import] done, {chunks} chunks loaded")
    except Exception as e:
        print(f"  [import] failed: {e}")
        await ws.send_json({
            "type": "import_vector_result",
            "success": False,
            "error": f"Import failed: {e}",
        })


async def handle_build_index(ws: WebSocket, msg: dict):
    """Handle vector store build request — calls vector_store.build_index() with progress reporting."""
    global _build_cancel_event, _build_pause_event

    source_path = msg.get("sourcePath")
    force = msg.get("force", True)

    # 重置控制信号
    with _build_lock:
        _build_cancel_event = threading.Event()
        _build_pause_event = threading.Event()

    await ws.send_json({
        "type": "status", "level": "info",
        "code": "building", "message": "Building vector store (this may take a few minutes)...",
    })

    start = time.time()
    loop = asyncio.get_running_loop()

    # 节流：每秒最多推一次进度，避免 2000+ 条 WebSocket 消息积压
    _throttle = {"last": 0.0}

    def on_progress(phase: str, current: int, total: int, message: str):
        now = time.time()
        if phase != "done" and now - _throttle["last"] < 5.0:
            return
        _throttle["last"] = now
        asyncio.run_coroutine_threadsafe(
            ws.send_json({
                "type": "build_index_progress",
                "phase": phase,
                "current": current,
                "total": total,
                "message": message,
            }),
            loop,
        )

    try:
        src_dir = Path(source_path) if source_path else config.PAPER_LIBRARY
        await loop.run_in_executor(
            None,
            lambda: vector_store.build_index(
                force=force,
                source_dir=src_dir,
                progress_callback=on_progress,
                cancel_event=_build_cancel_event,
                pause_event=_build_pause_event,
            ),
        )
        elapsed = round(time.time() - start, 1)
        await ws.send_json({
            "type": "build_index_result",
            "success": True,
            "chunks": vector_store.chunk_count,
            "duration": elapsed,
        })
        print(f"  [build_index] done: {vector_store.chunk_count} chunks in {elapsed}s")
    except Exception as e:
        print(f"  [build_index] failed: {e}")
        await ws.send_json({
            "type": "build_index_result",
            "success": False,
            "error": str(e),
        })
    finally:
        # 清理控制信号
        with _build_lock:
            _build_cancel_event = None
            _build_pause_event = None


async def handle_import_paper(ws: WebSocket, msg: dict):
    """Import a single paper PDF: parse→chunk→embed→write to ChromaDB, with real-time progress."""
    file_path = msg.get("filePath", "")
    if not file_path:
        await ws.send_json({
            "type": "import_paper_result",
            "success": False,
            "error": "No file path provided",
        })
        return

    src = Path(file_path)
    if not src.exists():
        await ws.send_json({
            "type": "import_paper_result",
            "success": False,
            "error": f"File does not exist: {file_path}",
        })
        return

    ext = src.suffix.lower()
    if ext not in (".pdf", ".docx", ".txt", ".md"):
        await ws.send_json({
            "type": "import_paper_result",
            "success": False,
            "error": f"Unsupported file type: {ext}",
        })
        return

    # Check if already imported
    doc_id = vector_store._extract_doc_id(src.stem)
    existing = vector_store.search_by_doc_ids(doc_id, [doc_id], top_k=1)
    already_imported = bool(existing)

    # Progress callback
    def on_progress(phase: str, message: str, progress: float = None):
        asyncio.ensure_future(ws.send_json({
            "type": "import_paper_progress",
            "phase": phase,
            "message": message,
            "progress": progress,
            "doc_id": doc_id,
            "filename": src.name,
        }))

    # If already imported, remove old chunks first (default: overwrite)
    if already_imported:
        await ws.send_json({
            "type": "import_paper_exists",
            "doc_id": doc_id,
            "filename": src.name,
            "chunks": len(existing),
        })
        collection = vector_store._get_collection()
        old_ids = [r.get("id", "") for r in existing if r.get("id")]
        if old_ids:
            collection.delete(ids=old_ids)
        on_progress("removing", f"Removing {len(old_ids)} existing chunks...")

    # Copy to pending directory
    pending_dir = config.CHROMA_PERSIST_DIR.parent / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(str(src), str(pending_dir / src.name))
    except Exception as e:
        print(f"  [import_paper] copy warning: {e}")

    # Ensure ChromaDB client + collection initialized in main thread
    # (avoids "no event loop in thread" when executor thread first touches ChromaDB)
    vector_store._get_collection()

    # Execute import
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: vector_store.import_single_paper(src, progress_callback=on_progress),
        )
        await ws.send_json({
            "type": "import_paper_result",
            **result,
        })
    except Exception as e:
        await ws.send_json({
            "type": "import_paper_result",
            "success": False,
            "error": f"Import failed: {e}",
        })


async def handle_add_papers(ws: WebSocket, msg: dict):
    """Receive paper files from frontend, copy to paper library for permanent storage."""
    files = msg.get("files", [])
    if not files:
        await ws.send_json({
            "type": "add_papers_result",
            "success": False,
            "error": "No file paths provided",
        })
        return

    library_dir = config.PAPER_LIBRARY
    library_dir.mkdir(parents=True, exist_ok=True)

    added = 0
    errors = []
    for f in files:
        src = Path(f)
        if not src.exists():
            errors.append(f"File does not exist: {f}")
            continue
        dst = library_dir / src.name
        # 同名文件自动重命名
        if dst.exists():
            stem = dst.stem
            suffix = dst.suffix
            counter = 1
            while dst.exists():
                dst = library_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        try:
            shutil.copy2(str(src), str(dst))
            added += 1
        except Exception as e:
            errors.append(f"{src.name}: {e}")

    # 返回更新后的论文库列表
    lib_papers = _list_library_papers()
    print(f"  [add_papers] {added} files added to library (total: {len(lib_papers)})")
    result: dict = {
        "type": "add_papers_result",
        "success": added > 0,
        "added": added,
        "library": lib_papers,
    }
    if errors:
        result["error"] = "; ".join(errors)
    await ws.send_json(result)


def _list_library_papers() -> list[dict]:
    """List papers stored in the library directory."""
    library_dir = config.PAPER_LIBRARY
    if not library_dir.exists():
        return []
    papers = []
    for f in sorted(library_dir.iterdir(), key=lambda p: _natural_sort_key(p.name)):
        if f.suffix.lower() in (".pdf", ".docx", ".txt", ".md"):
            papers.append({
                "name": f.name,
                "path": str(f),
                "size": f.stat().st_size,
                "mtime": f.stat().st_mtime,
            })
    return papers


async def handle_list_library(ws: WebSocket, msg: dict):
    """Return the current paper library list."""
    papers = _list_library_papers()
    await ws.send_json({
        "type": "list_library_result",
        "papers": papers,
    })


async def handle_clear_vector_store(ws: WebSocket, msg: dict):
    """清空整个向量库。"""
    try:
        client = vector_store._get_client()
        try:
            client.delete_collection("paper_chunks")
        except Exception:
            pass  # 集合可能不存在
        vector_store.collection = None
        vector_store._indexed = False
        # 重新创建空集合
        vector_store._get_collection()
        await ws.send_json({
            "type": "clear_vector_result",
            "success": True,
            "message": "向量库已清空",
        })
    except Exception as e:
        await ws.send_json({
            "type": "clear_vector_result",
            "success": False,
            "error": str(e),
        })


async def handle_remove_paper_vectors(ws: WebSocket, msg: dict):
    """删除指定论文的所有向量。"""
    doc_ids = msg.get("doc_ids", [])
    if not doc_ids:
        await ws.send_json({
            "type": "remove_paper_result",
            "success": False,
            "error": "未指定论文 ID",
        })
        return

    try:
        collection = vector_store._get_collection()
        removed = 0
        for doc_id in doc_ids:
            # 查询该 doc_id 的所有 chunks
            results = collection.get(
                where={"doc_id": doc_id},
                include=["metadatas"],
            )
            chunk_ids = results.get("ids", [])
            if chunk_ids:
                collection.delete(ids=chunk_ids)
                removed += len(chunk_ids)
                print(f"  [remove] doc_id={doc_id}: {len(chunk_ids)} chunks deleted")

        await ws.send_json({
            "type": "remove_paper_result",
            "success": True,
            "removed": removed,
            "message": f"已删除 {removed} 条向量",
        })
    except Exception as e:
        await ws.send_json({
            "type": "remove_paper_result",
            "success": False,
            "error": str(e),
        })


async def handle_delete_library_papers(ws: WebSocket, msg: dict):
    """删除知识库中的论文文件（同时删除对应向量）。"""
    paths = msg.get("paths", [])
    if not paths:
        await ws.send_json({"type": "delete_library_result", "success": False, "error": "未指定文件"})
        return

    deleted = 0
    errors = []
    for fp in paths:
        try:
            p = Path(fp)
            if not p.exists():
                errors.append(f"{p.name}: 文件不存在")
                continue
            doc_id = vector_store._extract_doc_id(p.stem)
            try:
                collection = vector_store._get_collection()
                results = collection.get(where={"doc_id": doc_id}, include=["metadatas"])
                chunk_ids = results.get("ids", [])
                if chunk_ids:
                    collection.delete(ids=chunk_ids)
            except Exception:
                pass
            p.unlink()
            deleted += 1
        except Exception as e:
            errors.append(f"{Path(fp).name}: {e}")

    await ws.send_json({
        "type": "delete_library_result",
        "success": len(errors) == 0,
        "deleted": deleted,
        "error": "; ".join(errors) if errors else None,
    })


def _natural_sort_key(filename: str) -> tuple:
    """自然排序：首字母 → 数字按数值比较（'3' 排在 '11' 前）。"""
    import re
    parts = re.split(r'(\d+)', filename)
    key = []
    for p in parts:
        key.append(int(p) if p.isdigit() else p.lower())
    return tuple(key)


async def handle_list_indexed_papers(ws: WebSocket, msg: dict):
    """列出向量库中所有已索引的论文（去重 doc_id），自然排序。"""
    try:
        collection = vector_store._get_collection()
        if collection.count() == 0:
            await ws.send_json({
                "type": "indexed_papers_result",
                "papers": [],
            })
            return

        results = collection.get(include=["metadatas"])
        metas = results.get("metadatas", [])

        paper_map: dict[str, dict] = {}
        for meta in metas:
            doc_id = meta.get("doc_id", "unknown")
            if doc_id not in paper_map:
                paper_map[doc_id] = {
                    "doc_id": doc_id,
                    "filename": meta.get("filename", ""),
                    "source": meta.get("source", ""),
                    "chunks": 0,
                }
            paper_map[doc_id]["chunks"] += 1

        papers = sorted(paper_map.values(), key=lambda x: _natural_sort_key(x["filename"]))
        await ws.send_json({
            "type": "indexed_papers_result",
            "papers": papers,
        })
    except Exception as e:
        await ws.send_json({
            "type": "indexed_papers_result",
            "success": False,
            "error": str(e),
        })
