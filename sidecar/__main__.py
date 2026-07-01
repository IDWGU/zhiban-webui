"""PyInstaller entry point — 启动知伴 Sidecar 服务器"""
import sys
import os
import threading
from pathlib import Path

threading.stack_size(64 * 1024 * 1024)  # 64MB for llama.cpp inference threads

# In PyInstaller bundle, the executable location determines base paths
if getattr(sys, 'frozen', False):
    _exe_dir = Path(sys.executable).parent  # sidecar-dist/
    # Models: bundled read-only alongside binary
    os.environ.setdefault("MODEL_CACHE", str(_exe_dir / "models"))
    # ChromaDB: MUST use writable user directory (app bundle is read-only)
    _data_dir = Path.home() / "Library" / "Application Support" / "ZhiBan"
    os.environ.setdefault("CHROMA_DIR", str(_data_dir / "chroma"))

from sidecar.server import main

if __name__ == "__main__":
    main()
