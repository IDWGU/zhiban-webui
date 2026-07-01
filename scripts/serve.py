#!/usr/bin/env python3
"""知伴 WebUI 生产服务器入口

设置环境变量、自动发现模型，然后启动 WebUI 服务。
"""

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# === 镜像配置 ===
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# === 自动发现 LLM 模型 ===
if not os.getenv("LLM_MODEL_PATH"):
    llm_dir = _PROJECT_ROOT / "models" / "llm"
    if llm_dir.is_dir():
        ggufs = sorted(
            [f for f in llm_dir.glob("*.gguf") if not f.name.startswith("mmproj")],
            key=lambda f: f.stat().st_size,
            reverse=True,
        )
        if ggufs:
            os.environ["LLM_MODEL_PATH"] = str(ggufs[0])
            print(f"   LLM: {ggufs[0].name} ({ggufs[0].stat().st_size // 1024 // 1024}MB)")

# === 自动发现翻译模型 ===
if not os.getenv("TRANSLATION_MODEL_PATH"):
    trans_dir = _PROJECT_ROOT / "models" / "translation"
    if trans_dir.is_dir():
        ggufs = sorted(
            [f for f in trans_dir.glob("*.gguf") if not f.name.startswith("mmproj")],
            key=lambda f: f.stat().st_size,
            reverse=True,
        )
        if ggufs:
            os.environ["TRANSLATION_MODEL_PATH"] = str(ggufs[0])
            print(f"   翻译: {ggufs[0].name} ({ggufs[0].stat().st_size // 1024 // 1024}MB)")

# === 用户自定义模型目录 ===
user_model_dir = os.getenv("ZHIBAN_MODEL_DIR", "")
if user_model_dir:
    user_path = Path(user_model_dir)
    if user_path.is_dir():
        # 扫描用户目录中的 GGUF 文件
        user_ggufs = sorted(
            [f for f in user_path.glob("*.gguf") if not f.name.startswith("mmproj")],
            key=lambda f: f.stat().st_size,
            reverse=True,
        )
        if user_ggufs and not os.getenv("LLM_MODEL_PATH"):
            os.environ["LLM_MODEL_PATH"] = str(user_ggufs[0])
            print(f"   LLM (自定义): {user_ggufs[0].name}")

if __name__ == "__main__":
    from sidecar.webui_launcher import main
    main()
