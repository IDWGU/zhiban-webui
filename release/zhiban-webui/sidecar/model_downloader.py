"""首次启动模型下载器 — 检查并自动下载模型到项目内置 models/bundled/ 目录"""

import os
import logging
import shutil
from pathlib import Path

from . import config

# Known embedding model specs — matched by substring of EMBEDDING_MODEL
_EMBEDDING_SPECS = {
    "bge-m3": {
        "name": "BGE-M3 (Embedding)",
        "repo": "BAAI/bge-m3",
        "allow_patterns": ["*.json", "*.txt", "*.md", "*.safetensors", "1_Pooling/*"],
        "size_gb": 2.2,
    },
    "jina-embeddings-v5-text-nano": {
        "name": "Jina-v5-Nano (Embedding)",
        "repo": "jinaai/jina-embeddings-v5-text-nano",
        "allow_patterns": None,
        "size_gb": 0.5,
    },
    "KaLM-Embedding": {
        "name": "KaLM-V2.5 (Embedding)",
        "repo": "KaLM-Embedding-V2.5",
        "allow_patterns": None,
        "size_gb": 0.8,
    },
}


def _get_required_models() -> list[dict]:
    """Build the required-models list based on the currently configured EMBEDDING_MODEL."""
    model_key = config.EMBEDDING_MODEL.lower()
    for key, spec in _EMBEDDING_SPECS.items():
        if key.lower() in model_key:
            return [spec]
    # Unknown model — skip pre-download (will be downloaded by embedding_engine.load)
    return []

def _is_model_cached(repo_id: str, model_dir: Path) -> bool:
    """Check if a HuggingFace model is already cached locally."""
    repo_dir = model_dir / ("models--" + repo_id.replace("/", "--"))
    if not repo_dir.exists():
        return False
    return any(repo_dir.rglob("*.safetensors")) or any(repo_dir.rglob("*.bin"))


async def _download_model(model: dict, model_dir: Path, progress_callback=None):
    """Download a HuggingFace model using snapshot_download (non-blocking)."""
    import asyncio

    repo_id = model["repo"]
    timeout = int(os.getenv("MODEL_DOWNLOAD_TIMEOUT", "600"))

    print(f"   Downloading {model['name']} ({model.get('size_gb', '?')}GB)...")

    if progress_callback:
        await progress_callback({
            "type": "status", "level": "info",
            "code": "model_downloading",
            "message": f"正在下载 {model['name']}，约 {model.get('size_gb', '?')}GB，请耐心等待...",
        })

    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: __import__('huggingface_hub').snapshot_download(
                repo_id=repo_id,
                cache_dir=str(model_dir),
                allow_patterns=model.get("allow_patterns"),
                resume_download=True,
                max_workers=4,
            )
        )
        print(f"   {model['name']} 下载完成")
    except Exception as e:
        print(f"   {model['name']} 下载失败: {e}")
        raise


async def ensure_models(progress_callback=None) -> bool:
    """检查模型缓存状态。内置捆绑版跳过下载，纯本地运行。"""
    model_dir = config.MODEL_CACHE
    all_ok = True

    # Check HuggingFace models
    for model in _get_required_models():
        repo_id = model["repo"]

        if _is_model_cached(repo_id, model_dir):
            logging.info(f"Model cached: {model['name']} (bundled)")
            if progress_callback:
                await progress_callback({
                    "type": "status", "level": "info",
                    "code": "model_cached",
                    "message": f"{model['name']} 已内置",
                })
            continue

        # 内置捆绑已提供，跳过网络下载；如果缺失由 SentenceTransformer 自动处理
        logging.warning(f"Model not bundled: {model['name']} — will load lazily if needed")
        if progress_callback:
            await progress_callback({
                "type": "status", "level": "warn",
                "code": "model_missing",
                "message": f"{model['name']} 未捆绑，首次使用将自动下载",
            })
        all_ok = False

    return all_ok


def clean_old_sidecar_dist():
    """清理旧的 sidecar-dist 中的 models"""
    old_models = config._EXE_DIR / "models"
    if old_models.exists() and old_models != config.MODEL_CACHE:
        shutil.rmtree(old_models, ignore_errors=True)
