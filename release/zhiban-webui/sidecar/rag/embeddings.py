"""Embedding 引擎 — jina-embeddings-v5-text-nano (GPU 优先, OOM 回退 CPU)

模型选型:
  - 默认: jina-embeddings-v5-text-nano (768 维, ~0.5GB, Apple Silicon 8GB 友好)
  - 备选: BAAI/bge-m3 (1024 维, ~2.2GB, 中文检索更强, 环境变量切换)
  - 备选: KaLM-Embedding-V2.5 (768 维, ~0.8GB)

GPU 加速:
  - NVIDIA: CUDA + Flash Attention 2
  - Apple Silicon: MPS (Metal Performance Shaders)
  - AMD: ROCm (PyTorch ROCm)
  - OOM → 自动回退 CPU

兼容性:
  jina-embeddings-v5-text-nano 使用 trust_remote_code + PeftMixedModel，
  transformers v5.x 需通过 _apply_jina_patch() 修复 config_class/cache_dir.
  tokenizer.json 需确保非 HTML 错误页（CDN 过期链接会导致下载损坏）.
"""

import os

# HuggingFace 国内镜像列表（按优先级排序）
HF_MIRRORS = [
    "https://hf-mirror.com",
    "https://huggingface.modelscope.cn",
    "https://mirrors.tuna.tsinghua.edu.cn/huggingface",
    "https://ai.gitcode.com/models",
    "https://aliendao.cn",
]

# 首次镜像设为默认值（可被环境变量覆盖）
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = HF_MIRRORS[0]

import gc
import logging
import platform
import numpy as np

from collections.abc import Callable

from .. import config

logger = logging.getLogger("zhiban.embeddings")


def _detect_device() -> str:
    """选择最优设备。

    jina PEFT 在 GPU 上长期批量推理存在内存碎片化问题（利用率随时间退化），
    默认统一走 CPU。如需 GPU 加速设 USE_GPU=1。
    """
    if os.getenv("USE_GPU", "0") != "1":
        return "cpu"

    try:
        import torch

        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if vram > 8:
                return "cuda"

        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            if platform.system() == "Darwin":
                return "mps"
    except ImportError:
        pass
    return "cpu"


class EmbeddingEngine:
    """文本向量化引擎 — 首次使用时才加载模型"""

    def __init__(self):
        self.model = None
        self._load_error: str | None = None
        self._model_name: str = ""
        self._device: str = "cpu"
        self._progress_callback: Callable | None = None

    def set_progress_callback(self, cb: Callable | None) -> None:
        """设置进度回调，接收 (percent: int, message: str)"""
        self._progress_callback = cb

    def _report_progress(self, pct: int, msg: str) -> None:
        if self._progress_callback:
            try:
                self._progress_callback(pct, msg)
            except Exception:
                pass

    @property
    def model_name(self) -> str:
        """当前加载的模型名称"""
        return self._model_name or config.EMBEDDING_MODEL

    # jina-embeddings-v5 依赖的 EuroBERT 基类文件。
    # 这些文件属于 jina 模型的 HF repo，但 snapshot 下载时可能漏掉。
    # 缺失时 trust_remote_code 会触发联网下载，离线环境下直接失败。
    _JINA_REQUIRED_FILES = [
        "configuration_jina_embeddings_v5.py",
        "modeling_jina_embeddings_v5.py",
        "custom_st.py",
        "configuration_eurobert.py",
        "modeling_eurobert.py",
    ]

    def _verify_jina_snapshot(self, snap_dir: "Path") -> list[str]:
        """检查 jina 模型快照中是否包含所有必需文件，返回缺失列表。"""
        missing = []
        for fname in self._JINA_REQUIRED_FILES:
            fpath = snap_dir / fname
            if not fpath.exists() or (fpath.is_symlink() and not fpath.resolve().exists()):
                missing.append(fname)
        return missing

    def _apply_jina_patch(self, model_path: str) -> None:
        """在 transformers 模块缓存中预创建文件，修复 jina 的兼容性问题。

        在 transformers 加载自定义代码之前，把源文件复制到模块缓存目录，
        并注入 config_class 补丁（PeftMixedModel 在 v5.x 中的缺失）。

        同时将 EuroBERT 相对导入改为绝对导入，避免 trust_remote_code
        在离线模式下因找不到 configuration_eurobert.py 而失败。
        """
        try:
            from pathlib import Path
            import shutil

            src_dir = Path(model_path)
            if not src_dir.exists():
                return

            py_files = {f.name for f in src_dir.glob("*.py") if f.is_file()}
            if not py_files:
                return

            # transformers 模块缓存路径:
            # ~/.cache/huggingface/modules/transformers_modules/{org}/{hyphenated_model}/{snapshot_hash}/
            hub_modules = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
            model_fs_name = "jinaai/jina_hyphen_embeddings_hyphen_v5_hyphen_text_hyphen_nano"
            mod_dir = hub_modules / model_fs_name / src_dir.name
            mod_dir.mkdir(parents=True, exist_ok=True)

            # 复制所有 .py 文件
            for py_file in py_files:
                src = src_dir / py_file
                dst = mod_dir / py_file
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
                    logger.debug("Copied %s -> cache", py_file)

            # 补丁 0: 修复 EuroBERT 相对导入 → 绝对导入
            # jina-embeddings-v5 的 .py 文件通过相对导入引用 EuroBERT 基类，
            # 但 configuration_eurobert.py / modeling_eurobert.py 不在模型快照中。
            # 改为从 transformers 内置模块导入，避免 trust_remote_code 加载时报 ImportError。
            for fname, old_import, new_import in [
                (
                    "configuration_jina_embeddings_v5.py",
                    "from .configuration_eurobert import EuroBertConfig",
                    "from transformers.models.eurobert.configuration_eurobert import EuroBertConfig",
                ),
                (
                    "modeling_jina_embeddings_v5.py",
                    "from .modeling_eurobert import EuroBertModel",
                    "from transformers.models.eurobert.modeling_eurobert import EuroBertModel",
                ),
            ]:
                target = mod_dir / fname
                if target.exists():
                    content = target.read_text(encoding="utf-8")
                    if old_import in content:
                        content = content.replace(old_import, new_import)
                        target.write_text(content, encoding="utf-8")
                        logger.info("Patched EuroBERT import in %s", fname)

            # 注入补丁
            modeling_file = mod_dir / "modeling_jina_embeddings_v5.py"
            if modeling_file.exists():
                content = modeling_file.read_text(encoding="utf-8")

                # 补丁 1: config_class (PeftMixedModel v5.x 兼容)
                if "config_class" not in content:
                    patch = (
                        '\n# Patched by zhiban (transformers v5.x compat)\n'
                        'JinaEmbeddingsV5Model.config_class = JinaEmbeddingsV5Config\n'
                    )
                    content += patch

                # 补丁 2: 给所有 from_pretrained 和 snapshot_download 加上 cache_dir
                # jina 的 from_pretrained 不传 **kwargs，导致 cache_dir 丢失
                patches = {
                    "base_model = EuroBertModel.from_pretrained(\n            pretrained_model_name_or_path,\n            config=base_config,\n            dtype=kwargs.pop(\"dtype\", torch.float32),\n        )": "base_model = EuroBertModel.from_pretrained(\n            pretrained_model_name_or_path,\n            config=base_config,\n            dtype=kwargs.pop(\"dtype\", torch.float32),\n            **kwargs,\n        )",
                }
                for old_text, new_text in patches.items():
                    if old_text in content:
                        content = content.replace(old_text, new_text)
                        logger.info("Patched: %s", old_text[:50])

                # 补丁 3: snapshot_download(repo_id=base_model.name_or_path, allow_patterns=["adapters/*"])
                # 需要加上 cache_dir
                old_adapter = 'adapter_cache_path = snapshot_download(\n                repo_id=base_model.name_or_path,\n                allow_patterns=["adapters/*"],\n            )'
                new_adapter = 'adapter_cache_path = snapshot_download(\n                repo_id=base_model.name_or_path,\n                allow_patterns=["adapters/*"],\n                cache_dir=kwargs.get("cache_dir", None),\n            )'
                if old_adapter in content:
                    content = content.replace(old_adapter, new_adapter)
                    logger.info("Patched adapter snapshot_download with cache_dir")

                modeling_file.write_text(content, encoding="utf-8")

            # 确保 __init__.py 存在
            init_file = mod_dir / "__init__.py"
            if not init_file.exists():
                init_file.write_text("")

        except Exception as e:
            logger.warning("Module cache patch failed (non-fatal): %s", e)

    def reload(self, model_name: str | None = None) -> None:
        """重新加载嵌入模型（运行时切换模型）。

        自动尝试多个国内镜像，直到加载成功或全部失败。

        Args:
          model_name: HuggingFace 模型 ID，None 时使用 config.EMBEDDING_MODEL
        """
        self.unload()
        self._load_error = None
        if model_name:
            self._model_name = model_name
        else:
            self._model_name = config.EMBEDDING_MODEL

        self._report_progress(5, "准备下载...")

        last_error = ""
        for mirror_idx, mirror in enumerate(HF_MIRRORS):
            os.environ["HF_ENDPOINT"] = mirror
            try:
                import huggingface_hub.constants as _hc
                _hc.HF_ENDPOINT = mirror
            except ImportError:
                pass
            logger.info("Trying HF mirror: %s", mirror)
            self._report_progress(10, f"尝试镜像 {mirror_idx+1}/{len(HF_MIRRORS)}...")

            # 使用 hf_hub_download 带进度回调下载模型快照
            try:
                from huggingface_hub import snapshot_download

                self._report_progress(15, f"正在连接 {mirror}...")
                snapshot_download(
                    repo_id=self._model_name,
                    cache_dir=str(config.MODEL_CACHE),
                    local_files_only=False,
                    tqdm_class=None,
                )
                self._report_progress(60, "下载完成，正在加载模型...")

                # 从本地缓存加载
                self._load_error = None
                self.load()
                if self.model is not None:
                    self._report_progress(100, "加载完成")
                    logger.info("Model loaded via %s", mirror)
                    return
            except Exception as e:
                last_error = str(e)
                logger.warning("Mirror %s failed: %s", mirror, e)
                self._load_error = None
                self.model = None
                continue

            # 备用: 直接 load
            try:
                self._load_error = None
                self.load()
                if self.model is not None:
                    self._report_progress(100, "加载完成")
                    logger.info("Model loaded via %s (direct)", mirror)
                    return
            except Exception as e:
                last_error = str(e)
                logger.warning("Mirror %s direct load failed: %s", mirror, e)
                self._load_error = None
                self.model = None

        self._report_progress(0, f"加载失败: {last_error}")
        raise RuntimeError(f"所有镜像都无法加载模型 {self._model_name}: {last_error}")

    def unload(self) -> None:
        """卸载当前模型，释放显存"""
        self.model = None
        self._device = "cpu"
        self._load_error = None

    def load(self, force_device: str | None = None):
        """加载模型。force_device 可强制指定设备 ('cpu'/'mps'/'cuda')。

        embed_query 强制 CPU 以节省显存；build_index 走自动检测。
        """
        if self.model is not None:
            if force_device and self._device != force_device:
                logger.info("切换设备 %s → %s", self._device, force_device)
                self.unload()
                self._load_error = None
            else:
                return
        if self._load_error:
            return  # Don't retry if already failed
        try:
            from pathlib import Path
            from sentence_transformers import SentenceTransformer

            device = force_device or _detect_device()
            model_to_load = self._model_name or config.EMBEDDING_MODEL

            cache_dir = config.MODEL_CACHE / f"models--{model_to_load.replace('/', '--')}"
            snap_dir = cache_dir / "snapshots"
            local_only = cache_dir.exists()

            # jina 模型: 检查 snapshot 完整性，缺失时尝试联网补全
            if "jina" in model_to_load.lower() and snap_dir.exists():
                snapshots = list(snap_dir.iterdir())
                if snapshots:
                    snap_path = snapshots[0]
                    missing = self._verify_jina_snapshot(snap_path)
                    if missing:
                        logger.warning(
                            "Jina snapshot 缺少文件: %s。尝试从 HF 下载...", ", ".join(missing)
                        )
                        for fname in missing:
                            try:
                                from huggingface_hub import hf_hub_download
                                hf_hub_download(
                                    repo_id=model_to_load,
                                    filename=fname,
                                    cache_dir=str(config.MODEL_CACHE),
                                    local_files_only=False,
                                )
                                logger.info("下载成功: %s", fname)
                            except Exception as e:
                                logger.warning("下载 %s 失败: %s", fname, e)

                        still_missing = self._verify_jina_snapshot(snap_path)
                        if still_missing:
                            self._load_error = (
                                f"Embedding 模型文件不完整，缺少: {', '.join(still_missing)}。"
                                f"请确保模型缓存目录包含所有必需文件，或联网后重启应用自动下载。"
                            )
                            logger.error(self._load_error)
                            raise RuntimeError(self._load_error)

                    self._apply_jina_patch(str(snap_path))

            if device == "cpu":
                logger.info("Loading %s on CPU", model_to_load)
                self._load_model(SentenceTransformer, device, local_only)
            else:
                try:
                    logger.info("Loading %s on %s (GPU)", model_to_load, device.upper())
                    self._load_model(SentenceTransformer, device, local_only)
                except Exception as gpu_err:
                    # OOM or GPU not supported → fallback to CPU
                    if "memory" in str(gpu_err).lower() or "oom" in str(gpu_err).lower() or "MPS" in str(gpu_err):
                        logger.warning(
                            "GPU (%s) failed: %s. Falling back to CPU.", device, gpu_err
                        )
                    else:
                        logger.warning(
                            "GPU (%s) load failed: %s. Falling back to CPU.", device, gpu_err
                        )
                    self._load_model(SentenceTransformer, "cpu", local_only)

        except ImportError as e:
            self._load_error = f"sentence-transformers 未安装: {e}"
            raise
        except Exception as e:
            self._load_error = str(e)
            raise

    def _load_model(self, SentenceTransformer, device: str, local_only: bool, model_name: str | None = None):
        """实际加载模型到指定设备"""
        model_kwargs = {}
        tokenizer_kwargs = {}
        model_name = model_name or self._model_name or config.EMBEDDING_MODEL

        if device == "cuda":
            model_kwargs["torch_dtype"] = "auto"
        elif device == "mps":
            # fp16 在 jina PEFT + MPS 组合下不稳定，保持 fp32
            model_kwargs["torch_dtype"] = "float32"

        # jina 模型需要 default_task 才能在 encode 时正常工作
        if "jina" in model_name.lower():
            model_kwargs["default_task"] = "retrieval"

        # 强制离线模式：trust_remote_code 会尝试联网下载 .py 文件覆盖补丁，
        # HF_HUB_OFFLINE=1 阻止所有网络请求，确保本地缓存和补丁不会被覆盖
        _prev_offline = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            self.model = SentenceTransformer(
                model_name,
                cache_folder=str(config.MODEL_CACHE),
                trust_remote_code=True,
                device=device,
                model_kwargs=model_kwargs if model_kwargs else None,
                tokenizer_kwargs=tokenizer_kwargs if tokenizer_kwargs else None,
                local_files_only=local_only,
            )
        finally:
            if _prev_offline is None:
                del os.environ["HF_HUB_OFFLINE"]
            else:
                os.environ["HF_HUB_OFFLINE"] = _prev_offline

        self._device = device
        logger.info("Embedding model loaded on %s, dim=%d", device, self.dim)

    @property
    def _is_jina(self) -> bool:
        return "jina" in (self._model_name or config.EMBEDDING_MODEL).lower()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """批量文本转向量 — 自动选择 GPU/CPU"""
        self.load()
        encode_kwargs = dict(normalize_embeddings=True, show_progress_bar=False)
        if self._is_jina:
            encode_kwargs["prompt_name"] = "document"
        embeddings = self.model.encode(texts, **encode_kwargs)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """单条查询转向量 — 强制 CPU，省显存且更稳定"""
        self.load(force_device="cpu")
        encode_kwargs = dict(normalize_embeddings=True, show_progress_bar=False)
        if self._is_jina:
            encode_kwargs["prompt_name"] = "query"
        embeddings = self.model.encode([query], **encode_kwargs)
        return embeddings[0].tolist()

    @property
    def is_available(self) -> bool:
        # 只检查 model 是否已加载，不依赖 sentence_transformers 是否可导入。
        # unload() 后 model 为 None，必须返回 False，否则翻译结束后
        # embedding 永远不会被重新加载。
        return self.model is not None

    @property
    def dim(self) -> int:
        self.load()
        d = self.model.get_embedding_dimension()
        if d is not None:
            return d
        # jina 模型的 get_embedding_dimension() 返回 None，从 config 获取
        if hasattr(self.model, '_model_meta') and self.model._model_meta:
            return self.model._model_meta.get("embedding_dim", 768)
        return 768


# Singleton
embedding_engine = EmbeddingEngine()
