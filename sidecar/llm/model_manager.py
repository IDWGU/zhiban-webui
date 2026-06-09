"""模型生命周期管理器 — 翻译独立模型切换 + 内存压力感知

翻译和伴读共享统一内存（Apple Silicon）或显存，切换时需：
1. 等待伴读查询完成
2. 检测内存压力，必要时卸载 non-essential 模型
3. 加载翻译模型，执行翻译，卸载翻译模型，恢复伴读模型
"""

import asyncio
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("zhiban.model_manager")


class ModelManager:
    """管理伴读模型和翻译模型之间的切换"""

    def __init__(self):
        self._companion_path: str = ""
        self._translation_path: str = ""
        self._companion_loaded: bool = False
        self._translation_loaded: bool = False
        self._kv_savepoint: int = 0

    # ── 配置 ──

    def configure(self, companion: str = "", translation: str = "") -> None:
        if companion:
            self._companion_path = companion
        if translation:
            self._translation_path = translation
        logger.info("ModelManager configured: companion=%s translation=%s",
                    self._companion_path[:60] if self._companion_path else "(none)",
                    self._translation_path[:60] if self._translation_path else "(none)")

    @property
    def has_translation_model(self) -> bool:
        return bool(self._translation_path) and Path(self._translation_path).exists()

    @property
    def _is_low_memory(self) -> bool:
        """低内存机型 (≤16GB): psutil 总内存判断，不受当前可用内存波动影响。

        check_memory_pressure() 使用动态 available GB，16GB 机型上可能误判为
        "normal" 跳过卸载，导致翻译加载时 OOM。
        低内存机型强制互斥——翻译时无条件卸载伴读+embedding。
        """
        try:
            import psutil
            return psutil.virtual_memory().total <= 16 * 1024**3
        except Exception:
            return False

    # ── 内存压力检测 ──

    @staticmethod
    def check_memory_pressure() -> str:
        """macOS 统一内存压力级别: normal | warning | critical

        psutil 为主判断（available GB 可靠），memory_pressure 为辅。
        当 available < 2GB 时必然触发 critical，确保大模型加载前先释放。
        """
        # psutil: available GB 是硬指标（不含可回收缓存）
        try:
            import psutil
            mem = psutil.virtual_memory()
            avail_gb = mem.available / (1024**3)
            if avail_gb < 2.0:
                return "critical"
            if avail_gb < 5.0:
                return "warning"
        except Exception:
            avail_gb = -1

        # memory_pressure: macOS 系统级压力（含文件缓存视角，偏宽松）
        try:
            result = subprocess.run(
                ["memory_pressure"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.splitlines():
                if "System-wide memory free percentage" in line:
                    try:
                        pct = int(line.split(":")[-1].strip().rstrip("%"))
                        if pct < 5:
                            return "critical"
                        if pct < 15:
                            return "warning" if avail_gb > 2 else "critical"
                        return "normal" if avail_gb > 5 else "warning"
                    except ValueError:
                        pass
        except Exception:
            pass

        # final fallback
        if avail_gb < 0:
            return "normal"
        if avail_gb < 2.0:
            return "critical"
        if avail_gb < 5.0:
            return "warning"
        return "normal"

    # ── 等待伴读空闲 ──

    @staticmethod
    async def wait_for_idle(timeout: float = 30.0) -> bool:
        """等待 AI 伴读查询完成，超时返回 False"""
        from ..engine import engine as wf_engine
        deadline = time.time() + timeout
        while time.time() < deadline:
            state = wf_engine.session_state
            if state is None or state.is_idle:
                return True
            # 检查翻译取消事件
            from ..translation.handler import _translation_cancel
            if _translation_cancel and _translation_cancel.is_set():
                return False
            await asyncio.sleep(0.5)
        logger.warning("wait_for_idle: timeout after %.1fs", timeout)
        return False

    # ── 释放内存 ──

    def free_memory(self, level: str) -> None:
        """按优先级卸载模型释放统一内存

        normal:  不卸载
        warning: 卸载 embedding（释放 1-2GB）+ 卸载伴读（释放 5-15GB）
        critical: 卸载一切，保存 KV cache
        """
        if level == "normal":
            return

        # warning/critical 都卸载 embedding（轻量，先释放）
        self._unload_embedding()

        # warning 就卸载伴读：翻译模型只需要少量内存（~2-4GB），
        # 但伴读模型可能占用 5-15GB，不卸载会导致 OOM
        self._unload_companion(save_kv=(level == "critical"))

    def _unload_embedding(self) -> None:
        """卸载向量嵌入模型（release ~1-2GB）"""
        try:
            from ..rag.embeddings import embedding_engine
            if embedding_engine.is_available:
                logger.info("Free memory: unloading embedding model")
                embedding_engine.unload()
        except Exception as e:
            logger.warning("Failed to unload embedding: %s", e)

    def _unload_companion(self, save_kv: bool = True) -> None:
        """卸载伴读模型，可选保存 KV cache 状态"""
        try:
            from ..engine.llm_utils import get_local_engine
            engine = get_local_engine()
            if engine is not None and engine.is_loaded:
                if save_kv and engine.supports_kv_cache_ops:
                    self._kv_savepoint = engine.n_tokens
                    logger.info("Free memory: savepoint=%d, unloading companion model", self._kv_savepoint)
                else:
                    logger.info("Free memory: unloading companion model (no KV save)")
                engine.unload()
                self._companion_loaded = False
        except Exception as e:
            logger.warning("Failed to unload companion: %s", e)

    # ── 切换到翻译模型 ──

    async def switch_to_translation(self) -> bool:
        """切换到翻译模型，返回是否成功"""
        if not self.has_translation_model:
            logger.info("No translation model configured, using companion model")
            from ..engine.llm_utils import get_local_engine
            engine = get_local_engine()
            return engine is not None and engine.is_loaded

        # 1. 等待伴读空闲
        idled = await self.wait_for_idle(timeout=30.0)
        if not idled:
            logger.warning("switch_to_translation: companion still busy after timeout")

        # 2. 检测内存
        pressure = self.check_memory_pressure()
        logger.info("Memory pressure: %s (low_memory=%s)", pressure, self._is_low_memory)

        # 3. 释放内存
        # 低内存机型 (≤16GB): 强制互斥，无条件卸载 embedding + 伴读
        if self._is_low_memory:
            logger.info("Low memory machine: forcing unconditional unload of embedding + companion")
            self._unload_embedding()
            self._unload_companion(save_kv=False)
        else:
            self.free_memory(pressure)

        # 4. 加载前二次确认内存（free_memory 后应有足够空间）
        # 低内存机型已卸载全量，跳过二次检查
        if not self._is_low_memory:
            pressure2 = self.check_memory_pressure()
            if pressure2 == "critical":
                logger.warning("Memory still critical after free_memory, forcing companion unload")
                self._unload_companion(save_kv=False)

        # 5. 加载翻译模型
        try:
            from ..engine.llm_utils import load_local_engine, get_local_engine
            logger.info("Loading translation model: %s", self._translation_path[:60])
            status = await asyncio.to_thread(load_local_engine, self._translation_path)
            self._translation_loaded = True
            self._companion_loaded = False
            logger.info("Translation model ready: %s", status)
            return True
        except Exception as e:
            logger.error("Failed to load translation model: %s", e)
            return False

    # ── 恢复到伴读模型 ──

    async def switch_to_companion(self) -> bool:
        """卸载翻译模型，恢复伴读模型和 embedding"""
        # 卸载翻译模型
        try:
            from ..engine.llm_utils import get_local_engine, unload_local_engine
            engine = get_local_engine()
            if engine is not None and engine.is_loaded:
                engine.unload()
                self._translation_loaded = False
                logger.info("Translation model unloaded")
        except Exception as e:
            logger.warning("Failed to unload translation: %s", e)

        # 如果没有独立的翻译模型（和伴读共享），不用恢复
        if not self._companion_path:
            self._companion_loaded = True
            return True

        # 恢复伴读模型
        try:
            from ..engine.llm_utils import load_local_engine, get_local_engine
            logger.info("Restoring companion model: %s", self._companion_path[:60])
            status = await asyncio.to_thread(load_local_engine, self._companion_path)
            self._companion_loaded = True
            logger.info("Companion model restored: %s", status)
        except Exception as e:
            logger.error("Failed to restore companion model: %s", e)
            return False

        # 恢复 embedding（如果被卸载了）
        try:
            from ..rag.embeddings import embedding_engine
            if not embedding_engine.is_available:
                logger.info("Reloading embedding model...")
                await asyncio.to_thread(embedding_engine.load)
        except Exception as e:
            logger.warning("Failed to reload embedding: %s", e)

        return True


# 全局单例
model_manager = ModelManager()
