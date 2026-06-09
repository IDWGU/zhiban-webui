"""压缩历史索引 — 关键词倒排 + embedding 向量检索

检索时直接对比原始对话文本的向量（不是摘要），无信息损失。
关键词倒排查 O(1)，向量检索兜底。

MAX=20条/会话，关闭时持久化到 json。
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

from .kv_cache_config import MAX_ARCHIVED_ENTRIES

# 默认持久化目录：frozen 模式用 macOS 标准路径，dev 模式用项目内路径
if getattr(sys, 'frozen', False):
    DEFAULT_ARCHIVE_DIR = str(Path.home() / "Library" / "Application Support" / "ZhiBan" / "archived")
else:
    DEFAULT_ARCHIVE_DIR = str(Path(__file__).resolve().parent.parent.parent / ".cache" / "archived")


def _extract_keywords(text: str, max_kw: int = 8) -> list[str]:
    """简单关键词提取（无外部依赖）。

    提取中文词汇（2-4字）、英文单词（3+字母）、字母数字组合（如 Co3O4）。
    """
    keywords: list[str] = []

    # 中文词汇: 2-4 个连续汉字
    cn_pattern = re.compile(r'[一-鿿]{2,4}')
    cn_matches = cn_pattern.findall(text)
    keywords.extend(cn_matches)

    # 英文单词: 3+ 字母
    en_pattern = re.compile(r'[a-zA-Z]{3,}')
    en_matches = en_pattern.findall(text)
    keywords.extend([w.lower() for w in en_matches])

    # 字母数字组合: 化学式、术语编号等 (如 Co3O4, TiO2, Paper#3)
    alnum_pattern = re.compile(r'[a-zA-Z]+\d+[a-zA-Z\d]*')
    alnum_matches = alnum_pattern.findall(text)
    keywords.extend([w.lower() for w in alnum_matches])

    # 去重 + 限制数量
    seen: set[str] = set()
    result: list[str] = []
    for kw in keywords:
        if kw.lower() not in seen:
            seen.add(kw.lower())
            result.append(kw)
        if len(result) >= max_kw:
            break

    return result


class ArchivedIndex:
    """压缩历史的检索索引。

    存储结构:
      by_id: {archive_id: {text, keywords, embedding, created_at}}
      inverted: {keyword: [archive_id, ...]}

    检索流程:
      1. 关键词匹配 inverted index → O(1)
      2. 关键词未命中时走 embedding 向量检索 → cosine similarity
      3. 返回原始对话文本（不是摘要）
    """

    def __init__(self, embedding_dim: int | None = None):
        self._by_id: dict[str, dict[str, Any]] = {}
        self._inverted: dict[str, list[str]] = {}
        self._embedding_dim: int | None = embedding_dim  # None = auto-detect on first add
        self._next_id = 0

    # ── 属性 ──

    @property
    def entry_count(self) -> int:
        return len(self._by_id)

    @property
    def embedding_dim(self) -> int | None:
        """当前检测到的 embedding 维度（None 表示尚未有 embedding 被添加）"""
        return self._embedding_dim

    @property
    def is_full(self) -> bool:
        return self.entry_count >= MAX_ARCHIVED_ENTRIES

    # ── 添加 ──

    def add(
        self,
        text: str,
        embedding: list[float] | None = None,
        keywords: list[str] | None = None,
        conv_id: str = "",
    ) -> str | None:
        """添加一条压缩记录。

        Args:
          text: 原始对话文本（user + assistant 轮次）
          embedding: 向量，None 时跳过向量索引。首次传入时自动检测维度。
          keywords: 关键词列表，None 时自动提取
          conv_id: 所属会话 ID（用于删除时清理）

        Returns: archive_id，若已满返回 None
        """
        if self.is_full:
            return None

        # Auto-detect embedding dimension on first embedding
        if embedding is not None and self._embedding_dim is None:
            self._embedding_dim = len(embedding)

        # Validate dimension consistency
        if embedding is not None and self._embedding_dim is not None:
            if len(embedding) != self._embedding_dim:
                import logging
                logging.getLogger("zhiban.archived_index").warning(
                    "Embedding dimension mismatch: got %d, expected %d. Skipping vector storage.",
                    len(embedding), self._embedding_dim,
                )
                embedding = None  # Skip vector storage but still store text + keywords
        if self.is_full:
            return None

        if keywords is None:
            keywords = _extract_keywords(text)

        archive_id = f"archived_{self._next_id}"
        self._next_id += 1

        import time
        self._by_id[archive_id] = {
            "text": text,
            "keywords": keywords,
            "embedding": embedding,
            "created_at": time.time(),
            "conv_id": conv_id,
        }

        # 更新倒排索引
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in self._inverted:
                self._inverted[kw_lower] = []
            if archive_id not in self._inverted[kw_lower]:
                self._inverted[kw_lower].append(archive_id)

        return archive_id

    def remove_by_conv_id(self, conv_id: str) -> int:
        """删除指定会话的所有归档条目。返回删除数量。"""
        if not conv_id:
            return 0
        to_remove = [
            aid for aid, entry in self._by_id.items()
            if entry.get("conv_id") == conv_id
        ]
        for aid in to_remove:
            entry = self._by_id.pop(aid, None)
            if entry:
                for kw in entry.get("keywords", []):
                    kw_lower = kw.lower()
                    if kw_lower in self._inverted:
                        self._inverted[kw_lower] = [
                            x for x in self._inverted[kw_lower] if x != aid
                        ]
        return len(to_remove)

    # ── 检索 ──

    def search(
        self,
        query_text: str,
        query_vec: list[float] | None = None,
    ) -> list[dict]:
        """检索匹配的压缩记录。

        1. 关键词匹配 inverted index
        2. 关键词未命中时走 embedding 向量检索（cosine similarity）
        3. 返回原始对话文本

        Returns: [{archive_id, text, keywords, score}, ...]
        """
        # Step 1: 关键词匹配
        query_kw = _extract_keywords(query_text, max_kw=5)
        matched_ids: set[str] = set()
        for kw in query_kw:
            kw_lower = kw.lower()
            if kw_lower in self._inverted:
                matched_ids.update(self._inverted[kw_lower])

        if matched_ids:
            results = []
            for aid in matched_ids:
                entry = self._by_id.get(aid)
                if entry:
                    results.append({
                        "archive_id": aid,
                        "text": entry["text"],
                        "keywords": entry["keywords"],
                        "score": 1.0,  # 关键词精确匹配
                    })
            return results

        # Step 2: 向量检索
        if query_vec is not None and self._by_id:
            scored: list[tuple[float, str]] = []
            for aid, entry in self._by_id.items():
                emb = entry.get("embedding")
                if emb is None:
                    continue
                sim = self._cosine_similarity(query_vec, emb)
                if sim > 0.55:  # 阈值（低于摘要的 0.75，原文匹配更宽松）
                    scored.append((sim, aid))

            scored.sort(key=lambda x: x[0], reverse=True)
            results = []
            for sim, aid in scored[:5]:
                entry = self._by_id[aid]
                results.append({
                    "archive_id": aid,
                    "text": entry["text"],
                    "keywords": entry["keywords"],
                    "score": round(sim, 4),
                })
            return results

        return []

    # ── 向量相似度 ──

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ── 持久化 ──

    def persist(self, conv_id: str, directory: str = "") -> None:
        """持久化到 JSON 文件"""
        dir_path = Path(directory or DEFAULT_ARCHIVE_DIR)
        dir_path.mkdir(parents=True, exist_ok=True)

        data = {
            "conv_id": conv_id,
            "entries": {
                aid: {
                    "text": entry["text"],
                    "keywords": entry["keywords"],
                    # embedding 可能很大，可选保存
                    "embedding": entry.get("embedding"),
                    "created_at": entry["created_at"],
                }
                for aid, entry in self._by_id.items()
            },
            "inverted": self._inverted,
            "next_id": self._next_id,
        }

        filepath = dir_path / f"{conv_id}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, conv_id: str, directory: str = "") -> bool:
        """从 JSON 文件加载。返回是否成功。"""
        filepath = Path(directory or DEFAULT_ARCHIVE_DIR) / f"{conv_id}.json"
        if not filepath.exists():
            return False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._by_id = {}
            self._inverted = {}
            for aid, entry in data.get("entries", {}).items():
                self._by_id[aid] = {
                    "text": entry["text"],
                    "keywords": entry["keywords"],
                    "embedding": entry.get("embedding"),
                    "created_at": entry.get("created_at", 0),
                }
                for kw in entry["keywords"]:
                    kw_lower = kw.lower()
                    if kw_lower not in self._inverted:
                        self._inverted[kw_lower] = []
                    if aid not in self._inverted[kw_lower]:
                        self._inverted[kw_lower].append(aid)

            self._next_id = data.get("next_id", len(self._by_id))
            return True
        except (json.JSONDecodeError, KeyError, OSError):
            return False

    # ── 清空 ──

    def clear(self) -> None:
        self._by_id.clear()
        self._inverted.clear()
        self._next_id = 0

    def remove_oldest(self) -> str | None:
        """移除最旧的条目，为新条目腾出空间。返回被移除的 archive_id。"""
        if not self._by_id:
            return None

        oldest_id = min(
            self._by_id.keys(),
            key=lambda aid: self._by_id[aid].get("created_at", 0),
        )
        entry = self._by_id.pop(oldest_id)

        # 清理倒排索引
        for kw in entry.get("keywords", []):
            kw_lower = kw.lower()
            if kw_lower in self._inverted:
                self._inverted[kw_lower] = [
                    aid for aid in self._inverted[kw_lower]
                    if aid != oldest_id
                ]
                if not self._inverted[kw_lower]:
                    del self._inverted[kw_lower]

        return oldest_id
