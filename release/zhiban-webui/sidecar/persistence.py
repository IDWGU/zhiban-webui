"""
会话持久化模块 — SQLite 存储
==============================
使用 Python 标准库 sqlite3 + asyncio.to_thread 实现零新依赖的异步持久化。

表结构:
  conversations — 会话元数据
  messages      — 对话历史 (CASCADE 删除)
  open_papers   — 已打开论文 (CASCADE 删除)
"""

import asyncio
import json
import sqlite3
import os
from pathlib import Path
from datetime import datetime

from . import config


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS conversations (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL DEFAULT '新对话',
    current_topic       TEXT NOT NULL DEFAULT '',
    is_first_message    INTEGER NOT NULL DEFAULT 1,
    last_question       TEXT NOT NULL DEFAULT '',
    last_question_time  REAL NOT NULL DEFAULT 0.0,
    classify_fail_count INTEGER NOT NULL DEFAULT 0,
    is_active           INTEGER NOT NULL DEFAULT 0,
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    conv_id    TEXT NOT NULL,
    role       TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content    TEXT NOT NULL,
    seq        INTEGER NOT NULL,
    metadata   TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (conv_id) REFERENCES conversations(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conv_id, seq);

CREATE TABLE IF NOT EXISTS open_papers (
    conv_id   TEXT NOT NULL,
    paper_id  TEXT NOT NULL,
    title     TEXT NOT NULL DEFAULT '',
    filename  TEXT NOT NULL DEFAULT '',
    filepath  TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (conv_id, paper_id),
    FOREIGN KEY (conv_id) REFERENCES conversations(id) ON DELETE CASCADE
);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """自动迁移：为旧数据库补加缺失的列（CREATE TABLE IF NOT EXISTS 不改已有表）。"""
    cur = conn.execute("PRAGMA table_info(open_papers)")
    existing = {row[1] for row in cur.fetchall()}
    if "filepath" not in existing:
        conn.execute("ALTER TABLE open_papers ADD COLUMN filepath TEXT NOT NULL DEFAULT ''")

    cur = conn.execute("PRAGMA table_info(messages)")
    msg_cols = {row[1] for row in cur.fetchall()}
    if "metadata" not in msg_cols:
        conn.execute("ALTER TABLE messages ADD COLUMN metadata TEXT NOT NULL DEFAULT '{}'")


class ConversationStore:
    """SQLite 会话存储 (async-safe, 零新依赖)"""

    def __init__(self, db_path: str = ""):
        self._db_path = db_path or str(config.CONVERSATIONS_DB)

    # ── 生命周期 ──

    async def initialize(self) -> None:
        """创建表结构，启用 WAL 模式和外键"""
        await asyncio.to_thread(self._init_sync)

    def _init_sync(self):
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.executescript(SCHEMA_SQL)
            # 自动迁移：为旧表补加缺失的列
            _migrate(conn)
            conn.commit()
        finally:
            conn.close()

    async def close(self) -> None:
        """关闭 (当前无持久连接，预留接口)"""
        pass

    # ── 读取 ──

    async def load_all(self) -> tuple[dict, str | None]:
        """加载所有会话 → (conversations_dict, active_conv_id)"""
        return await asyncio.to_thread(self._load_all_sync)

    def _load_all_sync(self) -> tuple[dict, str | None]:
        from .engine import Conversation

        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            # 加载会话列表
            rows = conn.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC"
            ).fetchall()

            convs: dict[str, Conversation] = {}
            active_id: str | None = None

            for row in rows:
                cid = row["id"]
                conv = Conversation(
                    id=cid,
                    name=row["name"],
                    current_topic=row["current_topic"],
                    is_first_message=bool(row["is_first_message"]),
                    last_question=row["last_question"],
                    last_question_time=row["last_question_time"],
                    classify_fail_count=row["classify_fail_count"],
                    updated_at=row["updated_at"] or "",
                )

                # 加载消息
                msg_rows = conn.execute(
                    "SELECT role, content, metadata FROM messages WHERE conv_id=? ORDER BY seq",
                    (cid,)
                ).fetchall()
                conv.messages = []
                for r in msg_rows:
                    msg = {"role": r["role"], "content": r["content"]}
                    try:
                        meta = json.loads(r["metadata"] or "{}")
                        if isinstance(meta, dict):
                            msg.update(meta)
                    except (json.JSONDecodeError, TypeError):
                        pass
                    conv.messages.append(msg)

                # 加载论文
                paper_rows = conn.execute(
                    "SELECT paper_id, title, filename, filepath FROM open_papers WHERE conv_id=?",
                    (cid,)
                ).fetchall()
                conv.open_papers = [
                    {"paper_id": r["paper_id"], "title": r["title"], "filename": r["filename"], "filepath": r["filepath"]}
                    for r in paper_rows
                ]

                convs[cid] = conv
                if row["is_active"]:
                    active_id = cid

            return convs, active_id
        finally:
            conn.close()

    # ── 写入 ──

    async def save_full_conversation(self, conv) -> None:
        """原子保存：upsert conv + 替换 messages + 替换 papers"""
        await asyncio.to_thread(self._save_full_sync, conv)

    def _save_full_sync(self, conv) -> None:
        now = datetime.now().isoformat()
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("BEGIN")

            # Upsert conversation
            conn.execute("""
                INSERT INTO conversations
                    (id, name, current_topic, is_first_message,
                     last_question, last_question_time, classify_fail_count,
                     updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    current_topic=excluded.current_topic,
                    is_first_message=excluded.is_first_message,
                    last_question=excluded.last_question,
                    last_question_time=excluded.last_question_time,
                    classify_fail_count=excluded.classify_fail_count,
                    updated_at=excluded.updated_at
            """, (
                conv.id, conv.name, conv.current_topic,
                int(conv.is_first_message),
                conv.last_question, conv.last_question_time,
                conv.classify_fail_count, now,
            ))

            # Replace messages
            conn.execute("DELETE FROM messages WHERE conv_id=?", (conv.id,))
            for i, m in enumerate(conv.messages):
                meta = {}
                for k in ("timestamp", "mode", "model", "usage", "loopDetected",
                          "screenContext", "subtype"):
                    if k in m and m[k] is not None:
                        meta[k] = m[k]
                conn.execute(
                    "INSERT INTO messages (conv_id, role, content, seq, metadata) VALUES (?, ?, ?, ?, ?)",
                    (conv.id, m["role"], m["content"], i, json.dumps(meta, ensure_ascii=False)),
                )

            # Replace open_papers
            conn.execute("DELETE FROM open_papers WHERE conv_id=?", (conv.id,))
            for p in conv.open_papers:
                conn.execute(
                    "INSERT OR REPLACE INTO open_papers (conv_id, paper_id, title, filename, filepath) VALUES (?, ?, ?, ?, ?)",
                    (conv.id, str(p["paper_id"]), p.get("title", ""), p.get("filename", ""), p.get("filepath", "")),
                )

            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    async def update_active(self, conv_id: str) -> None:
        """标记活跃会话"""
        await asyncio.to_thread(self._update_active_sync, conv_id)

    def _update_active_sync(self, conv_id: str) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("UPDATE conversations SET is_active=0 WHERE is_active=1")
            conn.execute("UPDATE conversations SET is_active=1, updated_at=? WHERE id=?",
                         (datetime.now().isoformat(), conv_id))
            conn.commit()
        finally:
            conn.close()

    async def delete_conversation(self, conv_id: str) -> None:
        """删除会话 (CASCADE 自动清理关联表)"""
        await asyncio.to_thread(self._delete_sync, conv_id)

    def _delete_sync(self, conv_id: str) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("DELETE FROM conversations WHERE id=?", (conv_id,))
            conn.commit()
        finally:
            conn.close()

    async def rename_conversation(self, conv_id: str, new_name: str) -> None:
        """重命名会话"""
        await asyncio.to_thread(self._rename_sync, conv_id, new_name)

    def _rename_sync(self, conv_id: str, new_name: str) -> None:
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                "UPDATE conversations SET name=?, updated_at=? WHERE id=?",
                (new_name, datetime.now().isoformat(), conv_id),
            )
            conn.commit()
        finally:
            conn.close()
