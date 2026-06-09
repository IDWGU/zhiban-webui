"""向量存储 — ChromaDB 管理论文笔记和全文的向量索引"""

import gc
import hashlib
import json
import threading
import time
import chromadb
from pathlib import Path
from typing import Callable, Optional

from .. import config
from .embeddings import embedding_engine


# ── 论文身份追踪（防重复向量化）──

def _get_identity_store_path() -> Path:
    """返回论文身份 JSON 文件路径，存在 ChromaDB 目录下。"""
    return config.CHROMA_PERSIST_DIR / "paper_identities.json"


def _load_paper_identities() -> dict:
    """加载已索引论文的身份记录。{sha256: {size, filenames, doc_ids, chunk_count}}"""
    path = _get_identity_store_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_paper_identities(data: dict) -> None:
    """持久化论文身份记录。"""
    config.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    _get_identity_store_path().write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _compute_file_identity(file_path: Path) -> dict:
    """计算文件的身份特征：大小、SHA256。

    对于大文件（>100MB），先用前 64KB + 文件大小做快速指纹，
    再决定是否需要完整 SHA256。
    """
    stat = file_path.stat()
    size = stat.st_size
    sha = hashlib.sha256()

    with open(file_path, "rb") as f:
        if size <= 100 * 1024 * 1024:
            # 小文件：全量 SHA256
            while chunk := f.read(65536):
                sha.update(chunk)
        else:
            # 大文件：前 64KB + 后 64KB + 中间采样
            sha.update(f.read(65536))
            mid = size // 2
            f.seek(mid - 32768)
            sha.update(f.read(65536))
            f.seek(-65536, 2)
            sha.update(f.read(65536))

    return {"size": size, "sha256": sha.hexdigest()}


class VectorStore:
    """ChromaDB 向量存储，索引论文精读笔记和全文"""

    def __init__(self):
        self.client: chromadb.PersistentClient | None = None
        self.collection: chromadb.Collection | None = None
        self._indexed = False

    def _get_client(self) -> chromadb.PersistentClient:
        if self.client is None:
            config.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(config.CHROMA_PERSIST_DIR)
            )
        return self.client

    def _get_collection(self) -> chromadb.Collection:
        if self.collection is None:
            client = self._get_client()
            self.collection = client.get_or_create_collection(
                name="paper_chunks",
                metadata={"hnsw:space": "cosine"},
            )
        return self.collection

    def _check_dimension_mismatch(self) -> bool:
        """检查向量库维度是否与当前 embedding 模型匹配。不匹配返回 True。"""
        try:
            collection = self._get_collection()
            if collection.count() == 0:
                return False
            existing_dim = collection.metadata.get("embedding_dim")
            if existing_dim is None:
                existing_dim = collection.metadata.get("dimension")
            if existing_dim is None:
                return False  # 旧版没有记录维度，无法判断
            existing_dim = int(existing_dim)
            current_dim = embedding_engine.dim
            if existing_dim != current_dim:
                print(f"  [vector_store] 维度不匹配: 索引={existing_dim}, 模型={current_dim}")
                return True
        except Exception:
            pass
        return False

    def build_index(self, force: bool = False, source_dir: Path | None = None,
                    collection_name: str = "paper_chunks",
                    progress_callback: Callable | None = None,
                    cancel_event: threading.Event | None = None,
                    pause_event: threading.Event | None = None):
        """
        构建/重建向量索引。

        source_dir: 自定义文本目录。为 None 时使用默认知识库路径。
        progress_callback: 可选进度回调 (phase, current, total, message)
        支持三种文件类型:
          - .md   → 按 ## 章节切分，章节名作为 section_type
          - .txt  → 按段落切分，文件名 stem 作为 doc_id
          - .pdf/.docx → 自动提取文本

        元数据: {doc_id, source, section_type, chunk_index, filename}
        """
        if not embedding_engine.is_available:
            raise RuntimeError("Embedding 模型不可用，无法构建向量索引")

        collection = self._get_collection()
        existing = collection.count()

        if existing > 0 and not force and source_dir is None:
            self._indexed = True
            return

        # 收集已索引的 doc_id，避免重复向量化
        indexed_doc_ids: set[str] = set()
        if existing > 0 and not force:
            results = collection.get(include=["metadatas"])
            for m in (results.get("metadatas") or []):
                did = m.get("doc_id", "")
                if did:
                    indexed_doc_ids.add(did)

        if force and existing > 0:
            client = self._get_client()
            client.delete_collection(collection_name)
            self.collection = client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine", "embedding_dim": str(embedding_engine.dim)},
            )
            collection = self.collection
            indexed_doc_ids = set()

        embedding_engine.load()

        # Determine source directories
        if source_dir:
            notes_dir = source_dir if any(source_dir.glob("*.md")) else None
            texts_dir = source_dir
        else:
            notes_dir = config.PAPER_READING
            texts_dir = config.PAPER_TEXTS

        ids, documents, metadatas = [], [], []
        doc_counter = {}
        skipped = 0

        def _emit(phase, current, total, message):
            if progress_callback:
                progress_callback(phase, current, total, message)

        # Count total files first
        total_files = 0
        if notes_dir and notes_dir.exists():
            total_files += len([f for f in notes_dir.glob("*.md") if f.name != "00-knowledge-map.md"])
        if source_dir:
            total_files += len(list(source_dir.glob("*.pdf"))) + len(list(source_dir.glob("*.docx")))
        if texts_dir and texts_dir.exists():
            total_files += len(list(texts_dir.glob("*.txt")))
        total_files = max(total_files, 1)

        _emit('scanning', 0, total_files, '正在扫描论文文件...')

        # 快捷取消检查
        def _check_cancel():
            if cancel_event and cancel_event.is_set():
                raise RuntimeError("构建已取消")

        # 1. Index markdown notes
        file_idx = 0
        if notes_dir and notes_dir.exists():
            for note_file in sorted(notes_dir.glob("*.md")):
                _check_cancel()
                if note_file.name == "00-knowledge-map.md":
                    continue
                file_idx += 1
                doc_id = self._extract_doc_id(note_file.stem)
                if doc_id in indexed_doc_ids:
                    skipped += 1
                    continue
                _emit('extracting', file_idx, total_files, f'正在处理 {note_file.name} ({file_idx}/{total_files})')
                doc_counter[note_file.stem] = doc_counter.get(note_file.stem, 0)
                chunks = self._chunk_markdown(note_file.read_text(encoding="utf-8"))
                for section_type, chunk in chunks:
                    if len(chunk.strip()) < 20:
                        continue
                    idx = doc_counter[note_file.stem]
                    doc_counter[note_file.stem] += 1
                    ids.append(f"note_{doc_id}_{idx}")
                    documents.append(chunk)
                    metadatas.append({
                        "doc_id": doc_id,
                        "source": "markdown",
                        "section_type": section_type,
                        "chunk_index": idx,
                        "filename": note_file.name,
                    })

        # 2. Index PDF files (auto-extract text)
        if source_dir:
            for pdf_file in sorted(source_dir.glob("*.pdf")):
                _check_cancel()
                doc_id = self._extract_doc_id(pdf_file.stem)
                if doc_id in indexed_doc_ids:
                    skipped += 1
                    file_idx += 1
                    continue
                file_idx += 1
                _emit('extracting', file_idx, total_files, f'正在处理 {pdf_file.name} ({file_idx}/{total_files})')
                doc_counter[pdf_file.stem] = doc_counter.get(pdf_file.stem, 0)
                try:
                    full_text = self._extract_pdf_text(pdf_file)
                    sections = self._detect_pdf_sections(full_text)
                    chunks = self._chunk_text_with_sections(sections)
                except Exception as e:
                    print(f"  [skip] {pdf_file.name}: {e}")
                    continue
                for chunk in chunks:
                    # chunks 现在是 (section_type, chunk_text) 元组
                    sec_type, chunk_text = chunk if isinstance(chunk, tuple) else ("body", chunk)
                    if len(chunk_text.strip()) < 30:
                        continue
                    idx = doc_counter[pdf_file.stem]
                    doc_counter[pdf_file.stem] += 1
                    ids.append(f"pdf_{doc_id}_{idx}")
                    documents.append(chunk_text)
                    metadatas.append({
                        "doc_id": doc_id,
                        "source": "pdf",
                        "section_type": sec_type,
                        "chunk_index": idx,
                        "filename": pdf_file.name,
                    })

                # 生成论文结构概要 chunk — 每章取前 120 字，嵌入后天然匹配全篇关键词
                outline = self._build_paper_outline_chunk(sections)
                if outline:
                    idx = doc_counter[pdf_file.stem]
                    doc_counter[pdf_file.stem] += 1
                    ids.append(f"pdf_{doc_id}_{idx}")
                    documents.append(outline)
                    metadatas.append({
                        "doc_id": doc_id,
                        "source": "pdf",
                        "section_type": "paper_outline",
                        "chunk_index": idx,
                        "filename": pdf_file.name,
                    })

        # 3. Index DOCX files (auto-extract text)
        if source_dir:
            for docx_file in sorted(source_dir.glob("*.docx")):
                _check_cancel()
                doc_id = self._extract_doc_id(docx_file.stem)
                if doc_id in indexed_doc_ids:
                    skipped += 1
                    file_idx += 1
                    continue
                file_idx += 1
                _emit('extracting', file_idx, total_files, f'正在处理 {docx_file.name} ({file_idx}/{total_files})')
                doc_counter[docx_file.stem] = doc_counter.get(docx_file.stem, 0)
                try:
                    full_text = self._extract_docx_text(docx_file)
                    chunks = self._chunk_text(full_text)
                except Exception as e:
                    print(f"  [skip] {docx_file.name}: {e}")
                    continue
                for chunk in chunks:
                    if len(chunk.strip()) < 30:
                        continue
                    idx = doc_counter[docx_file.stem]
                    doc_counter[docx_file.stem] += 1
                    ids.append(f"docx_{doc_id}_{idx}")
                    documents.append(chunk)
                    metadatas.append({
                        "doc_id": doc_id,
                        "source": "docx",
                        "section_type": "paragraph",
                        "chunk_index": idx,
                        "filename": docx_file.name,
                    })

        # 4. Index plain text files (generic — works for any .txt)
        if texts_dir and texts_dir.exists():
            for txt_file in sorted(texts_dir.glob("*.txt")):
                _check_cancel()
                doc_id = self._extract_doc_id(txt_file.stem)
                if doc_id in indexed_doc_ids:
                    skipped += 1
                    file_idx += 1
                    continue
                file_idx += 1
                _emit('extracting', file_idx, total_files, f'正在处理 {txt_file.name} ({file_idx}/{total_files})')
                doc_counter[txt_file.stem] = doc_counter.get(txt_file.stem, 0)
                content = txt_file.read_text(encoding="utf-8")
                chunks = self._chunk_text(content)
                for chunk in chunks:
                    if len(chunk.strip()) < 30:
                        continue
                    idx = doc_counter[txt_file.stem]
                    doc_counter[txt_file.stem] += 1
                    ids.append(f"text_{doc_id}_{idx}")
                    documents.append(chunk)
                    metadatas.append({
                        "doc_id": doc_id,
                        "source": "text",
                        "section_type": "paragraph",
                        "chunk_index": idx,
                        "filename": txt_file.name,
                    })

        if not documents:
            msg = f'所有文件已索引，跳过 {skipped} 篇' if skipped > 0 else '未发现可索引的文件'
            _emit('done', 0, 0, msg)
            return

        # Batch embed and insert
        batch_size = 4
        total_chunks = len(documents)
        total_batches = (total_chunks + batch_size - 1) // batch_size
        embedded = 0
        for i in range(0, total_chunks, batch_size):
            batch_i = i // batch_size + 1

            # 暂停检查
            if pause_event:
                while pause_event.is_set():
                    if cancel_event and cancel_event.is_set():
                        raise RuntimeError("构建已取消")
                    _emit('embedding', embedded, total_chunks, f'已暂停 ({embedded}/{total_chunks})')
                    time.sleep(0.5)
            # 取消检查
            if cancel_event and cancel_event.is_set():
                raise RuntimeError("构建已取消")

            batch_docs = documents[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            embeddings = embedding_engine.embed(batch_docs)
            collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta,
                embeddings=embeddings,
            )
            embedded += len(batch_docs)
            _emit('embedding', embedded, total_chunks, f'嵌入向量 ({embedded}/{total_chunks})')

        self._indexed = True
        done_msg = f'向量库构建完成（新增 {len(documents)} chunks' + (f'，跳过 {skipped} 篇已索引' if skipped > 0 else '') + '）'
        _emit('done', total_files, total_files, done_msg)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """向量检索"""
        self._ensure_indexed()
        collection = self._get_collection()
        query_embedding = embedding_engine.embed_query(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                hits.append({
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1.0 - results["distances"][0][i],  # cosine → similarity
                })
        return hits

    def search_by_doc_ids(
        self, query: str, doc_ids: list[str], top_k: int = 10
    ) -> list[dict]:
        """在指定文档集合中检索"""
        self._ensure_indexed()
        collection = self._get_collection()
        query_embedding = embedding_engine.embed_query(query)

        # ChromaDB where clause: doc_id in list
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 3,
            where={"doc_id": {"$in": doc_ids}},
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                hits.append({
                    "id": doc_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1.0 - results["distances"][0][i],
                })
        hits.sort(key=lambda x: x["score"], reverse=True)
        return hits[:top_k]

    def _ensure_indexed(self):
        if not self._indexed:
            try:
                collection = self._get_collection()
                if collection.count() > 0:
                    if self._check_dimension_mismatch():
                        raise RuntimeError(
                            f"向量库维度与当前 embedding 模型不匹配。"
                            f"请在设置 → 数据库 中点击「构建向量库」重建索引。"
                        )
                    self._indexed = True
                    return
            except RuntimeError:
                raise
            except Exception:
                pass
            self.build_index()

    # ===== 分块工具 =====

    @staticmethod
    def _extract_doc_id(filename_stem: str) -> str:
        """
        从文件名提取文档 ID。
        短数字前缀（≤3位）后有实质内容 → 用完整 stem 避免碰撞
        （例如 "1-s2.0-S1369..." 和 "1-s2.0-S2211..." 都是 doc_id "1" 会互相覆盖）。
        纯数字或长数字前缀 → 保留旧行为。
        """
        import re
        m = re.match(r"(\d+)", filename_stem)
        if m:
            raw_prefix = m.group(1)
            num = str(int(raw_prefix))
            # 短数字前缀（≤3位有效数字）+ 后面有实质内容 → 很可能是自动生成的文件名
            # 例如 "1-s2.0-S1369702123003243-main" → 用完整 stem 而非 "1"
            if len(num) <= 3 and len(filename_stem) > len(raw_prefix) + 1:
                return filename_stem
            return num
        # Then try paper-NNN pattern
        m = re.search(r"paper[_\-\s]*(\d+)", filename_stem, re.I)
        if m:
            return str(int(m.group(1)))
        # Fallback: use the filename stem as ID
        return filename_stem

    @staticmethod
    def _chunk_markdown(text: str) -> list[tuple[str, str]]:
        """将 markdown 笔记按 ## 章节切分，返回 (section_type, text)"""
        chunks = []
        current_section = "header"
        current_text = ""

        for line in text.split("\n"):
            if line.startswith("## "):
                if current_text.strip():
                    chunks.append((current_section, current_text.strip()))
                current_section = line[3:].strip()
                current_text = ""
            elif line.startswith("# "):
                if current_text.strip():
                    chunks.append((current_section, current_text.strip()))
                current_section = "title"
                current_text = ""
            else:
                current_text += line + "\n"
        if current_text.strip():
            chunks.append((current_section, current_text.strip()))
        return chunks

    @staticmethod
    def _extract_pdf_text(pdf_path: Path) -> str:
        """从 PDF 提取纯文本"""
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        texts = []
        for page in doc:
            t = page.get_text()
            if t.strip():
                texts.append(t.strip())
        doc.close()
        return "\n\n".join(texts)

    @staticmethod
    def _extract_docx_text(docx_path: Path) -> str:
        """从 DOCX 提取纯文本。优先 python-docx，回退 zipfile+xml"""
        try:
            import docx
            doc = docx.Document(str(docx_path))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
        except ImportError:
            pass
        # Fallback: parse docx as zip + xml
        import zipfile
        from xml.etree import ElementTree
        text_parts = []
        with zipfile.ZipFile(str(docx_path), "r") as z:
            if "word/document.xml" not in z.namelist():
                raise ValueError("Not a valid DOCX file")
            xml_content = z.read("word/document.xml")
            root = ElementTree.fromstring(xml_content)
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            for p in root.iterfind(".//w:p", ns):
                line = "".join(
                    t.text or "" for t in p.iterfind(".//w:t", ns)
                )
                if line.strip():
                    text_parts.append(line.strip())
        return "\n\n".join(text_parts)

    @staticmethod
    def _chunk_text(text: str, max_chars: int = 500) -> list[str]:
        """将纯文本按段落切分，超长段落按句子拆分"""
        paragraphs = text.split("\n\n")
        chunks = []
        for para in paragraphs:
            para = para.strip()
            if len(para) <= max_chars:
                if para:
                    chunks.append(para)
            else:
                # Split long paragraphs by sentences
                sentences = para.replace("。", "。\n").replace(". ", ".\n").split("\n")
                buf = ""
                for sent in sentences:
                    if len(buf) + len(sent) > max_chars and buf:
                        chunks.append(buf.strip())
                        buf = sent
                    else:
                        buf += sent + " "
                if buf.strip():
                    chunks.append(buf.strip())
        return chunks

    @staticmethod
    def _chunk_text_with_sections(
        sections: list[tuple[str, str]], max_chars: int = 500
    ) -> list[tuple[str, str]]:
        """Like _chunk_text but preserves section_type for each chunk."""
        result: list[tuple[str, str]] = []
        for sec_type, sec_text in sections:
            for chunk in VectorStore._chunk_text(sec_text, max_chars):
                result.append((sec_type, chunk))
        return result


    @staticmethod
    def _detect_pdf_sections(full_text: str) -> list[tuple[str, str]]:
        """检测 PDF 文本中的章节边界，返回 [(section_type, section_text), ...]"""
        import re
        _SECTION_PATTERN = re.compile(
            r'^(\d+\.?\s*)?'
            r'(Abstract|摘要|'
            r'Introduction|引言|INTRODUCTION|'
            r'Experimental|实验|Methods?|方法|'
            r'Results?\s*(and|\&)\s*Discussion|结果与讨论|'
            r'Results?|结果|'
            r'Discussion|讨论|'
            r'Conclusion|结论|CONCLUSIONS?|'
            r'Background|背景|'
            r'Materials?|材料)'
            r'\s*$',
            re.IGNORECASE,
        )

        def _normalize_sec(name: str) -> str:
            n = name.strip().lower()
            if any(w in n for w in ['abstract', '摘要']):
                return '摘要'
            if any(w in n for w in ['introduction', '引言', 'background', '背景']):
                return '引言'
            if any(w in n for w in ['experimental', '实验', 'method', '方法', 'materials', '材料']):
                return '实验方法'
            if any(w in n for w in ['results and discussion', '结果与讨论']):
                return '结果与讨论'
            if any(w in n for w in ['discussion', '讨论']) and 'results' not in n:
                return '讨论'
            if any(w in n for w in ['results', '结果']) and 'discussion' not in n:
                return '结果'
            if any(w in n for w in ['conclusion', '结论']):
                return '结论'
            return name.strip()

        lines = full_text.split('\n')
        sections: list[tuple[str, str]] = []
        current_section = 'body'
        current_text: list[str] = []

        for line in lines:
            stripped = line.strip()
            m = _SECTION_PATTERN.match(stripped)
            if m and len(stripped) < 80:
                if current_text:
                    text = '\n'.join(current_text).strip()
                    if text:
                        sections.append((_normalize_sec(current_section), text))
                current_section = m.group(0) or stripped
                current_text = []
            else:
                current_text.append(line)

        if current_text:
            text = '\n'.join(current_text).strip()
            if text:
                sections.append((_normalize_sec(current_section), text))

        if not sections or (len(sections) == 1 and sections[0][0] == 'body'):
            return [('body', full_text)]

        return sections

    @staticmethod
    def _build_paper_outline_chunk(sections: list[tuple[str, str]]) -> str:
        """从章节列表生成论文结构概要 chunk。每个章节取前 120 字。"""
        if len(sections) <= 1:
            return ""
        parts = ["【论文结构概要】"]
        for sec_name, sec_text in sections:
            preview = sec_text[:120].replace('\n', ' ').strip()
            if preview:
                parts.append(f"- {sec_name}: {preview}...")
        return '\n'.join(parts) if len(parts) > 1 else ""


    def import_single_paper(
        self, file_path: Path,
        progress_callback: Callable | None = None,
    ) -> dict:
        """导入单篇论文：SHA256 去重 → 文本提取 → 分块 → 嵌入 → 写入。

        三层去重:
        1. SHA256 命中 → 相同文件已索引，只记录文件名
        2. 向量化后内容比对 → 文件名不同但内容相同 → 删多余向量
        3. doc_id 冲突 → 旧 chunks 先删再写
        """
        # 线程安全：确保 ChromaDB 在可能没有 event loop 的线程中也能工作
        import asyncio as _asyncio
        try:
            _asyncio.get_running_loop()
        except RuntimeError:
            _asyncio.set_event_loop(_asyncio.new_event_loop())

        ext = file_path.suffix.lower()
        doc_id = self._extract_doc_id(file_path.stem)
        source_type = ext.lstrip(".")

        def _emit(phase, message, progress=None):
            if progress_callback:
                progress_callback(phase, message, progress)

        # ── 第 1 层：文件哈希去重 ──
        identity = _compute_file_identity(file_path)
        identities = _load_paper_identities()
        existing = identities.get(identity["sha256"])

        if existing:
            # 完全相同的内容已索引，只加文件名
            filenames = existing.get("filenames", [])
            if file_path.name not in filenames:
                filenames.append(file_path.name)
            existing["filenames"] = filenames
            _save_paper_identities(identities)
            _emit("done", f"已存在: {existing['chunk_count']} chunks (SHA256 命中)", 1.0)
            return {
                "success": True,
                "doc_id": existing.get("doc_ids", [doc_id])[0] if existing.get("doc_ids") else doc_id,
                "chunks": existing.get("chunk_count", 0),
                "source": "cached",
                "filename": file_path.name,
                "duplicate": True,
                "existing_filenames": filenames,
                "size": identity["size"],
            }

        # ── 文本提取 ──
        _emit("extracting", f"正在解析 {file_path.name}...")
        try:
            if ext == ".pdf":
                full_text = self._extract_pdf_text(file_path)
            elif ext == ".docx":
                full_text = self._extract_docx_text(file_path)
            elif ext in (".txt", ".md"):
                full_text = file_path.read_text(encoding="utf-8")
            else:
                return {"success": False, "error": f"不支持的文件类型: {ext}"}
        except Exception as e:
            return {"success": False, "error": f"文本提取失败: {e}"}

        if not full_text.strip():
            return {"success": False, "error": "文件无有效文本内容"}

        # ── 第 2 层：内容相似度去重 ──
        similar_sha = self._find_content_duplicate(full_text, identity["size"], identities)
        if similar_sha:
            existing = identities[similar_sha]
            filenames = existing.get("filenames", [])
            if file_path.name not in filenames:
                filenames.append(file_path.name)
            existing["filenames"] = filenames
            _save_paper_identities(identities)
            _emit("done", f"内容重复(相似度匹配): {existing['chunk_count']} chunks 已存在", 1.0)
            return {
                "success": True,
                "doc_id": existing.get("doc_ids", [doc_id])[0] if existing.get("doc_ids") else doc_id,
                "chunks": existing.get("chunk_count", 0),
                "source": "cached",
                "filename": file_path.name,
                "duplicate": True,
                "matched_by": "content_similarity",
                "existing_filenames": filenames,
                "size": identity["size"],
            }

        # ── 分块 ──
        _emit("chunking", "正在分块...")
        if ext == ".md":
            chunks = self._chunk_markdown(full_text)
        else:
            text_chunks = self._chunk_text(full_text)
            chunks = [("paragraph", c) for c in text_chunks]

        valid_chunks = [(st, c) for st, c in chunks if len(c.strip()) >= 30]
        if not valid_chunks:
            return {"success": False, "error": "分块后无有效内容"}

        # ── 第 3 层：doc_id 冲突 → 删旧 ──
        collection = self._get_collection()
        existing_for_doc = collection.get(
            where={"doc_id": doc_id},
            include=["metadatas"],
        )
        if existing_for_doc["ids"]:
            collection.delete(ids=existing_for_doc["ids"])
            _emit("chunking", f"清理旧索引: {len(existing_for_doc['ids'])} chunks")

        # ── 嵌入 ──
        _emit("embedding", f"正在生成嵌入 ({len(valid_chunks)} chunks)...")
        if not embedding_engine.is_available:
            embedding_engine.load()
        if not embedding_engine.is_available:
            return {"success": False, "error": "嵌入模型不可用"}

        # 分批嵌入（与 build_index 一致）
        batch_size = 4
        all_ids, all_docs, all_metas, all_embeddings = [], [], [], []
        chunk_texts = [c for _, c in valid_chunks]
        for batch_start in range(0, len(valid_chunks), batch_size):
            batch_end = min(batch_start + batch_size, len(valid_chunks))
            batch_docs = chunk_texts[batch_start:batch_end]
            batch_embeddings = embedding_engine.embed(batch_docs)

            for j, (section_type, chunk_text) in enumerate(valid_chunks[batch_start:batch_end]):
                i = batch_start + j
                all_ids.append(f"import_{doc_id}_{i}")
                all_docs.append(chunk_text)
                all_metas.append({
                    "doc_id": doc_id,
                    "source": source_type,
                    "section_type": section_type,
                    "chunk_index": i,
                    "filename": file_path.name,
                })
                all_embeddings.append(batch_embeddings[j])

        _emit("writing", f"正在写入向量库 ({len(all_embeddings)} vectors)...")
        try:
            collection.add(ids=all_ids, documents=all_docs, metadatas=all_metas, embeddings=all_embeddings)
            self._indexed = True
        except Exception as e:
            return {"success": False, "error": f"写入向量库失败: {e}"}

        # ── 记录身份 ──
        content_fp = self._compute_content_fingerprint(full_text, identity["size"])
        identities[identity["sha256"]] = {
            "size": identity["size"],
            "filenames": [file_path.name],
            "doc_ids": [doc_id],
            "chunk_count": len(valid_chunks),
            "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "content_fingerprint": content_fp,
        }
        _save_paper_identities(identities)

        _emit("done", f"导入完成: {len(valid_chunks)} chunks", 1.0)
        return {
            "success": True,
            "doc_id": doc_id,
            "chunks": len(valid_chunks),
            "source": source_type,
            "filename": file_path.name,
            "size": identity["size"],
            "sha256": identity["sha256"][:12],
        }

    @staticmethod
    def _compute_content_fingerprint(text: str, file_size: int) -> str:
        """计算内容指纹：前 2000 字 × 文本总长 × 后 500 字。"""
        h = hashlib.sha256()
        h.update(text[:2000].encode("utf-8", errors="replace"))
        h.update(str(len(text)).encode())
        h.update(text[-500:].encode("utf-8", errors="replace"))
        h.update(str(file_size).encode())
        return h.hexdigest()

    @staticmethod
    def _find_content_duplicate(
        text: str, file_size: int, identities: dict,
    ) -> str | None:
        """用内容指纹查找是否已有相同内容的论文被索引。

        指纹 = SHA256(前2000字 + 文本总长 + 后500字 + 文件大小)
        文本内容相同 → 指纹相同 → 无论文件名怎么变都能识别。
        """
        fingerprint = VectorStore._compute_content_fingerprint(text, file_size)
        for sha, info in identities.items():
            if info.get("content_fingerprint") == fingerprint:
                return sha
        return None

    @property
    def chunk_count(self) -> int:
        try:
            return self._get_collection().count()
        except Exception:
            return 0


# Singleton
vector_store = VectorStore()
