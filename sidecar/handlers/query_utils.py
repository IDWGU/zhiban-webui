"""Shared utilities for WebSocket message handlers."""
import re

from ..rag.graph_store import graph_store
from ..rag.vector_store import vector_store


def sanitize_api_key(key: str) -> str:
    """Remove non-ASCII characters from API key (e.g. invisible Unicode from copy-paste)."""
    if not key:
        return key
    return key.encode("ascii", errors="ignore").decode("ascii").strip()


def get_doc_id(meta: dict) -> str:
    """Extract paper ID from metadata, compatible with old/new field names."""
    return str(meta.get("doc_id", meta.get("paper_id", "?")))


def extract_doc_ids(doc_name: str | None) -> list[str]:
    """Extract paper IDs from a document name string."""
    if not doc_name:
        return []
    ids = []
    for m in re.finditer(r"paper[_\-\s]*(\d+)", doc_name, re.I):
        ids.append(str(int(m.group(1))))
    for m in re.finditer(r"#(\d+)", doc_name):
        ids.append(str(int(m.group(1))))
    # Fallback: fuzzy match by paper title
    if not ids and doc_name:
        ids = [str(pid) for pid in graph_store.search_by_title(doc_name)]
    return ids


def parse_citation_refs(text: str) -> list[dict]:
    """Parse @Paper#N(section): "quoted text" citation references from user message.

    返回 [{paper_id, section, quoted_text}]。
    - section: 可选，如 "abstract"、"methods"
    - quoted_text: 可选，用户引用的 AI 回复原文
    """
    refs = []
    seen = set()
    for m in re.finditer(
        r"@Paper\s*#(\d+)"          # @Paper#N
        r"(?:\(([^)]*)\))?"          # 可选 (section)
        r"(?:\s*:\s*[\"“]([^\"”]+)[\"”])?",  # 可选 : "quoted text"
        text
    ):
        pid = str(int(m.group(1)))
        section = (m.group(2) or "").strip()
        quoted = (m.group(3) or "").strip()
        key = f"{pid}:{section}"
        if key in seen:
            continue
        seen.add(key)
        refs.append({"paper_id": pid, "section": section, "quoted_text": quoted})
    return refs


def build_citation_context(refs: list[dict], max_chars: int = 1200) -> str:
    """根据引用列表构建上下文注入 LLM prompt。

    优先使用用户提供的 quoted_text（直接从 AI 回复中引用的原文），
    否则从 ChromaDB 检索对应 chunk。
    """
    if not refs:
        return ""

    parts = []
    for ref in refs[:3]:
        pid = ref["paper_id"]
        section = ref.get("section", "")
        quoted = ref.get("quoted_text", "")

        # 获取 paper 标题
        title = ""
        try:
            info = graph_store.get_paper_info(int(pid))
            if info and info.get("title"):
                title = info["title"]
        except (ValueError, TypeError):
            pass

        # 用户提供了引用原文 —— 直接使用
        if quoted:
            sec_label = f" · {section}" if section else ""
            title_label = f" · {title}" if title else ""
            parts.append(
                f"【用户针对以下内容追问（Paper#{pid}{sec_label}{title_label}）】\n"
                f"AI 回复原文: \"{quoted}\"\n"
                f"请针对这段话回答用户的问题。"
            )
            continue

        # 无引用原文 —— 从 ChromaDB 检索
        try:
            results = vector_store.search_by_doc_ids(
                [pid], top_k=5,
                where={"doc_id": pid} if section else None,
            )
        except Exception:
            results = []

        matching = []
        for r in results:
            meta = r.get("metadata", {}) if isinstance(r, dict) else {}
            if section and section.lower() in (meta.get("section_type", "") or "").lower():
                matching.append(r)
        if not matching:
            matching = results[:2]

        if matching:
            meta = matching[0].get("metadata", {}) if isinstance(matching[0], dict) else {}
            chunk_text = matching[0].get("content", matching[0].get("doc", "")) if isinstance(matching[0], dict) else str(matching[0])
            fname = meta.get("filename", "")
            sec = meta.get("section_type", section or "?")
            label = title or fname
            parts.append(
                f"【用户引用 Paper#{pid} · {sec} · {label}】\n{chunk_text[:max_chars//3]}"
            )
        elif section:
            parts.append(f"【用户引用 Paper#{pid} · {section}】（未找到对应段落）")
        else:
            parts.append(f"【用户引用 Paper#{pid}】（未找到对应段落）")

    return "\n\n---\n".join(parts) if parts else ""


def build_rag_fallback(query: str, results: list[dict], top_k: int = 5,
                       screen_doc: str = "", screen_text: str = "") -> str:
    """Build RAG fallback text when no LLM API key is configured."""
    lines = []

    if screen_text:
        doc_label = screen_doc or "unknown"
        lines.append(f"**📖 Current Reading** (PaperViewer)\n> Doc: {doc_label}\n> Content: {screen_text}\n")

    lines.append("")

    if not results:
        lines.append("No relevant content found in knowledge base.")
        lines.append("\n---\n*LLM not configured (set API Key to use AI-generated responses)*")
        return "\n".join(lines)

    lines.append("---")
    lines.append(f"**Knowledge Base Results** (query: {query})\n")
    for i, r in enumerate(results[:top_k]):
        meta = r.get("metadata", {})
        doc_id = get_doc_id(meta)
        section = meta.get("section_type", r.get("source", ""))
        score = r.get("final_score", r.get("score", 0))
        content = r.get("content", "")[:300]
        lines.append(f"**[{i+1}] Doc #{doc_id}** — {section} (relevance: {score:.2f})")
        lines.append(f"{content}\n")
    lines.append("\n---\n*LLM not configured (set API Key to use AI-generated responses)*")
    return "\n".join(lines)


def build_citations(results: list[dict]) -> list[dict]:
    """Build citation list from search results."""
    citations = []
    for i, r in enumerate(results[:3]):
        meta = r.get("metadata", {})
        doc_id = get_doc_id(meta)
        citations.append({
            "index": i + 1,
            "paperId": doc_id,
            "title": meta.get("filename", f"Doc #{doc_id}"),
            "chunkText": r.get("content", "")[:200],
            "sectionType": meta.get("section_type", r.get("source", "")),
        })
    return citations
