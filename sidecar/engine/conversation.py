"""Conversation data model for the V10 workflow engine."""
from dataclasses import dataclass, field


@dataclass
class Conversation:
    """Session conversation — tracks papers, messages, and topic."""
    id: str
    name: str
    open_papers: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    current_topic: str = ""
    is_first_message: bool = True
    last_question: str = ""
    last_question_time: float = 0.0
    classify_fail_count: int = 0
    updated_at: str = ""

    def add_paper(self, pid: int | str, title: str, filename: str = "", filepath: str = ""):
        pid_int = int(pid) if isinstance(pid, str) and pid.isdigit() else pid
        if not any(p["paper_id"] == pid_int for p in self.open_papers):
            # 从文件名推导 ChromaDB doc_id（与 vector_store._extract_doc_id 一致）
            from pathlib import Path
            import re
            fname = filename or (Path(filepath).name if filepath else "")
            stem = Path(fname).stem if fname else ""
            if stem:
                m = re.match(r"(\d+)", stem)
                chroma_id = str(int(m.group(1))) if m else stem
            else:
                chroma_id = str(pid_int)
            self.open_papers.append({
                "paper_id": pid_int,
                "title": title,
                "filename": filename,
                "filepath": filepath,
                "chroma_doc_id": chroma_id,
            })
            return True
        return False

    def paper_ids(self) -> list[str] | None:
        """返回用于 ChromaDB 范围搜索的 doc_id 列表。优先使用 chroma_doc_id。"""
        ids = [str(p.get("chroma_doc_id", p["paper_id"])) for p in self.open_papers]
        return ids if ids else None

    def paper_titles_str(self) -> str:
        if not self.open_papers:
            return ""
        return "\n".join(
            f"  #{p['paper_id']}: {p['title'][:60]}" for p in self.open_papers
        )
