"""知识图谱存储 — 基于 NetworkX 加载 knowledge-graph.yaml"""

import yaml
import networkx as nx
from pathlib import Path
from typing import Optional

from .. import config


class GraphStore:
    """论文知识图谱"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.papers: dict[int, dict] = {}  # paper_id → metadata
        self._loaded = False

    def load(self, path: Optional[Path] = None):
        """加载 YAML 知识图谱"""
        path = path or config.KNOWLEDGE_GRAPH
        if not path.exists():
            return

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        for paper in data.get("papers", []):
            pid = paper["id"]
            self.papers[pid] = {
                "title": paper.get("title", ""),
                "year": paper.get("year", 0),
                "journal": paper.get("journal", ""),
                "direction": paper.get("direction", []),
                "core_members": paper.get("core_members", []),
                "tags": paper.get("tags", []),
            }
            self.graph.add_node(pid, **self.papers[pid])

            for rel in paper.get("relations", []):
                target = rel["target"]
                rel_type = rel["type"]
                evidence = rel.get("evidence", "")
                self.graph.add_edge(pid, target, type=rel_type, evidence=evidence)

        self._loaded = True

    def _ensure_loaded(self):
        if not self._loaded:
            try:
                self.load()
            except Exception:
                self._loaded = True

    def get_neighbors(self, paper_id: int, hops: int = 2) -> list[dict]:
        """获取论文的知识图谱邻居（含关系类型和evidence）"""
        self._ensure_loaded()
        if paper_id not in self.graph:
            return []

        neighbors = []
        visited = {paper_id}
        frontier = {paper_id}

        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                for _, neighbor, edge_data in self.graph.out_edges(node, data=True):  # pyright: ignore[reportArgumentType]
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
                    neighbors.append({
                        "paper_id": neighbor,
                        "relation": edge_data["type"],
                        "direction": "out",
                        "from_paper": node,
                        "evidence": edge_data.get("evidence", "")[:200],
                        **self.papers.get(neighbor, {}),
                    })
                for predecessor, _, edge_data in self.graph.in_edges(node, data=True):
                    if predecessor in visited:
                        continue
                    visited.add(predecessor)
                    next_frontier.add(predecessor)
                    neighbors.append({
                        "paper_id": predecessor,
                        "relation": edge_data["type"],
                        "direction": "in",
                        "from_paper": node,
                        "evidence": edge_data.get("evidence", "")[:200],
                        **self.papers.get(predecessor, {}),
                    })
            frontier = next_frontier

        return neighbors

    def get_paper_info(self, paper_id: int) -> dict | None:
        """获取单篇论文元数据"""
        self._ensure_loaded()
        return self.papers.get(paper_id)

    def search_by_direction(self, direction: str) -> list[int]:
        """按研究方向筛选论文ID"""
        self._ensure_loaded()
        return [pid for pid, p in self.papers.items() if direction in p.get("direction", [])]

    def search_by_title(self, title: str) -> list[int]:
        """按标题模糊搜索论文ID，不区分大小写"""
        self._ensure_loaded()
        title_lower = title.lower()
        return [
            pid for pid, p in self.papers.items()
            if title_lower in p.get("title", "").lower()
            or p.get("title", "").lower() in title_lower
        ]

    def get_all_paper_ids(self) -> list[int]:
        self._ensure_loaded()
        return list(self.papers.keys())

    @property
    def paper_count(self) -> int:
        self._ensure_loaded()
        return len(self.papers)

    @property
    def edge_count(self) -> int:
        self._ensure_loaded()
        return self.graph.number_of_edges()


# Singleton
graph_store = GraphStore()
