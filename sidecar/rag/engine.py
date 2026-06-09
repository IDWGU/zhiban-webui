"""
Hybrid RAG 引擎 — 向量检索 + 知识图谱遍历，两路并行后合并排序
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

from .. import config
from .vector_store import vector_store
from .graph_store import graph_store

_executor = ThreadPoolExecutor(max_workers=4)


def _get_doc_id(meta: dict) -> str:
    """从 ChromaDB metadata 中提取文档 ID，兼容新旧字段名"""
    return str(meta.get("doc_id", meta.get("paper_id", "?")))


class RAGEngine:
    """混合检索引擎"""

    async def search(
        self, query: str, context_doc_ids: list[str] | None = None, top_k: int = 5
    ) -> list[dict]:
        """
        混合检索：
        1. 向量检索 (全库 or 限定论文)
        2. 知识图谱邻居展开
        返回 Top-K 去重排序结果
        """
        # Parallel: vector search + graph neighbors
        # Ensure candidate pool >= requested top_k so final merge isn't starved
        vector_candidate_k = max(top_k * 2, config.VECTOR_TOP_K)
        vector_task = asyncio.get_event_loop().run_in_executor(
            _executor,
            lambda: vector_store.search(
                query,
                top_k=vector_candidate_k,
            ) if not context_doc_ids else vector_store.search_by_doc_ids(
                query, context_doc_ids, top_k=vector_candidate_k,
            ),
        )

        # Graph: if we have doc IDs, get their neighbors
        graph_task = None
        if context_doc_ids:
            graph_task = asyncio.get_event_loop().run_in_executor(
                _executor,
                lambda: self._get_graph_context(context_doc_ids),
            )

        vector_results = await vector_task
        graph_results = await graph_task if graph_task else []

        # Merge & deduplicate
        merged = self._merge_results(vector_results, graph_results)
        return merged[:top_k]

    def _get_graph_context(self, paper_ids: list[int]) -> list[dict]:
        """从知识图谱获取关联论文的上下文"""
        results = []
        seen = set()

        for pid in paper_ids:
            paper = graph_store.get_paper_info(pid)
            if paper and pid not in seen:
                seen.add(pid)
                results.append({
                    "source": "graph_self",
                    "paper_id": pid,
                    "content": f"#{pid} {paper['title']} ({paper['year']}) — {', '.join(paper.get('direction', []))}",
                    "score": 1.0,
                })

            neighbors = graph_store.get_neighbors(pid, hops=config.GRAPH_HOPS)
            for nb in neighbors:
                nid = nb["paper_id"]
                if nid not in seen:
                    seen.add(nid)
                    results.append({
                        "source": f"graph_{nb['relation']}",
                        "paper_id": nid,
                        "content": (
                            f"#{nid} {nb.get('title', '')} ({nb.get('year', '')}) "
                            f"[{nb['relation']}] {nb.get('evidence', '')}"
                        ),
                        "score": 0.85,
                        "graph_relation": nb["relation"],
                        "graph_from": nb.get("from_paper"),
                    })

        return results

    @staticmethod
    def _merge_results(vector_results: list[dict], graph_results: list[dict]) -> list[dict]:
        """合并向量和图谱结果，简单加权去重"""
        merged: dict[str, dict] = {}

        for r in vector_results:
            pid = _get_doc_id(r.get("metadata", {}))
            key = f"v_{r.get('id', pid)}"
            merged[key] = {**r, "final_score": r["score"]}

        for r in graph_results:
            pid = r.get("doc_id", r.get("paper_id", 0))
            key = f"g_{pid}"
            if key in merged:
                merged[key]["final_score"] = max(merged[key]["final_score"], r["score"])
                merged[key]["graph_relation"] = r.get("graph_relation")
            else:
                merged[key] = {**r, "final_score": r["score"] * 0.8}

        sorted_results = sorted(merged.values(), key=lambda x: x["final_score"], reverse=True)
        return sorted_results

    def build_context_prompt(self, results: list[dict], max_chars: int = 3000, count: int | None = None) -> str:
        """将检索结果组装为 prompt 上下文"""
        parts = []
        total_chars = 0
        for i, r in enumerate(results[:count or config.FINAL_TOP_K]):
            content = r.get("content", "")
            meta = r.get("metadata", {})
            doc_id = _get_doc_id(meta)

            header = f"[{i + 1}] Doc #{doc_id} "
            if meta.get("section_type"):
                header += f"— {meta['section_type']}"
            header += f" (相关度: {r['final_score']:.2f})"

            chunk = f"{header}\n{content}"
            if total_chars + len(chunk) > max_chars:
                chunk = chunk[:max_chars - total_chars]
                parts.append(chunk)
                break

            parts.append(chunk)
            total_chars += len(chunk)

        return "\n\n---\n\n".join(parts)


# Singleton
rag_engine = RAGEngine()
